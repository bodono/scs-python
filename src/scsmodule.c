#include <Python.h>
#include "amatrix.h"
#include "cones.h"
#include "glbopts.h"
#include "numpy/arrayobject.h"
#include "scs.h"
#include "util.h"

/* IMPORTANT: This code now uses numpy array types. It is a private C module
 * in the sense that end users only see the front-facing Python code in
 * "scs.py"; hence, we can get away with the inputs being numpy arrays of
 * the CSC data structures.
 *
 * WARNING: This code also does not check that the data for the sparse
 * matrices are *actually* in column compressed storage for a sparse matrix.
 * The C module is not designed to be used stand-alone. If the data provided
 * does not correspond to a CSR matrix, this code will just crash inelegantly.
 * Please use the "solve" interface in scs.py.
 */

/* The PyInt variable is a PyLong in Python3.x.
 */
#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
#endif

struct ScsPyData {
  PyArrayObject *Ax;
  PyArrayObject *Ai;
  PyArrayObject *Ap;
  PyArrayObject *b;
  PyArrayObject *c;
};


PyObject *scs_init_lin_sys_work_cb = SCS_NULL;
PyObject *scs_solve_lin_sys_cb = SCS_NULL;
PyObject *scs_accum_by_a_cb = SCS_NULL;
PyObject *scs_accum_by_atrans_cb = SCS_NULL;
PyObject *scs_normalize_a_cb = SCS_NULL;
PyObject *scs_un_normalize_a_cb = SCS_NULL;

/* Note, Python3.x may require special handling for the scs_int and scs_float
 * types. */
int scs_get_int_type(void) {
  switch (sizeof(scs_int)) {
    case 1:
      return NPY_INT8;
    case 2:
      return NPY_INT16;
    case 4:
      return NPY_INT32;
    case 8:
      return NPY_INT64;
    default:
      return NPY_INT32; /* defaults to 4 byte int */
  }
}

int scs_get_float_type(void) {
  switch (sizeof(scs_float)) {
    case 2:
      return NPY_FLOAT16;
    case 4:
      return NPY_FLOAT32;
    case 8:
      return NPY_FLOAT64;
    default:
      return NPY_FLOAT64; /* defaults to double */
  }
}

PyArrayObject *scs_get_contiguous(PyArrayObject *array, int typenum) {
  /* gets the pointer to the block of contiguous C memory */
  /* the overhead should be small unless the numpy array has been */
  /* reordered in some way or the data type doesn't quite match */
  /* */
  /* the "new_owner" pointer has to have Py_DECREF called on it; it owns */
  /* the "new" array object created by PyArray_Cast */
  /* */
  PyArrayObject *tmp_arr;
  PyArrayObject *new_owner;
  tmp_arr = PyArray_GETCONTIGUOUS(array);
  new_owner = (PyArrayObject *)PyArray_Cast(tmp_arr, typenum);
  Py_DECREF(tmp_arr);
  return new_owner;
}

static int printErr(char *key) {
  PySys_WriteStderr("error parsing '%s'\n", key);
  return -1;
}

/* returns -1 for failure */
static int parse_pos_scs_int(PyObject *in, scs_int *out) {
  if (PyInt_Check(in)) {
    *out = (scs_int)PyInt_AsLong(in);
  } else if (PyLong_Check(in)) {
    *out = (scs_int)PyLong_AsLong(in);
  } else {
    return -1;
  }
  return *out >= 0 ? 1 : -1;
}

static int get_pos_int_param(char *key, scs_int *v, scs_int defVal,
                             PyObject *opts) {
  *v = defVal;
  if (opts) {
    PyObject *obj = PyDict_GetItemString(opts, key);
    if (obj) {
      if (parse_pos_scs_int(obj, v) < 0) {
        return printErr(key);
      }
    }
  }
  return 0;
}

/* gets warm starts from warm dict, doesn't destroy input warm start data */
static scs_int get_warm_start(char *key, scs_float **x, scs_int l,
                              PyObject *warm) {
  PyArrayObject *x0 = (PyArrayObject *)PyDict_GetItemString(warm, key);
  *x = (scs_float *)scs_calloc(l, sizeof(scs_float));
  if (x0) {
    if (!PyArray_ISFLOAT(x0) || PyArray_NDIM(x0) != 1 ||
        PyArray_DIM(x0, 0) != l) {
      PySys_WriteStderr("Error parsing warm-start input\n");
      return 0;
    } else {
      PyArrayObject *px0 = scs_get_contiguous(x0, scs_get_float_type());
      memcpy(*x, (scs_float *)PyArray_DATA(px0), l * sizeof(scs_float));
      Py_DECREF(px0);
      return 1;
    }
  }
  return 0;
}

static int get_cone_arr_dim(char *key, scs_int **varr, scs_int *vsize,
                            PyObject *cone) {
  /* get cone['key'] */
  scs_int i, n = 0;
  scs_int *q = SCS_NULL;
  PyObject *obj = PyDict_GetItemString(cone, key);
  if (obj) {
    if (PyList_Check(obj)) {
      n = (scs_int)PyList_Size(obj);
      q = (scs_int *)scs_calloc(n, sizeof(scs_int));
      for (i = 0; i < n; ++i) {
        PyObject *qi = PyList_GetItem(obj, i);
        if (parse_pos_scs_int(qi, &(q[i])) < 0) {
          return printErr(key);
        }
      }
    } else if (PyInt_Check(obj) || PyLong_Check(obj)) {
      n = 1;
      q = (scs_int *)scs_malloc(sizeof(scs_int));
      if (parse_pos_scs_int(obj, q) < 0) {
        return printErr(key);
      }
    } else {
      return printErr(key);
    }
    if (PyErr_Occurred()) {
      /* potentially could have been triggered before */
      return printErr(key);
    }
  }
  *vsize = n;
  *varr = q;
  return 0;
}

static int get_cone_float_arr(char *key, scs_float **varr, scs_int *vsize,
                              PyObject *cone) {
  /* get cone['key'] */
  scs_int i, n = 0;
  scs_float *q = SCS_NULL;
  PyObject *obj = PyDict_GetItemString(cone, key);
  if (obj) {
    if (PyList_Check(obj)) {
      n = (scs_int)PyList_Size(obj);
      q = (scs_float *)scs_calloc(n, sizeof(scs_float));
      for (i = 0; i < n; ++i) {
        PyObject *qi = PyList_GetItem(obj, i);
        q[i] = (scs_float)PyFloat_AsDouble(qi);
      }
    } else if (PyInt_Check(obj) || PyLong_Check(obj) || PyFloat_Check(obj)) {
      n = 1;
      q = (scs_float *)scs_malloc(sizeof(scs_float));
      q[0] = (scs_float)PyFloat_AsDouble(obj);
    } else {
      return printErr(key);
    }
    if (PyErr_Occurred()) {
      /* potentially could have been triggered before */
      return printErr(key);
    }
  }
  *vsize = n;
  *varr = q;
  return 0;
}

static void free_py_scs_data(ScsData *d, ScsCone *k, struct ScsPyData *ps) {
  if (ps->Ax) {
    Py_DECREF(ps->Ax);
  }
  if (ps->Ai) {
    Py_DECREF(ps->Ai);
  }
  if (ps->Ap) {
    Py_DECREF(ps->Ap);
  }
  if (ps->b) {
    Py_DECREF(ps->b);
  }
  if (ps->c) {
    Py_DECREF(ps->c);
  }
  if (k) {
    if (k->q) {
      scs_free(k->q);
    }
    if (k->s) {
      scs_free(k->s);
    }
    if (k->p) {
      scs_free(k->p);
    }
    scs_free(k);
  }
  if (d) {
    if (d->A) {
      scs_free(d->A);
    }
    if (d->stgs) {
      scs_free(d->stgs);
    }
    scs_free(d);
  }
}

static PyObject *finish_with_error(ScsData *d, ScsCone *k, struct ScsPyData *ps,
                                   char *str) {
  PyErr_SetString(PyExc_ValueError, str);
  free_py_scs_data(d, k, ps);
  return SCS_NULL;
}

static PyObject *version(PyObject *self) {
  return Py_BuildValue("s", scs_version());
}

static PyObject *csolve(PyObject *self, PyObject *args, PyObject *kwargs) {
  /* data structures for arguments */
  PyArrayObject *Ax, *Ai, *Ap, *c, *b;
  PyObject *cone, *warm = SCS_NULL;
  PyObject *verbose = SCS_NULL;
  PyObject *normalize = SCS_NULL;
  /* get the typenum for the primitive scs_int and scs_float types */
  int scs_int_type = scs_get_int_type();
  int scs_float_type = scs_get_float_type();
  struct ScsPyData ps = {
      SCS_NULL, SCS_NULL, SCS_NULL, SCS_NULL, SCS_NULL,
  };
  /* scs data structures */
  ScsData *d = (ScsData *)scs_calloc(1, sizeof(ScsData));
  ScsCone *k = (ScsCone *)scs_calloc(1, sizeof(ScsCone));

  ScsMatrix *A;
  ScsSolution sol = {0};
  ScsInfo info;
  char *kwlist[] = {"shape",
                    "Ax",
                    "Ai",
                    "Ap",
                    "b",
                    "c",
                    "cone",
                    "warm",
                    "verbose",
                    "normalize",
                    "max_iters",
                    "scale",
                    "eps",
                    "cg_rate",
                    "alpha",
                    "rho_x",
                    "acceleration_lookback",
                    "write_data_filename",
                    "linsys_cbs",
                    SCS_NULL};

/* parse the arguments and ensure they are the correct type */
#ifdef DLONG
#ifdef SFLOAT
#ifdef PYTHON_LINSYS
#error // Unimplemented.
#endif
  char *argparse_string = "(ll)O!O!O!O!O!O!|O!O!O!lffffflz";
  char *outarg_string = "{s:l,s:l,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:s}";
#else
  char *argparse_string = "(ll)O!O!O!O!O!O!|O!O!O!ldddddlz(OOOOOO)";
  char *outarg_string = "{s:l,s:l,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:s}";
#endif
#else
#ifdef SFLOAT
#ifdef PYTHON_LINSYS
#error // Unimplemented.
#endif
  char *argparse_string = "(ii)O!O!O!O!O!O!|O!O!O!ifffffiz";
  char *outarg_string = "{s:i,s:i,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:s}";
#else
#ifdef PYTHON_LINSYS
#error // Unimplemented.
#endif
  char *argparse_string = "(ii)O!O!O!O!O!O!|O!O!O!idddddiz";
  char *outarg_string = "{s:i,s:i,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:s}";
#endif
#endif
  npy_intp veclen[1];
  PyObject *x, *y, *s, *return_dict, *info_dict;

  d->stgs = (ScsSettings *)scs_malloc(sizeof(ScsSettings));

  /* set defaults */
  SCS(set_default_settings)(d);

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, argparse_string, kwlist, &(d->m), &(d->n),
          &PyArray_Type, &Ax, &PyArray_Type, &Ai, &PyArray_Type, &Ap,
          &PyArray_Type, &b, &PyArray_Type, &c, &PyDict_Type, &cone,
          &PyDict_Type, &warm, &PyBool_Type, &verbose, &PyBool_Type, &normalize,
          &(d->stgs->max_iters), &(d->stgs->scale), &(d->stgs->eps),
          &(d->stgs->cg_rate), &(d->stgs->alpha), &(d->stgs->rho_x),
          &(d->stgs->acceleration_lookback),
          &(d->stgs->write_data_filename),
          &scs_init_lin_sys_work_cb, &scs_solve_lin_sys_cb,
          &scs_accum_by_a_cb, &scs_accum_by_atrans_cb,
          &scs_normalize_a_cb, &scs_un_normalize_a_cb)) {
    PySys_WriteStderr("error parsing inputs\n");
    return SCS_NULL;
  }

  if (d->m < 0) {
    PyErr_SetString(PyExc_ValueError, "m must be a positive integer");
    return SCS_NULL;
  }

  if (d->n < 0) {
    PyErr_SetString(PyExc_ValueError, "n must be a positive integer");
    return SCS_NULL;
  }

  /* set A */
  if (!PyArray_ISFLOAT(Ax) || PyArray_NDIM(Ax) != 1) {
    return finish_with_error(d, k, &ps, "Ax must be a numpy array of floats");
  }
  if (!PyArray_ISINTEGER(Ai) || PyArray_NDIM(Ai) != 1) {
    return finish_with_error(d, k, &ps, "Ai must be a numpy array of ints");
  }
  if (!PyArray_ISINTEGER(Ap) || PyArray_NDIM(Ap) != 1) {
    return finish_with_error(d, k, &ps, "Ap must be a numpy array of ints");
  }
  ps.Ax = scs_get_contiguous(Ax, scs_float_type);
  ps.Ai = scs_get_contiguous(Ai, scs_int_type);
  ps.Ap = scs_get_contiguous(Ap, scs_int_type);

  A = (ScsMatrix *)scs_malloc(sizeof(ScsMatrix));
  A->n = d->n;
  A->m = d->m;
  A->x = (scs_float *)PyArray_DATA(ps.Ax);
  A->i = (scs_int *)PyArray_DATA(ps.Ai);
  A->p = (scs_int *)PyArray_DATA(ps.Ap);
  d->A = A;
  /* set c */
  if (!PyArray_ISFLOAT(c) || PyArray_NDIM(c) != 1) {
    return finish_with_error(
        d, k, &ps, "c must be a dense numpy array with one dimension");
  }
  if (PyArray_DIM(c, 0) != d->n) {
    return finish_with_error(d, k, &ps, "c has incompatible dimension with A");
  }
  ps.c = scs_get_contiguous(c, scs_float_type);
  d->c = (scs_float *)PyArray_DATA(ps.c);
  /* set b */
  if (!PyArray_ISFLOAT(b) || PyArray_NDIM(b) != 1) {
    return finish_with_error(
        d, k, &ps, "b must be a dense numpy array with one dimension");
  }
  if (PyArray_DIM(b, 0) != d->m) {
    return finish_with_error(d, k, &ps, "b has incompatible dimension with A");
  }
  ps.b = scs_get_contiguous(b, scs_float_type);
  d->b = (scs_float *)PyArray_DATA(ps.b);

  if (get_pos_int_param("f", &(k->f), 0, cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field f");
  }
  if (get_pos_int_param("l", &(k->l), 0, cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field l");
  }
  if (get_cone_arr_dim("q", &(k->q), &(k->qsize), cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field q");
  }
  if (get_cone_arr_dim("s", &(k->s), &(k->ssize), cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field s");
  }
  if (get_cone_float_arr("p", &(k->p), &(k->psize), cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field p");
  }
  if (get_pos_int_param("ep", &(k->ep), 0, cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field ep");
  }
  if (get_pos_int_param("ed", &(k->ed), 0, cone) < 0) {
    return finish_with_error(d, k, &ps, "failed to parse cone field ed");
  }

  d->stgs->verbose = verbose ? (scs_int)PyObject_IsTrue(verbose) : VERBOSE;
  d->stgs->normalize =
      normalize ? (scs_int)PyObject_IsTrue(normalize) : NORMALIZE;

  if (d->stgs->max_iters < 0) {
    return finish_with_error(d, k, &ps, "max_iters must be positive");
  }
  if (d->stgs->acceleration_lookback < 0) {
    return finish_with_error(d, k, &ps,
                             "acceleration_lookback must be positive");
  }
  if (d->stgs->scale < 0) {
    return finish_with_error(d, k, &ps, "scale must be positive");
  }
  if (d->stgs->eps < 0) {
    return finish_with_error(d, k, &ps, "eps must be positive");
  }
  if (d->stgs->cg_rate < 0) {
    return finish_with_error(d, k, &ps, "cg_rate must be positive");
  }
  if (d->stgs->alpha < 0) {
    return finish_with_error(d, k, &ps, "alpha must be positive");
  }
  if (d->stgs->rho_x < 0) {
    return finish_with_error(d, k, &ps, "rho_x must be positive");
  }
  /* parse warm start if set */
  d->stgs->warm_start = WARM_START;
  if (warm) {
    d->stgs->warm_start = get_warm_start("x", &(sol.x), d->n, warm);
    d->stgs->warm_start |= get_warm_start("y", &(sol.y), d->m, warm);
    d->stgs->warm_start |= get_warm_start("s", &(sol.s), d->m, warm);
  }

#ifdef PYTHON_LINSYS
  if (!PyCallable_Check(scs_init_lin_sys_work_cb)) {
    PyErr_SetString(PyExc_ValueError, "scs_init_lin_sys_work_cb not a valid callback");
    return SCS_NULL;
  }

  if (!PyCallable_Check(scs_solve_lin_sys_cb)) {
    PyErr_SetString(PyExc_ValueError, "scs_solve_lin_sys_cb not a valid callback");
    return SCS_NULL;
  }

  if (!PyCallable_Check(scs_accum_by_a_cb)) {
    PyErr_SetString(PyExc_ValueError, "scs_accum_by_a_cb not a valid callback");
    return SCS_NULL;
  }

  if (!PyCallable_Check(scs_accum_by_atrans_cb)) {
    PyErr_SetString(PyExc_ValueError, "scs_accum_by_atrans_cb not a valid callback");
    return SCS_NULL;
  }

  if (!PyCallable_Check(scs_normalize_a_cb)) {
    PyErr_SetString(PyExc_ValueError, "scs_normalize_a_cb not a valid callback");
    return SCS_NULL;
  }

  if (!PyCallable_Check(scs_un_normalize_a_cb)) {
    PyErr_SetString(PyExc_ValueError, "scs_un_normalize_a_cb not a valid callback");
    return SCS_NULL;
  }
#endif


#ifndef PYTHON_LINSYS
  /* release the GIL */
  Py_BEGIN_ALLOW_THREADS;
#endif
  /* Solve! */
  scs(d, k, &sol, &info);
#ifndef PYTHON_LINSYS
  /* reacquire the GIL */
  Py_END_ALLOW_THREADS;
#endif

  veclen[0] = d->n;
  x = PyArray_SimpleNewFromData(1, veclen, scs_float_type, sol.x);
  PyArray_ENABLEFLAGS((PyArrayObject *)x, NPY_ARRAY_OWNDATA);

  veclen[0] = d->m;
  y = PyArray_SimpleNewFromData(1, veclen, scs_float_type, sol.y);
  PyArray_ENABLEFLAGS((PyArrayObject *)y, NPY_ARRAY_OWNDATA);

  veclen[0] = d->m;
  s = PyArray_SimpleNewFromData(1, veclen, scs_float_type, sol.s);
  PyArray_ENABLEFLAGS((PyArrayObject *)s, NPY_ARRAY_OWNDATA);

  info_dict = Py_BuildValue(
      outarg_string, "statusVal", (scs_int)info.status_val, "iter",
      (scs_int)info.iter, "pobj", (scs_float)info.pobj, "dobj",
      (scs_float)info.dobj, "resPri", (scs_float)info.res_pri, "resDual",
      (scs_float)info.res_dual, "relGap", (scs_float)info.rel_gap, "resInfeas",
      (scs_float)info.res_infeas, "resUnbdd", (scs_float)info.res_unbdd,
      "solveTime", (scs_float)(info.solve_time), "setupTime",
      (scs_float)(info.setup_time), "status", info.status);

  return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}", "x", x, "y", y, "s", s,
                              "info", info_dict);
  /* give up ownership to the return dictionary */
  Py_DECREF(x);
  Py_DECREF(y);
  Py_DECREF(s);
  Py_DECREF(info_dict);

  /* no longer need pointers to arrays that held primitives */
  free_py_scs_data(d, k, &ps);
  return return_dict;
}

static PyMethodDef scs_methods[] = {
    {"csolve", (PyCFunction)csolve, METH_VARARGS | METH_KEYWORDS,
     "Solve a convex cone problem using scs."},
    {"version", (PyCFunction)version, METH_NOARGS, "Version number for SCS."},
    {SCS_NULL, SCS_NULL, 0, SCS_NULL} /* sentinel */
};

/* Module initialization */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_scs",                                   /* m_name */
    "Solve a convex cone problem using SCS.", /* m_doc */
    -1,                                       /* m_size */
    scs_methods,                              /* m_methods */
    SCS_NULL,                                 /* m_reload */
    SCS_NULL,                                 /* m_traverse */
    SCS_NULL,                                 /* m_clear */
    SCS_NULL,                                 /* m_free */
};
#endif

static PyObject *moduleinit(void) {
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
#ifdef INDIRECT
  m = Py_InitModule("_scs_indirect", scs_methods);
#elif defined GPU
  m = Py_InitModule("_scs_gpu", scs_methods);
#elif defined PYTHON_LINSYS
  m = Py_InitModule("_scs_python", scs_methods);
#else
  m = Py_InitModule("_scs_direct", scs_methods);
#endif
#endif

  /*if (import_array() < 0) return SCS_NULL; // for numpy arrays */
  /*if (import_cvxopt() < 0) return SCS_NULL; // for cvxopt support */

  if (m == SCS_NULL) {
    return SCS_NULL;
  }
  return m;
};

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
#ifdef INDIRECT
PyInit__scs_indirect(void)
#elif defined GPU
PyInit__scs_gpu(void)
#elif defined PYTHON_LINSYS
PyInit__scs_python(void)
#else
PyInit__scs_direct(void)
#endif
{
  import_array(); /* for numpy arrays */
  return moduleinit();
}
#else
PyMODINIT_FUNC
#ifdef INDIRECT
init_scs_indirect(void)
#elif defined GPU
init_scs_gpu(void)
#elif defined PYTHON_LINSYS
init_scs_python(void)
#else
init_scs_direct(void)
#endif
{
  import_array(); /* for numpy arrays */
  moduleinit();
}
#endif
