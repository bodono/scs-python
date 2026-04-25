#include "pythoncapi-compat/pythoncapi_compat.h"

#ifndef PY_SCSOBJECT_H
#define PY_SCSOBJECT_H

/* SCS Object type */
typedef struct {
  PyObject_HEAD
  ScsWork *work; /* Workspace */
  ScsSolution *sol;            /* Solution, keep around for warm-starts */
  scs_int m, n;
  PyThread_type_lock lock;     /* Per-instance lock protecting work/sol */
} SCS;

/* Just a helper struct to store the PyArrayObjects that need Py_DECREF */
struct ScsPyData {
  PyArrayObject *Ax;
  PyArrayObject *Ai;
  PyArrayObject *Ap;
  PyArrayObject *Px;
  PyArrayObject *Pi;
  PyArrayObject *Pp;
  PyArrayObject *b;
  PyArrayObject *c;
};

/* Note, Python3.x may require special handling for the scs_int and scs_float
 * types. */
static int scs_get_int_type(void) {
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

static int scs_get_float_type(void) {
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

/* Returns a new strong reference (caller must Py_DECREF), or NULL with a
 * Python exception set on failure (OOM or cast error). The overhead is
 * small unless the input array has been reordered or its dtype differs. */
static PyArrayObject *scs_get_contiguous(PyArrayObject *array, int typenum) {
  PyArrayObject *tmp_arr = PyArray_GETCONTIGUOUS(array);
  if (!tmp_arr) {
    return NULL;
  }
  PyArrayObject *new_owner = (PyArrayObject *)PyArray_Cast(tmp_arr, typenum);
  Py_DECREF(tmp_arr);
  return new_owner;  /* NULL on Cast failure; exception already set */
}

/* Set a Python exception describing a cone-field parsing failure, then
 * return -1. The outer caller (SCS_init) can then just propagate -1 —
 * finish_with_error is no-clobber, so the specific message we set here
 * survives up to the user. */
static int printErr(char *key) {
  if (!PyErr_Occurred()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid value for cone field '%s'", key);
  }
  return -1;
}

/* returns 1 on success, -1 on failure. A failure leaves *out unchanged.
 * Rejects non-ints, overflow (Python int too large for scs_int), and
 * negative values. PyInt_Check is aliased to PyLong_Check in Python 3,
 * so a single PyLong_Check covers both. */
static int parse_pos_scs_int(PyObject *in, scs_int *out) {
  if (!PyLong_Check(in)) {
    return -1;
  }
  long long v = PyLong_AsLongLong(in);
  if (v == -1 && PyErr_Occurred()) {
    /* overflow or other error — clear so we can fall back cleanly */
    PyErr_Clear();
    return -1;
  }
  if (v < 0) {
    return -1;
  }
  /* Round-trip check catches silent downcast on 32-bit scs_int builds. */
  if ((long long)(scs_int)v != v) {
    return -1;
  }
  *out = (scs_int)v;
  return 1;
}

static int get_pos_int_param(char *key, scs_int *v, scs_int defVal,
                             PyObject *opts) {
  *v = defVal;
  if (opts) {
    PyObject *obj = NULL;
    int rc = PyDict_GetItemStringRef(opts, key, &obj);
    if (rc < 0) {
      return printErr(key);
    }
    if (obj) {
      if (parse_pos_scs_int(obj, v) < 0) {
        Py_DECREF(obj);
        return printErr(key);
      }
      Py_DECREF(obj);
    }
  }
  return 0;
}

/* If warm start x0 is set, copy it to input location x. Sets a Python
 * exception and returns -1 on failure. */
static scs_int get_warm_start(scs_float *x, scs_int l, PyArrayObject *x0) {
  /* PyArray_Check first: PyArray_ISFLOAT/NDIM/DIM read numpy-specific
   * struct fields, so on a non-array object they'd dereference garbage. */
  if (!PyArray_Check((PyObject *)x0) || !PyArray_ISFLOAT(x0) ||
      PyArray_NDIM(x0) != 1 || PyArray_DIM(x0, 0) != (npy_intp)l) {
    PyErr_Format(PyExc_ValueError,
                 "Warm-start must be a 1-D float array of length %ld",
                 (long)l);
    return -1;
  }
  PyArrayObject *px0 = scs_get_contiguous(x0, scs_get_float_type());
  if (!px0) {
    return -1;
  }
  memcpy(x, (scs_float *)PyArray_DATA(px0), l * sizeof(scs_float));
  Py_DECREF(px0);
  return 0;
}

static int get_cone_arr_dim(char *key, scs_int **varr, scs_int *vsize,
                            PyObject *cone) {
  /* get cone['key'] */
  scs_int i, n = 0;
  scs_int *q = NULL;
  PyObject *obj = NULL;
  int rc = PyDict_GetItemStringRef(cone, key, &obj);
  if (rc < 0) {
    return printErr(key);
  }
  if (obj) {
    if (PyList_Check(obj)) {
      n = (scs_int)PyList_Size(obj);
      q = (scs_int *)scs_calloc(n, sizeof(scs_int));
      if (n > 0 && !q) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return -1;
      }
      for (i = 0; i < n; ++i) {
        PyObject *qi = PyList_GetItemRef(obj, i);
        if (!qi || parse_pos_scs_int(qi, &(q[i])) < 0) {
          Py_XDECREF(qi);
          scs_free(q);
          Py_DECREF(obj);
          return printErr(key);
        }
        Py_DECREF(qi);
      }
    } else if (PyLong_Check(obj)) {
      n = 1;
      q = (scs_int *)scs_malloc(sizeof(scs_int));
      if (!q) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return -1;
      }
      if (parse_pos_scs_int(obj, q) < 0) {
        scs_free(q);
        Py_DECREF(obj);
        return printErr(key);
      }
    } else if (PyArray_Check(obj)) {
      PyArrayObject *pobj = (PyArrayObject *)obj;
      if (!PyArray_ISINTEGER(pobj) || PyArray_NDIM(pobj) != 1) {
        Py_DECREF(obj);
        return printErr(key);
      }
      n = (scs_int)PyArray_Size((PyObject *)obj);
      q = (scs_int *)scs_calloc(n, sizeof(scs_int));
      if (n > 0 && !q) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return -1;
      }
      PyArrayObject *px0 = scs_get_contiguous(pobj, scs_get_int_type());
      if (!px0) {
        scs_free(q);
        Py_DECREF(obj);
        return -1;
      }
      memcpy(q, (scs_int *)PyArray_DATA(px0), n * sizeof(scs_int));
      Py_DECREF(px0);
      /* Match the list/scalar branches (which use parse_pos_scs_int):
       * cone dimensions must be non-negative. Without this check a
       * user-supplied numpy array with negative entries would slip
       * through Python-side validation and surface later as a generic
       * "ScsWork allocation error!" from SCS core. */
      for (i = 0; i < n; ++i) {
        if (q[i] < 0) {
          scs_free(q);
          Py_DECREF(obj);
          return printErr(key);
        }
      }
    } else {
      Py_DECREF(obj);
      return printErr(key);
    }
    if (PyErr_Occurred()) {
      /* potentially could have been triggered before */
      scs_free(q);
      Py_DECREF(obj);
      return printErr(key);
    }
    Py_DECREF(obj);
  }
  *vsize = n;
  *varr = q;
  return 0;
}

static int get_cone_float_arr(char *key, scs_float **varr, scs_int *vsize,
                              PyObject *cone) {
  /* get cone['key'] */
  scs_int i, n = 0;
  scs_float *q = NULL;
  PyObject *obj = NULL;
  int rc = PyDict_GetItemStringRef(cone, key, &obj);
  if (rc < 0) {
    return printErr(key);
  }
  if (obj) {
    if (PyList_Check(obj)) {
      n = (scs_int)PyList_Size(obj);
      q = (scs_float *)scs_calloc(n, sizeof(scs_float));
      if (n > 0 && !q) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return -1;
      }
      for (i = 0; i < n; ++i) {
        PyObject *qi = PyList_GetItemRef(obj, i);
        if (!qi) {
          scs_free(q);
          Py_DECREF(obj);
          return printErr(key);
        }
        double v = PyFloat_AsDouble(qi);
        Py_DECREF(qi);
        if (v == -1.0 && PyErr_Occurred()) {
          scs_free(q);
          Py_DECREF(obj);
          return printErr(key);
        }
        q[i] = (scs_float)v;
      }
    } else if (PyLong_Check(obj) || PyFloat_Check(obj)) {
      n = 1;
      q = (scs_float *)scs_malloc(sizeof(scs_float));
      if (!q) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return -1;
      }
      double v = PyFloat_AsDouble(obj);
      if (v == -1.0 && PyErr_Occurred()) {
        scs_free(q);
        Py_DECREF(obj);
        return printErr(key);
      }
      q[0] = (scs_float)v;
    } else if (PyArray_Check(obj)) {
      PyArrayObject *pobj = (PyArrayObject *)obj;
      if (!PyArray_ISFLOAT(pobj) || PyArray_NDIM(pobj) != 1) {
        Py_DECREF(obj);
        return printErr(key);
      }
      n = (scs_int)PyArray_Size((PyObject *)obj);
      q = (scs_float *)scs_calloc(n, sizeof(scs_float));
      if (n > 0 && !q) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return -1;
      }
      PyArrayObject *px0 = scs_get_contiguous(pobj, scs_get_float_type());
      if (!px0) {
        scs_free(q);
        Py_DECREF(obj);
        return -1;
      }
      memcpy(q, (scs_float *)PyArray_DATA(px0), n * sizeof(scs_float));
      Py_DECREF(px0);
    } else {
      Py_DECREF(obj);
      return printErr(key);
    }
    if (PyErr_Occurred()) {
      /* potentially could have been triggered before */
      scs_free(q);
      Py_DECREF(obj);
      return printErr(key);
    }
    Py_DECREF(obj);
  }
  *vsize = n;
  *varr = q;
  return 0;
}

static void free_py_scs_data(ScsData *d, ScsCone *k, ScsSettings *stgs,
                             struct ScsPyData *ps) {
  if (ps->Ax) {
    Py_DECREF(ps->Ax);
  }
  if (ps->Ai) {
    Py_DECREF(ps->Ai);
  }
  if (ps->Ap) {
    Py_DECREF(ps->Ap);
  }
  if (ps->Px) {
    Py_DECREF(ps->Px);
  }
  if (ps->Pi) {
    Py_DECREF(ps->Pi);
  }
  if (ps->Pp) {
    Py_DECREF(ps->Pp);
  }
  if (ps->b) {
    Py_DECREF(ps->b);
  }
  if (ps->c) {
    Py_DECREF(ps->c);
  }
  if (k) {
    if (k->bu) {
      scs_free(k->bu);
    }
    if (k->bl) {
      scs_free(k->bl);
    }
    if (k->q) {
      scs_free(k->q);
    }
    if (k->s) {
      scs_free(k->s);
    }
    if (k->cs) {
      scs_free(k->cs);
    }
    if (k->p) {
      scs_free(k->p);
    }
#ifdef USE_SPECTRAL_CONES
    if (k->d) {
      scs_free(k->d);
    }
    if (k->nuc_m) {
      scs_free(k->nuc_m);
    }
    if (k->nuc_n) {
      scs_free(k->nuc_n);
    }
    if (k->ell1) {
      scs_free(k->ell1);
    }
    if (k->sl_n) {
      scs_free(k->sl_n);
    }
    if (k->sl_k) {
      scs_free(k->sl_k);
    }
#endif
    scs_free(k);
  }
  if (d) {
    if (d->A) {
      scs_free(d->A);
    }
    if (d->P) {
      scs_free(d->P);
    }
    scs_free(d);
  }
  if (stgs) {
    scs_free(stgs);
  }
}

/* The finish_with_* / none_with_* helpers do NOT clobber a pending
 * exception. This matters when a lower-level helper (e.g. a cone parser
 * or scs_get_contiguous) has already set a specific TypeError or
 * MemoryError; we want that specific error to reach the user, not a
 * generic caller-level message. */
static int finish_with_error(char *str) {
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, str);
  }
  return -1;
}

static int finish_with_type_error(char *str) {
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, str);
  }
  return -1;
}

static PyObject *none_with_error(char *str) {
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, str);
  }
  return (PyObject *)NULL;
}

static PyObject *none_with_type_error(char *str) {
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, str);
  }
  return (PyObject *)NULL;
}

static int SCS_init(SCS *self, PyObject *args, PyObject *kwargs) {
  /* data structures for arguments */
  PyArrayObject *Ax, *Ai, *Ap, *Px, *Pi, *Pp, *c, *b;
  PyObject *cone;
  PyObject *verbose = NULL;
  PyObject *normalize = NULL;
  PyObject *adaptive_scale = NULL;
  /* get the typenum for the primitive scs_int and scs_float types */
  int scs_int_type = scs_get_int_type();
  int scs_float_type = scs_get_float_type();
  scs_int bsizeu, bsizel, f_tmp;
  struct ScsPyData ps = {0};
  /* scs data structures */
  ScsData *d = (ScsData *)scs_calloc(1, sizeof(ScsData));
  ScsCone *k = (ScsCone *)scs_calloc(1, sizeof(ScsCone));
  ScsSettings *stgs = (ScsSettings *)scs_calloc(1, sizeof(ScsSettings));
  if (!d || !k || !stgs) {
    scs_free(d);
    scs_free(k);
    scs_free(stgs);
    PyErr_NoMemory();
    return -1;
  }

  ScsMatrix *A, *P;
  char *kwlist[] = {"shape",
                    "Ax",
                    "Ai",
                    "Ap",
                    "Px",
                    "Pi",
                    "Pp",
                    "b",
                    "c",
                    "cone",
                    "verbose",
                    "normalize",
                    "adaptive_scale",
                    "max_iters",
                    "scale",
                    "eps_abs",
                    "eps_rel",
                    "eps_infeas",
                    "alpha",
                    "rho_x",
                    "time_limit_secs",
                    "acceleration_lookback",
                    "acceleration_interval",
                    "write_data_filename",
                    "log_csv_filename",
                    NULL};

/* parse the arguments and ensure they are the correct type */
/* Use 'L' (long long) for DLONG so that scs_int fields are parsed correctly
   on Windows where sizeof(long) < sizeof(long long) (LLP64 model). */
#ifdef DLONG
#ifdef SFLOAT
  char *argparse_string = "(LL)O!O!O!OOOO!O!O!|O!O!O!LfffffffLLzz";
#else
  char *argparse_string = "(LL)O!O!O!OOOO!O!O!|O!O!O!LdddddddLLzz";
#endif
#else
#ifdef SFLOAT
  char *argparse_string = "(ii)O!O!O!OOOO!O!O!|O!O!O!ifffffffiizz";
#else
  char *argparse_string = "(ii)O!O!O!OOOO!O!O!|O!O!O!idddddddiizz";
#endif
#endif

  /* Check that the workspace is not already initialized */
  if (self->work) {
    return finish_with_error("Workspace already setup!");
  }

  /* set defaults */
  scs_set_default_settings(stgs);

  /* clang-format off */
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, argparse_string, kwlist, &(d->m), &(d->n),
          &PyArray_Type, &Ax,
          &PyArray_Type, &Ai,
          &PyArray_Type, &Ap,
          /* P can be None, so don't check is PyArray_Type */
          /* TODO: Is there some other type that can handle None? */
          &Px, &Pi, &Pp,
          &PyArray_Type, &b,
          &PyArray_Type, &c,
          &PyDict_Type, &cone,
          &PyBool_Type, &verbose,
          &PyBool_Type, &normalize,
          &PyBool_Type, &adaptive_scale,
          &(stgs->max_iters),
          &(stgs->scale),
          &(stgs->eps_abs),
          &(stgs->eps_rel),
          &(stgs->eps_infeas),
          &(stgs->alpha),
          &(stgs->rho_x),
          &(stgs->time_limit_secs),
          &(stgs->acceleration_lookback),
          &(stgs->acceleration_interval),
          &(stgs->write_data_filename),
          &(stgs->log_csv_filename))) {
    /* PyArg_ParseTupleAndKeywords already set an informative TypeError
     * (e.g. "argument 14 must be int, not str"). Overwriting it with a
     * generic ValueError would hide which input was rejected. */
    free_py_scs_data(d, k, stgs, &ps);
    return -1;
  }
  /* clang-format on */

  if (d->m <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("m must be a positive integer");
  }

  if (d->n <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("n must be a positive integer");
  }

  self->n = d->n;
  self->m = d->m;

  /* set A */
  if (!PyArray_ISFLOAT(Ax) || PyArray_NDIM(Ax) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_type_error("Ax must be a 1-D numpy array of floats");
  }
  if (!PyArray_ISINTEGER(Ai) || PyArray_NDIM(Ai) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_type_error("Ai must be a 1-D numpy array of ints");
  }
  if (!PyArray_ISINTEGER(Ap) || PyArray_NDIM(Ap) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_type_error("Ap must be a 1-D numpy array of ints");
  }
  ps.Ax = scs_get_contiguous(Ax, scs_float_type);
  ps.Ai = scs_get_contiguous(Ai, scs_int_type);
  ps.Ap = scs_get_contiguous(Ap, scs_int_type);
  if (!ps.Ax || !ps.Ai || !ps.Ap) {
    free_py_scs_data(d, k, stgs, &ps);
    return -1;  /* numpy set the exception */
  }

  A = (ScsMatrix *)scs_malloc(sizeof(ScsMatrix));
  if (!A) {
    free_py_scs_data(d, k, stgs, &ps);
    PyErr_NoMemory();
    return -1;
  }
  A->n = d->n;
  A->m = d->m;
  A->x = (scs_float *)PyArray_DATA(ps.Ax);
  A->i = (scs_int *)PyArray_DATA(ps.Ai);
  A->p = (scs_int *)PyArray_DATA(ps.Ap);
  d->A = A;

  /* set P if passed in */
  if (!Py_IsNone((PyObject *)Px) && !Py_IsNone((PyObject *)Pi) &&
      !Py_IsNone((PyObject *)Pp)) {
    /* Px/Pi/Pp are parsed with 'O' (to allow None), so we must guard
     * PyArray_ISFLOAT/ISINTEGER with PyArray_Check — those macros read
     * PyArrayObject-specific fields and are UB on non-array objects. */
    if (!PyArray_Check((PyObject *)Px) || !PyArray_ISFLOAT(Px) ||
        PyArray_NDIM(Px) != 1) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_type_error("Px must be a 1-D numpy array of floats");
    }
    if (!PyArray_Check((PyObject *)Pi) || !PyArray_ISINTEGER(Pi) ||
        PyArray_NDIM(Pi) != 1) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_type_error("Pi must be a 1-D numpy array of ints");
    }
    if (!PyArray_Check((PyObject *)Pp) || !PyArray_ISINTEGER(Pp) ||
        PyArray_NDIM(Pp) != 1) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_type_error("Pp must be a 1-D numpy array of ints");
    }
    ps.Px = scs_get_contiguous(Px, scs_float_type);
    ps.Pi = scs_get_contiguous(Pi, scs_int_type);
    ps.Pp = scs_get_contiguous(Pp, scs_int_type);
    if (!ps.Px || !ps.Pi || !ps.Pp) {
      free_py_scs_data(d, k, stgs, &ps);
      return -1;  /* numpy set the exception */
    }

    P = (ScsMatrix *)scs_malloc(sizeof(ScsMatrix));
    if (!P) {
      free_py_scs_data(d, k, stgs, &ps);
      PyErr_NoMemory();
      return -1;
    }
    P->n = d->n;
    P->m = d->n;
    P->x = (scs_float *)PyArray_DATA(ps.Px);
    P->i = (scs_int *)PyArray_DATA(ps.Pi);
    P->p = (scs_int *)PyArray_DATA(ps.Pp);
    d->P = P;
  } else {
    d->P = NULL;
  }
  /* set c */
  if (!PyArray_ISFLOAT(c) || PyArray_NDIM(c) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_type_error(
        "c must be a 1-D numpy array of floats");
  }
  if (PyArray_DIM(c, 0) != (npy_intp)d->n) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("c has incompatible dimension with A");
  }
  ps.c = scs_get_contiguous(c, scs_float_type);
  if (!ps.c) {
    free_py_scs_data(d, k, stgs, &ps);
    return -1;
  }
  d->c = (scs_float *)PyArray_DATA(ps.c);
  /* set b */
  if (!PyArray_ISFLOAT(b) || PyArray_NDIM(b) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_type_error(
        "b must be a 1-D numpy array of floats");
  }
  if (PyArray_DIM(b, 0) != (npy_intp)d->m) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("b has incompatible dimension with A");
  }
  ps.b = scs_get_contiguous(b, scs_float_type);
  if (!ps.b) {
    free_py_scs_data(d, k, stgs, &ps);
    return -1;
  }
  d->b = (scs_float *)PyArray_DATA(ps.b);

  if (get_pos_int_param("f", &(f_tmp), 0, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field f");
  }
  if (get_pos_int_param("z", &(k->z), 0, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field z");
  }
  if (f_tmp > 0) {
    /* PyErr_WarnEx returns -1 if the warning was promoted to an exception
     * (e.g. warnings.filterwarnings("error")); in that case the exception
     * is already set, so we just need to clean up and return -1. */
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "The 'f' cone field is deprecated; use 'z' (Zero cone) "
                     "instead. If both 'f' and 'z' are set they are summed.",
                     1) < 0) {
      free_py_scs_data(d, k, stgs, &ps);
      return -1;
    }
    k->z += f_tmp;
  }
  if (get_pos_int_param("l", &(k->l), 0, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field l");
  }
  /* box cone */
  if (get_cone_float_arr("bu", &(k->bu), &bsizeu, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field bu");
  }
  if (get_cone_float_arr("bl", &(k->bl), &bsizel, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field bl");
  }
  if (bsizeu != bsizel) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("bu different dimension to bl");
  }
  if (bsizeu > 0) {
    k->bsize = bsizeu + 1; /* cone = (t,s), bsize = total length */
  }
  /* end box cone */
  if (get_cone_arr_dim("q", &(k->q), &(k->qsize), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field q");
  }
  if (get_cone_arr_dim("s", &(k->s), &(k->ssize), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field s");
  }
  if (get_cone_arr_dim("cs", &(k->cs), &(k->cssize), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field cs");
  }
  if (get_cone_float_arr("p", &(k->p), &(k->psize), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("failed to parse cone field p");
  }
  if (get_pos_int_param("ep", &(k->ep), 0, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field ep");
  }
  if (get_pos_int_param("ed", &(k->ed), 0, cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field ed");
  }

#ifdef USE_SPECTRAL_CONES
  /* logdet cone */
  if (get_cone_arr_dim("d", &(k->d), &(k->dsize), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field d");
  }
  /* nuclear norm cone */
  if (get_cone_arr_dim("nuc_m", &(k->nuc_m), &(k->nucsize), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field nuc_m");
  }
  {
    scs_int nuc_n_size = 0;
    if (get_cone_arr_dim("nuc_n", &(k->nuc_n), &nuc_n_size, cone) < 0) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("Failed to parse cone field nuc_n");
    }
    if (nuc_n_size != k->nucsize) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("nuc_m and nuc_n must have the same length");
    }
  }
  /* ell1 cone */
  if (get_cone_arr_dim("ell1", &(k->ell1), &(k->ell1_size), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field ell1");
  }
  /* sum of largest eigenvalues cone */
  if (get_cone_arr_dim("sl_n", &(k->sl_n), &(k->sl_size), cone) < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Failed to parse cone field sl_n");
  }
  {
    scs_int sl_k_size = 0;
    if (get_cone_arr_dim("sl_k", &(k->sl_k), &sl_k_size, cone) < 0) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("Failed to parse cone field sl_k");
    }
    if (sl_k_size != k->sl_size) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("sl_n and sl_k must have the same length");
    }
  }
#endif

  stgs->verbose = verbose ? (scs_int)PyObject_IsTrue(verbose) : VERBOSE;
  stgs->normalize = normalize ? (scs_int)PyObject_IsTrue(normalize) : NORMALIZE;
  stgs->adaptive_scale = adaptive_scale
                             ? (scs_int)PyObject_IsTrue(adaptive_scale)
                             : ADAPTIVE_SCALE;

  /* Ranges below match SCS's own validate() in scs_source/src/scs.c, plus
   * explicit isnan/isfinite guards that SCS's validate() lacks. Without
   * those, NaN values slip through every `x <= 0` / `x < 0` / `x >= 2`
   * check (IEEE NaN comparisons always return false) and the solver runs
   * to completion producing NaN iterates; +inf on scale or rho_x either
   * crashes the linear-system factorization with a misleading "ScsWork
   * allocation error!" or silently produces NaN. Better to raise a
   * Python exception naming the offending setting up front. */
  if (stgs->max_iters <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("max_iters must be positive");
  }
  /* acceleration_lookback: positive selects type-I AA, negative selects
   * type-II; the magnitude is used as the memory size. Zero disables
   * acceleration. Passed through unchanged — see scs.c's aa_init call. */
  if (stgs->acceleration_interval <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("acceleration_interval must be positive");
  }
  if (!isfinite((double)stgs->scale) || stgs->scale <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("scale must be a positive finite number");
  }
  /* time_limit_secs: 0 disables the limit, +inf is equivalent, both allowed. */
  if (isnan((double)stgs->time_limit_secs) || stgs->time_limit_secs < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("time_limit_secs must be nonnegative");
  }
  /* eps_*: +inf is allowed (effectively disables that stopping criterion);
   * NaN is not — it would make every tolerance comparison false. */
  if (isnan((double)stgs->eps_abs) || stgs->eps_abs < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("eps_abs must be nonnegative");
  }
  if (isnan((double)stgs->eps_rel) || stgs->eps_rel < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("eps_rel must be nonnegative");
  }
  if (isnan((double)stgs->eps_infeas) || stgs->eps_infeas < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("eps_infeas must be nonnegative");
  }
  if (!isfinite((double)stgs->alpha) || stgs->alpha <= 0 || stgs->alpha >= 2) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("alpha must be in (0, 2)");
  }
  if (!isfinite((double)stgs->rho_x) || stgs->rho_x <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("rho_x must be a positive finite number");
  }
  stgs->warm_start = WARM_START; /* False by default */

  /* Initialize solution struct. These allocations feed into the lifetime
   * of self — SCS_finish unconditionally scs_free's them. Accept a
   * zero-length calloc returning NULL only when the corresponding
   * dimension is zero. */
  self->sol = (ScsSolution *)scs_calloc(1, sizeof(ScsSolution));
  if (!self->sol) {
    free_py_scs_data(d, k, stgs, &ps);
    PyErr_NoMemory();
    return -1;
  }
  self->sol->x = (scs_float *)scs_calloc(self->n, sizeof(scs_float));
  self->sol->y = (scs_float *)scs_calloc(self->m, sizeof(scs_float));
  self->sol->s = (scs_float *)scs_calloc(self->m, sizeof(scs_float));
  if ((self->n > 0 && !self->sol->x) ||
      (self->m > 0 && (!self->sol->y || !self->sol->s))) {
    free_py_scs_data(d, k, stgs, &ps);
    /* SCS_finish (via tp_dealloc) will free whichever of x/y/s succeeded. */
    PyErr_NoMemory();
    return -1;
  }

  /* Allocate per-instance lock to protect work/sol from concurrent access.
   * Needed because Py_BEGIN_ALLOW_THREADS releases the GIL during scs_solve,
   * allowing concurrent calls on the same instance to race. */
  self->lock = PyThread_allocate_lock();
  if (!self->lock) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Unable to allocate instance lock");
  }

  /* release the GIL */
  Py_BEGIN_ALLOW_THREADS;
  self->work = scs_init(d, k, stgs);
  /* reacquire the GIL */
  Py_END_ALLOW_THREADS;

  /* no longer need pointers to arrays that held primitives */
  free_py_scs_data(d, k, stgs, &ps);

  if (self->work) { /* Workspace allocation correct */
    return 0;
  }
  return finish_with_error("ScsWork allocation error!");
}

static PyObject *SCS_solve(SCS *self, PyObject *args) {
  ScsInfo info = {0};
  ScsSolution *sol = self->sol;
  npy_intp veclen[1];
  int scs_float_type = scs_get_float_type();

  PyArrayObject *warm_x, *warm_y, *warm_s;
  PyObject *warm_start;

  /* clang-format off */
  /* warm_* can be None, so don't check is PyArray_Type */
  if (!PyArg_ParseTuple(args, "O!OOO",
                        &PyBool_Type, &warm_start,
                        &warm_x,
                        &warm_y,
                        &warm_s)) {
    /* PyArg_ParseTuple already set an informative TypeError; propagate it. */
    return (PyObject *)NULL;
  }
  /* clang-format on */

  scs_int _warm_start = (scs_int)PyObject_IsTrue(warm_start);

  /* Acquire per-instance lock. Release the GIL first to avoid deadlock:
   * another thread may hold this lock inside scs_solve (with GIL released),
   * so we must not hold the GIL while waiting for the lock. */
  int lock_ok;
  Py_BEGIN_ALLOW_THREADS;
  lock_ok = (PyThread_acquire_lock(self->lock, WAIT_LOCK) == PY_LOCK_ACQUIRED);
  Py_END_ALLOW_THREADS;

  if (!lock_ok) {
    return none_with_error("Failed to acquire instance lock");
  }

  /* Check workspace under lock to avoid TOCTOU race with SCS_finish */
  if (!self->work) {
    PyThread_release_lock(self->lock);
    return none_with_error("Workspace not initialized!");
  }

  if (_warm_start) {
    /* If any of these of missing, we use the values in sol */
    if (!Py_IsNone((PyObject *)warm_x)) {
      if (get_warm_start(self->sol->x, self->n, warm_x) < 0) {
        PyThread_release_lock(self->lock);
        return none_with_error("Unable to parse x warm-start");
      }
    }
    if (!Py_IsNone((PyObject *)warm_y)) {
      if (get_warm_start(self->sol->y, self->m, warm_y) < 0) {
        PyThread_release_lock(self->lock);
        return none_with_error("Unable to parse y warm-start");
      }
    }
    if (!Py_IsNone((PyObject *)warm_s)) {
      if (get_warm_start(self->sol->s, self->m, warm_s) < 0) {
        PyThread_release_lock(self->lock);
        return none_with_error("Unable to parse s warm-start");
      }
    }
  }
  /* else: SCS will overwite sol if _warm_start is false */
  /* so we don't need to set to zeros here */

  PyObject *x, *y, *s, *return_dict, *info_dict, *aa_stats_dict;
  scs_float *_x, *_y, *_s;
  /* release the GIL */
  Py_BEGIN_ALLOW_THREADS;
  /* Solve! */
  scs_solve(self->work, sol, &info, _warm_start);
  Py_END_ALLOW_THREADS;

  /* Copy results out of sol while still holding the lock, because another
   * thread's solve could overwrite sol as soon as we release.
   * Note: unlike SCS_update, we release the lock after Py_END_ALLOW_THREADS
   * because we need to read from sol (shared state) under lock protection. */
  _x = scs_malloc(self->n * sizeof(scs_float));
  _y = scs_malloc(self->m * sizeof(scs_float));
  _s = scs_malloc(self->m * sizeof(scs_float));
  if ((self->n > 0 && !_x) || (self->m > 0 && (!_y || !_s))) {
    scs_free(_x);
    scs_free(_y);
    scs_free(_s);
    PyThread_release_lock(self->lock);
    PyErr_NoMemory();
    return NULL;
  }
  memcpy(_x, sol->x, self->n * sizeof(scs_float));
  memcpy(_y, sol->y, self->m * sizeof(scs_float));
  memcpy(_s, sol->s, self->m * sizeof(scs_float));

  PyThread_release_lock(self->lock);

  /* Build numpy arrays from the copied data (no longer under lock since
   * these are thread-local copies). If PyArray_SimpleNewFromData fails
   * (OOM), it sets a Python exception but does NOT take ownership of the
   * buffer — so we must free the raw buffer ourselves and Py_DECREF any
   * arrays already built (which own their buffers via NPY_ARRAY_OWNDATA). */
  veclen[0] = self->n;
  x = PyArray_SimpleNewFromData(1, veclen, scs_float_type, _x);
  if (!x) {
    scs_free(_x);
    scs_free(_y);
    scs_free(_s);
    return NULL;
  }
  PyArray_ENABLEFLAGS((PyArrayObject *)x, NPY_ARRAY_OWNDATA);

  veclen[0] = self->m;
  y = PyArray_SimpleNewFromData(1, veclen, scs_float_type, _y);
  if (!y) {
    scs_free(_y);
    scs_free(_s);
    Py_DECREF(x);
    return NULL;
  }
  PyArray_ENABLEFLAGS((PyArrayObject *)y, NPY_ARRAY_OWNDATA);

  veclen[0] = self->m;
  s = PyArray_SimpleNewFromData(1, veclen, scs_float_type, _s);
  if (!s) {
    scs_free(_s);
    Py_DECREF(x);
    Py_DECREF(y);
    return NULL;
  }
  PyArray_ENABLEFLAGS((PyArrayObject *)s, NPY_ARRAY_OWNDATA);

/* output arguments */
/* Use 'L' (long long) for scs_int under DLONG to match scs_int's typedef
 * (long long); 'l' (long) would truncate to 32 bits on LLP64 Windows.
 * Mirrors the argparse_string choice in SCS_init. */
#ifdef DLONG
#ifdef SFLOAT
  char *outarg_string = "{s:L,s:L,s:L,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,"
                        "s:f,s:f,s:f,s:f,s:f,s:L,s:L,s:s}";
  char *aa_stats_string = "{s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:f,s:f}";
#else
  char *outarg_string = "{s:L,s:L,s:L,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,"
                        "s:d,s:d,s:d,s:d,s:d,s:L,s:L,s:s}";
  char *aa_stats_string = "{s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:d,s:d}";
#endif
#else
#ifdef SFLOAT
  char *outarg_string = "{s:i,s:i,s:i,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,"
                        "s:f,s:f,s:f,s:f,s:f,s:i,s:i,s:s}";
  char *aa_stats_string = "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:f,s:f}";
#else
  char *outarg_string = "{s:i,s:i,s:i,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,"
                        "s:d,s:d,s:d,s:d,s:d,s:i,s:i,s:s}";
  char *aa_stats_string = "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:d,s:d}";
#endif
#endif

  /* clang-format off */
  /* if you add fields to this remember to update outarg_string */
  info_dict = Py_BuildValue(
      outarg_string,
      "status_val", (scs_int)info.status_val,
      "iter", (scs_int)info.iter,
      "scale_updates", (scs_int)info.scale_updates,
      "scale", (scs_float)info.scale,
      "pobj", (scs_float)info.pobj,
      "dobj", (scs_float)info.dobj,
      "res_pri", (scs_float)info.res_pri,
      "res_dual", (scs_float)info.res_dual,
      "gap", (scs_float)info.gap,
      "res_infeas", (scs_float)info.res_infeas,
      "res_unbdd_a", (scs_float)info.res_unbdd_a,
      "res_unbdd_p", (scs_float)info.res_unbdd_p,
      "comp_slack", (scs_float)info.comp_slack,
      "solve_time", (scs_float)(info.solve_time),
      "setup_time", (scs_float)(info.setup_time),
      "lin_sys_time", (scs_float)(info.lin_sys_time),
      "cone_time", (scs_float)(info.cone_time),
      "accel_time", (scs_float)(info.accel_time),
      "rejected_accel_steps", (scs_int)info.rejected_accel_steps,
      "accepted_accel_steps", (scs_int)info.accepted_accel_steps,
      "status", info.status);
  aa_stats_dict = Py_BuildValue(
      aa_stats_string,
      "iter", (scs_int)info.aa_stats.iter,
      "n_accept", (scs_int)info.aa_stats.n_accept,
      "n_reject_lapack", (scs_int)info.aa_stats.n_reject_lapack,
      "n_reject_rank0", (scs_int)info.aa_stats.n_reject_rank0,
      "n_reject_nonfinite", (scs_int)info.aa_stats.n_reject_nonfinite,
      "n_reject_weight_cap", (scs_int)info.aa_stats.n_reject_weight_cap,
      "n_safeguard_reject", (scs_int)info.aa_stats.n_safeguard_reject,
      "last_rank", (scs_int)info.aa_stats.last_rank,
      "last_aa_norm", (scs_float)info.aa_stats.last_aa_norm,
      "last_regularization", (scs_float)info.aa_stats.last_regularization);
  /* clang-format on */

  if (!info_dict || !aa_stats_dict ||
      PyDict_SetItemString(info_dict, "aa_stats", aa_stats_dict) < 0) {
    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(s);
    Py_XDECREF(info_dict);
    Py_XDECREF(aa_stats_dict);
    return NULL;
  }

  return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}", "x", x, "y", y, "s", s,
                              "info", info_dict);
  /* Give up ownership to the return dictionary. x/y/s/info_dict are non-NULL
   * here, and Py_BuildValue borrowed each with "O". */
  Py_DECREF(x);
  Py_DECREF(y);
  Py_DECREF(s);
  Py_DECREF(info_dict);
  Py_DECREF(aa_stats_dict);

  return return_dict;
}

static PyObject *SCS_update(SCS *self, PyObject *args) {
  /* data structures for arguments */

  /* get the typenum for the primitive scs_float type */
  int scs_float_type = scs_get_float_type();

  PyArrayObject *b_in, *c_in;
  /* Contiguous copies we own (strong refs) or NULL when the corresponding
   * input was None or we haven't made a copy yet. */
  PyArrayObject *b_contig = NULL, *c_contig = NULL;
  scs_float *b = NULL, *c = NULL;

  /* b, c can be None, so don't check is PyArray_Type */
  if (!PyArg_ParseTuple(args, "OO", &b_in, &c_in)) {
    /* PyArg_ParseTuple already set an informative TypeError; propagate it. */
    return (PyObject *)NULL;
  }
  /* set c */
  if (!Py_IsNone((PyObject *)c_in)) {
    /* Parsed with 'O' (to allow None): must PyArray_Check before using
     * PyArray_ISFLOAT/NDIM/DIM, which are UB on non-array objects. */
    if (!PyArray_Check((PyObject *)c_in) || !PyArray_ISFLOAT(c_in) ||
        PyArray_NDIM(c_in) != 1) {
      return none_with_type_error(
          "c_new must be a 1-D numpy array of floats");
    }
    if (PyArray_DIM(c_in, 0) != (npy_intp)self->n) {
      return none_with_error("c_new has incompatible dimension with A");
    }
    c_contig = scs_get_contiguous(c_in, scs_float_type);
    if (!c_contig) {
      return NULL;  /* numpy set the exception */
    }
    c = (scs_float *)PyArray_DATA(c_contig);
  }
  /* set b */
  if (!Py_IsNone((PyObject *)b_in)) {
    /* Parsed with 'O' (to allow None): must PyArray_Check before using
     * PyArray_ISFLOAT/NDIM/DIM, which are UB on non-array objects. */
    if (!PyArray_Check((PyObject *)b_in) || !PyArray_ISFLOAT(b_in) ||
        PyArray_NDIM(b_in) != 1) {
      Py_XDECREF(c_contig);
      return none_with_type_error(
          "b_new must be a 1-D numpy array of floats");
    }
    if (PyArray_DIM(b_in, 0) != (npy_intp)self->m) {
      Py_XDECREF(c_contig);
      return none_with_error("b_new has incompatible dimension with A");
    }
    b_contig = scs_get_contiguous(b_in, scs_float_type);
    if (!b_contig) {
      Py_XDECREF(c_contig);
      return NULL;
    }
    b = (scs_float *)PyArray_DATA(b_contig);
  }

  /* Acquire per-instance lock (release GIL first to avoid deadlock) */
  int lock_ok;
  Py_BEGIN_ALLOW_THREADS;
  lock_ok = (PyThread_acquire_lock(self->lock, WAIT_LOCK) == PY_LOCK_ACQUIRED);
  Py_END_ALLOW_THREADS;

  if (!lock_ok) {
    Py_XDECREF(b_contig);
    Py_XDECREF(c_contig);
    return none_with_error("Failed to acquire instance lock");
  }

  /* Check workspace under lock to avoid TOCTOU race with SCS_finish */
  if (!self->work) {
    PyThread_release_lock(self->lock);
    Py_XDECREF(b_contig);
    Py_XDECREF(c_contig);
    return none_with_error("Workspace not initialized!");
  }

  /* Release the GIL, run the update, then release the instance lock before
   * re-acquiring the GIL. This avoids holding the instance lock while
   * waiting for the GIL, which could deadlock if another GIL-holding thread
   * is waiting on this lock. SCS_solve uses a different order (release lock
   * after Py_END_ALLOW_THREADS) because it must copy results out of sol
   * while still holding the lock. */
  Py_BEGIN_ALLOW_THREADS;
  scs_update(self->work, b, c);
  PyThread_release_lock(self->lock);
  Py_END_ALLOW_THREADS;

  Py_XDECREF(b_contig);
  Py_XDECREF(c_contig);

  Py_RETURN_NONE;
}

/* Deallocate SCS object. Signature must match tp_dealloc
 * (void (*)(PyObject *)). Using the type's tp_free slot (rather than
 * PyObject_Free directly) is the standard C-API pattern and works
 * correctly for subclasses. */
static void SCS_finish(SCS *self) {
  if (self->work) {
    /* Acquire lock to ensure no concurrent solve/update is in progress.
     * We don't check the return value here because this is the dealloc
     * path — the object must be cleaned up regardless of lock status,
     * and there is no Python caller to return an error to. */
    if (self->lock) {
      PyThread_acquire_lock(self->lock, WAIT_LOCK);
    }
    scs_finish(self->work);
    self->work = NULL;
    if (self->lock) {
      PyThread_release_lock(self->lock);
    }
  }
  if (self->lock) {
    PyThread_free_lock(self->lock);
    self->lock = NULL;
  }
  if (self->sol) {
    scs_free(self->sol->x);
    scs_free(self->sol->y);
    scs_free(self->sol->s);
    scs_free(self->sol);
    self->sol = NULL;
  }

  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyMethodDef scs_obj_methods[] = {
    {"solve", (PyCFunction)SCS_solve, METH_VARARGS, PyDoc_STR("Solve problem")},
    {"update", (PyCFunction)SCS_update, METH_VARARGS,
     PyDoc_STR("Update b or c vectors")},
    {NULL, NULL} /* sentinel */
};

/* Define workspace type object */
static PyTypeObject SCS_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "scs.SCS", /* tp_name */
    sizeof(SCS),                              /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)SCS_finish,                   /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                       /* tp_flags */
    "SCS solver",                             /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    scs_obj_methods,                          /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)SCS_init,                       /* tp_init */
    0,                                        /* tp_alloc */
    PyType_GenericNew,                        /* tp_new */
};

#endif
