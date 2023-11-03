#ifndef PY_SCSOBJECT_H
#define PY_SCSOBJECT_H

/* SCS Object type */
typedef struct {
  PyObject_HEAD ScsWork *work; /* Workspace */
  ScsSolution *sol;            /* Solution, keep around for warm-starts */
  scs_int m, n;
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
  /* the "new_owner" pointer has to have Py_DECREF called on it; it owns */
  /* the "new" array object created by PyArray_Cast */
  PyArrayObject *tmp_arr;
  PyArrayObject *new_owner;
  tmp_arr = PyArray_GETCONTIGUOUS(array);
  new_owner = (PyArrayObject *)PyArray_Cast(tmp_arr, typenum);
  Py_DECREF(tmp_arr);
  return new_owner;
}

static int printErr(char *key) {
  PySys_WriteStderr("Error parsing '%s'\n", key);
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

/* If warm start x0 is set, copy it to input location x */
static scs_int get_warm_start(scs_float *x, scs_int l, PyArrayObject *x0) {
  if (!PyArray_ISFLOAT(x0) || PyArray_NDIM(x0) != 1 ||
      PyArray_DIM(x0, 0) != l) {
    PySys_WriteStderr("Error parsing warm-start input\n");
    return -1;
  }
  PyArrayObject *px0 = scs_get_contiguous(x0, scs_get_float_type());
  memcpy(x, (scs_float *)PyArray_DATA(px0), l * sizeof(scs_float));
  Py_DECREF(px0);
  return 0;
}

static int get_cone_arr_dim(char *key, scs_int **varr, scs_int *vsize,
                            PyObject *cone) {
  /* get cone['key'] */
  scs_int i, n = 0;
  scs_int *q = NULL;
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
    } else if (PyArray_Check(obj)) {
      PyArrayObject *pobj = (PyArrayObject *)obj;
      if (!PyArray_ISINTEGER(pobj) || PyArray_NDIM(pobj) != 1) {
        return printErr(key);
      }
      n = (scs_int)PyArray_Size((PyObject *)obj);
      q = (scs_int *)scs_calloc(n, sizeof(scs_int));
      PyArrayObject *px0 = scs_get_contiguous(pobj, scs_get_int_type());
      memcpy(q, (scs_int *)PyArray_DATA(px0), n * sizeof(scs_int));
      Py_DECREF(px0);
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
  scs_float *q = NULL;
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
    } else if (PyArray_Check(obj)) {
      PyArrayObject *pobj = (PyArrayObject *)obj;
      if (!PyArray_ISFLOAT(pobj) || PyArray_NDIM(pobj) != 1) {
        return printErr(key);
      }
      n = (scs_int)PyArray_Size((PyObject *)obj);
      q = (scs_float *)scs_calloc(n, sizeof(scs_float));
      PyArrayObject *px0 = scs_get_contiguous(pobj, scs_get_float_type());
      memcpy(q, (scs_float *)PyArray_DATA(px0), n * sizeof(scs_float));
      Py_DECREF(px0);
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
    if (k->p) {
      scs_free(k->p);
    }
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

static int finish_with_error(char *str) {
  PyErr_SetString(PyExc_ValueError, str);
  return -1;
}

PyObject *none_with_error(char *str) {
  PyErr_SetString(PyExc_ValueError, str);
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
#ifdef DLONG
#ifdef SFLOAT
  char *argparse_string = "(ll)O!O!O!OOOO!O!O!|O!O!O!lfffffffllzz";
#else
  char *argparse_string = "(ll)O!O!O!OOOO!O!O!|O!O!O!ldddddddllzz";
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
    finish_with_error("Workspace already setup!");
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
    return finish_with_error("Error parsing inputs\n");
  }
  /* clang-format on */

  if (d->m < 0) {
    return finish_with_error("m must be a positive integer");
  }

  if (d->n < 0) {
    return finish_with_error("n must be a positive integer");
  }

  self->n = d->n;
  self->m = d->m;

  /* set A */
  if (!PyArray_ISFLOAT(Ax) || PyArray_NDIM(Ax) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Ax must be a numpy array of floats");
  }
  if (!PyArray_ISINTEGER(Ai) || PyArray_NDIM(Ai) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Ai must be a numpy array of ints");
  }
  if (!PyArray_ISINTEGER(Ap) || PyArray_NDIM(Ap) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("Ap must be a numpy array of ints");
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

  /* set P if passed in */
  if ((void *)Px != Py_None && (void *)Pi != Py_None && (void *)Pp != Py_None) {
    if (!PyArray_ISFLOAT(Px) || PyArray_NDIM(Px) != 1) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("Px must be a numpy array of floats");
    }
    if (!PyArray_ISINTEGER(Pi) || PyArray_NDIM(Pi) != 1) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("Pi must be a numpy array of ints");
    }
    if (!PyArray_ISINTEGER(Pp) || PyArray_NDIM(Pp) != 1) {
      free_py_scs_data(d, k, stgs, &ps);
      return finish_with_error("Pp must be a numpy array of ints");
    }
    ps.Px = scs_get_contiguous(Px, scs_float_type);
    ps.Pi = scs_get_contiguous(Pi, scs_int_type);
    ps.Pp = scs_get_contiguous(Pp, scs_int_type);

    P = (ScsMatrix *)scs_malloc(sizeof(ScsMatrix));
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
    return finish_with_error(
        "c must be a dense numpy array with one dimension");
  }
  if (PyArray_DIM(c, 0) != d->n) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("c has incompatible dimension with A");
  }
  ps.c = scs_get_contiguous(c, scs_float_type);
  d->c = (scs_float *)PyArray_DATA(ps.c);
  /* set b */
  if (!PyArray_ISFLOAT(b) || PyArray_NDIM(b) != 1) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error(
        "b must be a dense numpy array with one dimension");
  }
  if (PyArray_DIM(b, 0) != d->m) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("b has incompatible dimension with A");
  }
  ps.b = scs_get_contiguous(b, scs_float_type);
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
    scs_printf("SCS deprecation warning: The 'f' field in the cone struct \n"
               "has been replaced by 'z' to better reflect the Zero cone. \n"
               "Please replace usage of 'f' with 'z'. If both 'f' and 'z' \n"
               "are set then we sum the two fields to get the final zero \n"
               "cone size.\n");
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

  stgs->verbose = verbose ? (scs_int)PyObject_IsTrue(verbose) : VERBOSE;
  stgs->normalize = normalize ? (scs_int)PyObject_IsTrue(normalize) : NORMALIZE;
  stgs->adaptive_scale = adaptive_scale
                             ? (scs_int)PyObject_IsTrue(adaptive_scale)
                             : ADAPTIVE_SCALE;

  if (stgs->max_iters < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("max_iters must be positive");
  }
  if (stgs->acceleration_lookback < 0) {
    /* hack - use type-I AA when lookback is < 0 */
    /* free_py_scs_data(d, k, stgs, &ps); */
    /* return finish_with_error("acceleration_lookback must be positive"); */
  }
  if (stgs->acceleration_interval < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("acceleration_interval must be positive");
  }
  if (stgs->scale <= 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("scale must be positive");
  }
  if (stgs->time_limit_secs < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("time_limit_secs must be nonnegative");
  }
  if (stgs->eps_abs < 0) {
    return finish_with_error("eps_abs must be positive");
  }
  if (stgs->eps_rel < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("eps_rel must be positive");
  }
  if (stgs->eps_infeas < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("eps_infeas must be positive");
  }
  if (stgs->alpha < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("alpha must be positive");
  }
  if (stgs->rho_x < 0) {
    free_py_scs_data(d, k, stgs, &ps);
    return finish_with_error("rho_x must be positive");
  }
  stgs->warm_start = WARM_START; /* False by default */

  /* Initialize solution struct */
  self->sol = (ScsSolution *)scs_calloc(1, sizeof(ScsSolution));
  self->sol->x = (scs_float *)scs_calloc(self->n, sizeof(scs_float));
  self->sol->y = (scs_float *)scs_calloc(self->m, sizeof(scs_float));
  self->sol->s = (scs_float *)scs_calloc(self->m, sizeof(scs_float));

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

  if (!self->work) {
    return none_with_error("Workspace not initialized!");
  }

  PyArrayObject *warm_x, *warm_y, *warm_s;
  PyObject *warm_start;

  /* clang-format off */
  /* warm_* can be None, so don't check is PyArray_Type */
  if (!PyArg_ParseTuple(args, "O!OOO",
                        &PyBool_Type, &warm_start,
                        &warm_x,
                        &warm_y,
                        &warm_s)) {
    return none_with_error("Error parsing inputs");
  }
  /* clang-format on */

  scs_int _warm_start = (scs_int)PyObject_IsTrue(warm_start);

  if (_warm_start) {
    /* If any of these of missing, we use the values in sol */
    if ((void *)warm_x != Py_None) {
      if (get_warm_start(self->sol->x, self->n, warm_x) < 0) {
        return none_with_error("Unable to parse x warm-start");
      }
    }
    if ((void *)warm_y != Py_None) {
      if (get_warm_start(self->sol->y, self->m, warm_y) < 0) {
        return none_with_error("Unable to parse y warm-start");
      }
    }
    if ((void *)warm_s != Py_None) {
      if (get_warm_start(self->sol->s, self->m, warm_s) < 0) {
        return none_with_error("Unable to parse s warm-start");
      }
    }
  }
  /* else: SCS will overwite sol if _warm_start is false */
  /* so we don't need to set to zeros here */

  PyObject *x, *y, *s, *return_dict, *info_dict;
  scs_float *_x, *_y, *_s;
  /* release the GIL */
  Py_BEGIN_ALLOW_THREADS;
  /* Solve! */
  scs_solve(self->work, sol, &info, _warm_start);
  /* reacquire the GIL */
  Py_END_ALLOW_THREADS;

  veclen[0] = self->n;
  _x = scs_malloc(self->n * sizeof(scs_float));
  memcpy(_x, sol->x, self->n * sizeof(scs_float));
  x = PyArray_SimpleNewFromData(1, veclen, scs_float_type, _x);
  PyArray_ENABLEFLAGS((PyArrayObject *)x, NPY_ARRAY_OWNDATA);

  veclen[0] = self->m;
  _y = scs_malloc(self->m * sizeof(scs_float));
  memcpy(_y, sol->y, self->m * sizeof(scs_float));
  y = PyArray_SimpleNewFromData(1, veclen, scs_float_type, _y);
  PyArray_ENABLEFLAGS((PyArrayObject *)y, NPY_ARRAY_OWNDATA);

  veclen[0] = self->m;
  _s = scs_malloc(self->m * sizeof(scs_float));
  memcpy(_s, sol->s, self->m * sizeof(scs_float));
  s = PyArray_SimpleNewFromData(1, veclen, scs_float_type, _s);
  PyArray_ENABLEFLAGS((PyArrayObject *)s, NPY_ARRAY_OWNDATA);

/* output arguments */
#ifdef DLONG
#ifdef SFLOAT
  char *outarg_string = "{s:l,s:l,s:l,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,"
                        "s:f,s:f,s:f,s:f,s:f,s:l,s:l,s:s}";
#else
  char *outarg_string = "{s:l,s:l,s:l,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,"
                        "s:d,s:d,s:d,s:d,s:d,s:l,s:l,s:s}";
#endif
#else
#ifdef SFLOAT
  char *outarg_string = "{s:i,s:i,s:i,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,s:f,"
                        "s:f,s:f,s:f,s:f,s:f,s:i,s:i,s:s}";
#else
  char *outarg_string = "{s:i,s:i,s:i,s:f,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,"
                        "s:d,s:d,s:d,s:d,s:d,s:i,s:i,s:s}";
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
  /* clang-format on */

  return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}", "x", x, "y", y, "s", s,
                              "info", info_dict);
  /* give up ownership to the return dictionary */
  Py_DECREF(x);
  Py_DECREF(y);
  Py_DECREF(s);
  Py_DECREF(info_dict);

  return return_dict;
}

PyObject *SCS_update(SCS *self, PyObject *args) {
  /* data structures for arguments */

  /* get the typenum for the primitive scs_float type */
  int scs_float_type = scs_get_float_type();

  PyArrayObject *b_new, *c_new;
  scs_float *b = NULL, *c = NULL;

  /* Check that the workspace is already initialized */
  if (!self->work) {
    return none_with_error("Workspace not initialized!");
  }

  /* b, c can be None, so don't check is PyArray_Type */
  if (!PyArg_ParseTuple(args, "OO", &b_new, &c_new)) {
    return none_with_error("Error parsing inputs");
  }
  /* set c */
  if ((void *)c_new != Py_None) {
    if (!PyArray_ISFLOAT(c_new) || PyArray_NDIM(c_new) != 1) {
      return none_with_error(
          "c_new must be a dense numpy array with one dimension");
    }
    if ((scs_int)PyArray_DIM(c_new, 0) != self->n) {
      return none_with_error("c_new has incompatible dimension with A");
    }
    c_new = scs_get_contiguous(c_new, scs_float_type);
    c = (scs_float *)PyArray_DATA(c_new);
  }
  /* set b */
  if ((void *)b_new != Py_None) {
    if (!PyArray_ISFLOAT(b_new) || PyArray_NDIM(b_new) != 1) {
      return none_with_error(
          "b must be a dense numpy array with one dimension");
    }
    if (PyArray_DIM(b_new, 0) != self->m) {
      return none_with_error("b_new has incompatible dimension with A");
    }
    b_new = scs_get_contiguous(b_new, scs_float_type);
    b = (scs_float *)PyArray_DATA(b_new);
  }

  /* release the GIL */
  Py_BEGIN_ALLOW_THREADS;
  scs_update(self->work, b, c);
  /* reacquire the GIL */
  Py_END_ALLOW_THREADS;

  Py_DECREF(b_new);
  Py_DECREF(c_new);

  Py_RETURN_NONE;
}

/* Deallocate SCS object */
static scs_int SCS_finish(SCS *self) {
  if (self->work) {
    scs_finish(self->work);
  }
  if (self->sol) {
    scs_free(self->sol->x);
    scs_free(self->sol->y);
    scs_free(self->sol->s);
    scs_free(self->sol);
  }

  /* Del python object */
  PyObject_Del(self);

  return 0;
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
    0,                                        /* tp_new */
};

#endif
