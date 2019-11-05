#include <Python.h>
#include "numpy/arrayobject.h"
#include "private.h"

// The following are shared with scsmodule.c, which
// sets the callbacks and defines helper functions.
extern PyObject *scs_init_lin_sys_work_cb;
extern PyObject *scs_solve_lin_sys_cb;
extern PyObject *scs_accum_by_a_cb;
extern PyObject *scs_accum_by_atrans_cb;
extern PyObject *scs_normalize_a_cb;
extern PyObject *scs_un_normalize_a_cb;

extern int scs_get_float_type(void);
extern int scs_get_int_type(void);
extern PyArrayObject *scs_get_contiguous(PyArrayObject *array, int typenum);

char *SCS(get_lin_sys_method)(const ScsMatrix *A, const ScsSettings *stgs) {
  char *str = (char *)scs_malloc(sizeof(char) * 128);
  sprintf(str, "Python");
  return str;
}

char *SCS(get_lin_sys_summary)(ScsLinSysWork *p, const ScsInfo *info) {
  char *str = (char *)scs_malloc(sizeof(char) * 128);
  sprintf(str,
          "\tLin-sys: avg solve time: %1.2es\n",
          p->total_solve_time / (info->iter + 1) / 1e3);
  p->total_solve_time = 0;
  return str;
}

void SCS(free_lin_sys_work)(ScsLinSysWork *p) {
  if (p) {
    scs_free(p);
  }
}

void SCS(accum_by_atrans)(const ScsMatrix *A, ScsLinSysWork *p,
                          const scs_float *x, scs_float *y) {
  int scs_float_type = scs_get_float_type();

  npy_intp veclen[1];
  veclen[0] = A->m;
  PyObject *x_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, (void *)x);

  veclen[0] = A->n;
  PyObject *y_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, y);

  // TODO: Should we not let numpy own the data since we're just
  // using this in a callback?
  PyArray_ENABLEFLAGS((PyArrayObject *)x_np, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)y_np, NPY_ARRAY_OWNDATA);

  PyObject *arglist = Py_BuildValue("(OO)", x_np, y_np);
  PyObject_CallObject(scs_accum_by_atrans_cb, arglist);
  Py_DECREF(arglist);
}

void SCS(accum_by_a)(const ScsMatrix *A, ScsLinSysWork *p, const scs_float *x,
                     scs_float *y) {
  int scs_float_type = scs_get_float_type();

  npy_intp veclen[1];
  veclen[0] = A->n;
  PyObject *x_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, (void *)x);

  veclen[0] = A->m;
  PyObject *y_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, y);

  // TODO: Should we not let numpy own the data since we're just
  // using this in a callback?
  PyArray_ENABLEFLAGS((PyArrayObject *)x_np, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)y_np, NPY_ARRAY_OWNDATA);

  PyObject *arglist = Py_BuildValue("(OO)", x_np, y_np);
  PyObject_CallObject(scs_accum_by_a_cb, arglist);
  Py_DECREF(arglist);
}

ScsLinSysWork *SCS(init_lin_sys_work)(const ScsMatrix *A,
                                      const ScsSettings *stgs) {
  _import_array();

  ScsLinSysWork *p = (ScsLinSysWork *)scs_calloc(1, sizeof(ScsLinSysWork));
  p->total_solve_time = 0;

#ifdef SFLOAT
  PyObject *arglist = Py_BuildValue("(f)", stgs->rho_x);
#else
  PyObject *arglist = Py_BuildValue("(d)", stgs->rho_x);
#endif
  PyObject_CallObject(scs_init_lin_sys_work_cb, arglist);
  Py_DECREF(arglist);

  return p;
}

scs_int SCS(solve_lin_sys)(const ScsMatrix *A, const ScsSettings *stgs,
                           ScsLinSysWork *p, scs_float *b, const scs_float *s,
                           scs_int iter) {
  SCS(timer) linsys_timer;
  SCS(tic)(&linsys_timer);

  npy_intp veclen[1];
  veclen[0] = A->n + A->m;
  int scs_float_type = scs_get_float_type();
  PyObject *b_py = PyArray_SimpleNewFromData(1, veclen, scs_float_type, b);
  PyArray_ENABLEFLAGS((PyArrayObject *)b_py, NPY_ARRAY_OWNDATA);

  PyObject *s_py = Py_None;
  if (s) {
    s_py = PyArray_SimpleNewFromData(1, veclen, scs_float_type, (void *)s);
    PyArray_ENABLEFLAGS((PyArrayObject *)s_py, NPY_ARRAY_OWNDATA);
  }

  PyObject *arglist = Py_BuildValue("(OOi)", b_py, s_py, iter);
  PyObject_CallObject(scs_solve_lin_sys_cb, arglist);
  Py_DECREF(arglist);

  p->total_solve_time += SCS(tocq)(&linsys_timer);
  return 0;
}


void SCS(normalize_a)(ScsMatrix *A, const ScsSettings *stgs,
                      const ScsCone *k, ScsScaling *scal) {
  _import_array();

  int scs_int_type = scs_get_int_type();
  int scs_float_type = scs_get_float_type();

  scs_int *boundaries;
  npy_intp veclen[1];
  veclen[0] = SCS(get_cone_boundaries)(k, &boundaries);
  PyObject *boundaries_py = PyArray_SimpleNewFromData(
    1, veclen, scs_int_type, boundaries);
  PyArray_ENABLEFLAGS((PyArrayObject *)boundaries_py, NPY_ARRAY_OWNDATA);

#ifdef SFLOAT
  PyObject *arglist = Py_BuildValue("(Of)", boundaries_py, stgs->scale);
#else
  PyObject *arglist = Py_BuildValue("(Od)", boundaries_py, stgs->scale);
#endif
  PyObject *result = PyObject_CallObject(scs_normalize_a_cb, arglist);
  Py_DECREF(arglist);
  scs_free(boundaries);

#ifdef SFLOAT
  char *argparse_string = "O!O!ff";
#else
  char *argparse_string = "O!O!dd";
#endif

  PyArrayObject *D_py = SCS_NULL;
  PyArrayObject *E_py = SCS_NULL;
  PyArg_ParseTuple(result, argparse_string, &PyArray_Type, &D_py,
                   &PyArray_Type, &E_py,
                   &scal->mean_norm_row_a, &scal->mean_norm_col_a);

  D_py = scs_get_contiguous(D_py, scs_float_type);
  E_py = scs_get_contiguous(E_py, scs_float_type);

  scal->D = (scs_float *)PyArray_DATA(D_py);
  scal->E = (scs_float *)PyArray_DATA(E_py);
}


void SCS(un_normalize_a)(ScsMatrix *A, const ScsSettings *stgs,
                         const ScsScaling *scal) {
  int scs_float_type = scs_get_float_type();

  npy_intp veclen[1];
  veclen[0] = A->m;
  PyObject *D_py = PyArray_SimpleNewFromData(1, veclen,
                                             scs_float_type, scal->D);
  PyArray_ENABLEFLAGS((PyArrayObject *)D_py, NPY_ARRAY_OWNDATA);

  veclen[0] = A->n;
  PyObject *E_py = PyArray_SimpleNewFromData(1, veclen,
                                             scs_float_type, scal->E);
  PyArray_ENABLEFLAGS((PyArrayObject *)E_py, NPY_ARRAY_OWNDATA);


#ifdef SFLOAT
  PyObject *arglist = Py_BuildValue("(OOf)", D_py, E_py, stgs->scale);
#else
  PyObject *arglist = Py_BuildValue("(OOd)", D_py, E_py, stgs->scale);
#endif
  PyObject_CallObject(scs_un_normalize_a_cb, arglist);
  Py_DECREF(arglist);
}
