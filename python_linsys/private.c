#include <Python.h>
#include "numpy/arrayobject.h"
#include "common.h"
#include "private.h"

// The following are shared with scsmodule.c, which
// sets the callbacks.
extern PyObject *solve_lin_sys_cb;
extern PyObject *accum_by_a_cb;
extern PyObject *accum_by_atrans_cb;
extern int scs_get_float_type(void);

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
  PyObject *x_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, x);

  veclen[0] = A->n;
  PyObject *y_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, y);

  // TODO: Should we not let numpy own the data since we're just
  // using this in a callback?
  PyArray_ENABLEFLAGS((PyArrayObject *)x_np, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)y_np, NPY_ARRAY_OWNDATA);

  PyObject *arglist = Py_BuildValue("(OO)", x_np, y_np);
  PyObject_CallObject(accum_by_atrans_cb, arglist);
}

void SCS(accum_by_a)(const ScsMatrix *A, ScsLinSysWork *p, const scs_float *x,
                     scs_float *y) {
  int scs_float_type = scs_get_float_type();

  npy_intp veclen[1];
  veclen[0] = A->n;
  PyObject *x_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, x);

  veclen[0] = A->m;
  PyObject *y_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, y);

  // TODO: Should we not let numpy own the data since we're just
  // using this in a callback?
  PyArray_ENABLEFLAGS((PyArrayObject *)x_np, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)y_np, NPY_ARRAY_OWNDATA);

  PyObject *arglist = Py_BuildValue("(OO)", x_np, y_np);
  PyObject_CallObject(accum_by_a_cb, arglist);
}

ScsLinSysWork *SCS(init_lin_sys_work)(const ScsMatrix *A,
                                      const ScsSettings *stgs) {
  import_array(); // TODO: Move this somewhere else?

  ScsLinSysWork *p = (ScsLinSysWork *)scs_calloc(1, sizeof(ScsLinSysWork));
  p->total_solve_time = 0;
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
  PyObject *b_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, b);
  PyObject *s_np = PyArray_SimpleNewFromData(1, veclen, scs_float_type, s);

  // TODO: Should we not let numpy own the data since we're just
  // using this in a callback?
  PyArray_ENABLEFLAGS((PyArrayObject *)b_np, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS((PyArrayObject *)s_np, NPY_ARRAY_OWNDATA);

  PyObject *arglist = Py_BuildValue("(OOi)", b_np, s_np, iter);
  PyObject_CallObject(solve_lin_sys_cb, arglist);

  p->total_solve_time += SCS(tocq)(&linsys_timer);
  return 0;
}
