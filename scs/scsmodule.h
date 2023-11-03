#ifndef PY_SCSMODULE_H
#define PY_SCSMODULE_H

static PyObject *version(PyObject *self) {
  return Py_BuildValue("s", scs_version());
}

static PyObject *sizeof_int(PyObject *self) {
  return Py_BuildValue("n", sizeof(scs_int));
}

static PyObject *sizeof_float(PyObject *self) {
  return Py_BuildValue("n", sizeof(scs_float));
}

static PyMethodDef scs_module_methods[] = {
    {"version", (PyCFunction)version, METH_NOARGS, "Version number for SCS."},
    {"sizeof_int", (PyCFunction)sizeof_int, METH_NOARGS,
     "Int size (in bytes) SCS uses."},
    {"sizeof_float", (PyCFunction)sizeof_float, METH_NOARGS,
     "Float size (in bytes) SCS uses."},
    {NULL, NULL} /* sentinel */
};

/* Module initialization */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_scs",                                   /* m_name */
    "Solve a convex cone problem using SCS.", /* m_doc */
    -1,                                       /* m_size */
    scs_module_methods,                       /* m_methods */
    NULL,                                     /* m_reload */
    NULL,                                     /* m_traverse */
    NULL,                                     /* m_clear */
    NULL,                                     /* m_free */
};
#endif

static PyObject *moduleinit(void) {
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
#ifdef PY_INDIRECT
  m = Py_InitModule("_scs_indirect", scs_module_methods);
#elif defined PY_GPU
  m = Py_InitModule("_scs_gpu", scs_module_methods);
#elif defined PY_MKL
  m = Py_InitModule("_scs_mkl", scs_module_methods);
#else
  m = Py_InitModule("_scs_direct", scs_module_methods);
#endif
#endif

  if (m == NULL) {
    return NULL;
  }

  /* Initialize SCS_Type */
  SCS_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&SCS_Type) < 0)
    return NULL;

  /* Add type to the module dictionary and initialize it */
  Py_INCREF(&SCS_Type);
  if (PyModule_AddObject(m, "SCS", (PyObject *)&SCS_Type) < 0)
    return NULL;

  return m;
};

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
#ifdef PY_INDIRECT
PyInit__scs_indirect(void)
#elif defined PY_GPU
PyInit__scs_gpu(void)
#elif defined PY_MKL
PyInit__scs_mkl(void)
#else
PyInit__scs_direct(void)
#endif
{
  import_array(); /* for numpy arrays */
  return moduleinit();
}
#else
PyMODINIT_FUNC
#ifdef PY_INDIRECT
init_scs_indirect(void)
#elif defined PY_GPU
init_scs_gpu(void)
#elif defined PY_MKL
init_scs_mkl(void)
#else
init_scs_direct(void)
#endif
{
  import_array(); /* for numpy arrays */
  moduleinit();
}
#endif

#endif
