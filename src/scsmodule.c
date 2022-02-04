// Use not deprecated Numpy API (numpy > 1.7)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h" // Python API
#include "glbopts.h"
#include "numpy/arrayobject.h" // Numpy C API
#include "scs.h"               // SCS API
#include "scs_types.h"         // SCS API

// SCS Object type
typedef struct {
  PyObject_HEAD ScsWork *work; /* Workspace */
  ScsSolution *sol;            /* Solution, keep around for warm-starts */
  scs_int m, n;
} SCS;

static PyTypeObject SCS_Type;

#include "scsobject.h" // SCS object

/************************
 * Interface Methods    *
 ************************/

static PyMethodDef scs_module_methods[] = {
    {"version", (PyCFunction)version, METH_NOARGS, "Version number for SCS."},
    {"sizeof_int", (PyCFunction)sizeof_int, METH_NOARGS,
     "Int size (in bytes) SCS uses."},
    {"sizeof_float", (PyCFunction)sizeof_float, METH_NOARGS,
     "Float size (in bytes) SCS uses."},
    {SCS_NULL, SCS_NULL} /* sentinel */
};

/* Module initialization */
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_scs",                                   /* m_name */
    "Solve a convex cone problem using SCS.", /* m_doc */
    -1,                                       /* m_size */
    scs_module_methods,                       /* m_methods */
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
#ifdef PY_INDIRECT
  m = Py_InitModule("_scs_indirect", scs_module_methods);
#elif defined PY_GPU
  m = Py_InitModule("_scs_gpu", scs_module_methods);
#else
  m = Py_InitModule("_scs_direct", scs_module_methods);
#endif
#endif

  if (m == SCS_NULL) {
    return SCS_NULL;
  }

  // Initialize SCS_Type
  SCS_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&SCS_Type) < 0)
    return NULL;

  // Add type to the module dictionary and initialize it
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
#else
init_scs_direct(void)
#endif
{
  import_array(); /* for numpy arrays */
  moduleinit();
}
#endif

static PyMethodDef scs_obj_methods[] = {
    {"solve", (PyCFunction)SCS_solve, METH_VARARGS, PyDoc_STR("Solve problem")},
    {"update_b_c", (PyCFunction)SCS_update_b_c, METH_VARARGS,
     PyDoc_STR("Update b and / or c vector")},
    {SCS_NULL, SCS_NULL} /* sentinel */
};

// Define workspace type object
static PyTypeObject SCS_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "scs.SCS", /*tp_name*/
    sizeof(SCS),                              /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)SCS_dealloc,                  /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                       /*tp_flags*/
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
