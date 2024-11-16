/* Use not deprecated Numpy API (numpy > 1.7) */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/* IMPORTANT: This code now uses numpy array types. It is a private C module
 * in the sense that end users only see the front-facing Python code in
 * "scs.py"; hence, we can get away with the inputs being numpy arrays of
 * the CSC data structures.
 *
 * WARNING: This code also does not check that the data for the sparse
 * matrices are *actually* in column compressed storage for a sparse matrix.
 * The C module is not designed to be used stand-alone. If the data provided
 * does not correspond to a CSR matrix, this code will just crash inelegantly.
 */

#include "Python.h"            /* Python API */
#include "glbopts.h"           /* Constants and *alloc */
#include "numpy/arrayobject.h" /* Numpy C API */
#include "scs.h"               /* SCS API */
#include "scs_types.h"         /* SCS primitive types */

/* The PyInt variable is a PyLong in Python3.x. */
#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
#endif

static PyTypeObject SCS_Type; /* Declare SCS object type */

#include "scsmodule.h" /* SCS module definition */
#include "scsobject.h" /* SCS object definition */
