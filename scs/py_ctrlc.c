/*
 * Process-shared interrupt (ctrl-c) state for the SCS Python extensions,
 * replacing scs_source/src/ctrlc.c in Python builds.
 *
 * Each extension compiles its own copy of the solver; with per-extension
 * static state, overlapping solves from two extensions restored the SIGINT
 * handler non-LIFO and left a dead SCS handler installed, swallowing SIGINT
 * for the rest of the process. One state struct per process is published as
 * a PyCapsule in a registry module's dict (PyDict_SetDefault, so racing
 * imports agree on it); each extension resolves it once at import time. The
 * solver-facing functions touch no Python API (they run with the GIL
 * released). The registry dict keeps the capsule alive forever, so the
 * borrowed reference PyDict_SetDefault returns is stable.
 */

#include "ctrlc.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>

#if (defined _WIN32 || defined _WIN64 || defined _WINDLL)
#include <windows.h>
typedef struct {
  volatile LONG int_detected;
  CRITICAL_SECTION cs;
  int listener_count;
  PHANDLER_ROUTINE handler;
} scs_py_interrupt_state;
#define SCS_PY_LOCK(s) EnterCriticalSection(&(s)->cs)
#define SCS_PY_UNLOCK(s) LeaveCriticalSection(&(s)->cs)
#else
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
typedef struct {
  /* written by the signal handler, read by every solving thread: a C11
   * atomic is both signal-safe and thread-safe (volatile sig_atomic_t is
   * only the former, and TSan flags it under free-threading) */
  atomic_int int_detected;
  struct sigaction oact;
  pthread_mutex_t mutex;
  int listener_count;
} scs_py_interrupt_state;
#define SCS_PY_LOCK(s) pthread_mutex_lock(&(s)->mutex)
#define SCS_PY_UNLOCK(s) pthread_mutex_unlock(&(s)->mutex)
#endif

#define SCS_CTRLC_REGISTRY "_scs_interrupt_registry"
#define SCS_CTRLC_CAPSULE "scs_interrupt_state_v1"

static scs_py_interrupt_state *shared_state = NULL;

/* Called from each extension's module init with the GIL held. Idempotent;
 * racing imports agree on one state struct. Returns 0 on success. */
int scs_py_ctrlc_init(void) {
  PyObject *registry, *dict, *key, *capsule, *winner;
  scs_py_interrupt_state *state;
  if (shared_state) {
    return 0;
  }
  registry = PyImport_AddModule(SCS_CTRLC_REGISTRY);
  if (!registry) {
    return -1;
  }
  dict = PyModule_GetDict(registry); /* borrowed */
  state = (scs_py_interrupt_state *)calloc(1, sizeof(*state));
  if (!state) {
    PyErr_NoMemory();
    return -1;
  }
#if (defined _WIN32 || defined _WIN64 || defined _WINDLL)
  InitializeCriticalSection(&state->cs);
#else
  pthread_mutex_init(&state->mutex, NULL);
  atomic_init(&state->int_detected, 0);
#endif
  capsule = PyCapsule_New(state, SCS_CTRLC_CAPSULE, NULL);
  key = PyUnicode_FromString(SCS_CTRLC_CAPSULE);
  winner = (capsule && key) ? PyDict_SetDefault(dict, key, capsule) : NULL;
  Py_XDECREF(key);
  if (winner && winner != capsule) {
    free(state); /* another extension's import won the race */
  }
  Py_XDECREF(capsule); /* the registry dict holds the winning capsule */
  if (!winner) {
    return -1;
  }
  shared_state =
      (scs_py_interrupt_state *)PyCapsule_GetPointer(winner, SCS_CTRLC_CAPSULE);
  return shared_state ? 0 : -1;
}

#if (defined _WIN32 || defined _WIN64 || defined _WINDLL)

static BOOL WINAPI scs_handle_ctrlc(DWORD dwCtrlType) {
  if (dwCtrlType != CTRL_C_EVENT) {
    return FALSE;
  }
  InterlockedExchange(&shared_state->int_detected, 1);
  return TRUE;
}

void scs_start_interrupt_listener(void) {
  scs_py_interrupt_state *s = shared_state;
  SCS_PY_LOCK(s);
  if (s->listener_count++ == 0) {
    InterlockedExchange(&s->int_detected, 0);
    s->handler = scs_handle_ctrlc;
    SetConsoleCtrlHandler(s->handler, TRUE);
  }
  SCS_PY_UNLOCK(s);
}

void scs_end_interrupt_listener(void) {
  scs_py_interrupt_state *s = shared_state;
  SCS_PY_LOCK(s);
  if (s->listener_count > 0 && --s->listener_count == 0) {
    /* The last solver may belong to a different extension, whose static
     * handler has a different address. Remove the one we installed. */
    SetConsoleCtrlHandler(s->handler, FALSE);
  }
  SCS_PY_UNLOCK(s);
}

int scs_is_interrupted(void) {
  return (int)InterlockedCompareExchange(&shared_state->int_detected, 0, 0);
}

#else /* POSIX */

static void scs_handle_ctrlc(int sig) {
  atomic_store(&shared_state->int_detected, sig ? sig : -1);
}

void scs_start_interrupt_listener(void) {
  scs_py_interrupt_state *s = shared_state;
  SCS_PY_LOCK(s);
  if (s->listener_count++ == 0) {
    struct sigaction act;
    atomic_store(&s->int_detected, 0);
    act.sa_flags = 0;
    sigemptyset(&act.sa_mask);
    act.sa_handler = scs_handle_ctrlc;
    sigaction(SIGINT, &act, &s->oact);
  }
  SCS_PY_UNLOCK(s);
}

void scs_end_interrupt_listener(void) {
  scs_py_interrupt_state *s = shared_state;
  SCS_PY_LOCK(s);
  if (s->listener_count > 0 && --s->listener_count == 0) {
    sigaction(SIGINT, &s->oact, NULL);
  }
  SCS_PY_UNLOCK(s);
}

int scs_is_interrupted(void) {
  return (int)atomic_load(&shared_state->int_detected);
}

#endif
