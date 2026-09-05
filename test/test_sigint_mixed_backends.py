"""One SIGINT must interrupt overlapping solves from two different
extensions, and Python's handler must be back afterwards (scs/py_ctrlc.c)."""
import subprocess
import sys

import pytest

_COMMON = r"""
import threading, os, signal, time
import numpy as np, scipy.sparse as sp
import scs

def make(n=600):
    rng = np.random.RandomState(0)
    P = sp.csc_matrix(np.eye(n))
    A = sp.vstack([sp.eye(n), sp.csc_matrix(rng.randn(n // 2, n))]).tocsc()
    return (dict(P=P, A=A, b=np.ones(A.shape[0]), c=rng.randn(n)),
            dict(l=A.shape[0]))

DATA, CONE = make()

def make_solver(backend, max_iters, eps):
    # construct in the main thread (numpy/scipy conversions race under
    # free-threading); threads then call only solve()
    return scs.SCS(DATA, CONE, linear_solver=backend, verbose=False,
                   max_iters=max_iters, eps_abs=eps, eps_rel=eps)
"""


def _run(body, **kwargs):
    return subprocess.run(
        [sys.executable, "-c", _COMMON + body],
        capture_output=True, text=True, timeout=120, **kwargs,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_sigint_during_and_after_mixed_backend_overlap():
    body = r"""
results = {}
start_barrier = threading.Barrier(3)

# Construct both solvers in the main thread (see make_solver note);
# eps 0 cannot be met, so the solves can only end via the interrupt.
solvers = {
    "a": make_solver(scs.LinearSolver.QDLDL, max_iters=10**9, eps=0.0),
    "b": make_solver(scs.LinearSolver.CPU_INDIRECT, max_iters=10**9, eps=0.0),
}

def bg(name):
    start_barrier.wait()  # synchronize solver startup with the signaler
    results[name] = solvers[name].solve()

t1 = threading.Thread(target=bg, args=("a",))
t2 = threading.Thread(target=bg, args=("b",))
t1.start(); t2.start()
start_barrier.wait()
time.sleep(1.0)  # both threads are well inside scs_solve by now
os.kill(os.getpid(), signal.SIGINT)
try:
    t1.join(timeout=60); t2.join(timeout=60)
except KeyboardInterrupt:
    t1.join(timeout=60); t2.join(timeout=60)

ok = len(results) == 2 and all(r["info"]["status_val"] == scs.SIGINT for r in results.values())
print("BOTH_INTERRUPTED" if ok else f"BAD: {results}")
# and the handler must be Python's again afterwards
try:
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(1.0)
    print("SWALLOWED")
except KeyboardInterrupt:
    print("RESTORED")
"""
    r = _run(body)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "BOTH_INTERRUPTED" in r.stdout and "RESTORED" in r.stdout, (
        r.stdout, r.stderr)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console handlers")
def test_windows_handler_removed_after_non_lifo_overlap():
    body = r"""
import ctypes, sys

solvers = {
    "a": scs.SCS(DATA, CONE, linear_solver=scs.LinearSolver.QDLDL,
                 verbose=True, max_iters=1),
    "b": scs.SCS(DATA, CONE, linear_solver=scs.LinearSolver.CPU_INDIRECT,
                 verbose=True, max_iters=1),
}
entered = {name: threading.Event() for name in solvers}
release = {name: threading.Event() for name in solvers}
output = sys.stdout

class Gate:
    def write(self, text):
        name = threading.current_thread().name
        if name in entered and not entered[name].is_set():
            # scs_solve prints its header after installing the listener.
            # Pause here to enforce A-start, B-start, A-end, B-end exactly.
            entered[name].set()
            assert release[name].wait(30), "solver release timed out"
        return output.write(text)

    def flush(self):
        output.flush()

sys.stdout = Gate()
threads = {
    name: threading.Thread(name=name, target=solver.solve, daemon=True)
    for name, solver in solvers.items()
}
for name, thread in threads.items():
    thread.start()
    assert entered[name].wait(30), f"{name} did not enter scs_solve"
for name, thread in threads.items():
    release[name].set()
    thread.join(30)
    assert not thread.is_alive(), f"{name} did not finish"
sys.stdout = output

# This child has its own console: the event cannot reach pytest or its runner.
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
try:
    if not kernel32.GenerateConsoleCtrlEvent(signal.CTRL_C_EVENT, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(5)
except KeyboardInterrupt:
    print("RESTORED")
else:
    raise AssertionError("SCS swallowed Ctrl+C after both solves finished")
"""
    r = _run(body, creationflags=subprocess.CREATE_NEW_CONSOLE)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "RESTORED" in r.stdout, (r.stdout, r.stderr)
