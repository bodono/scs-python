"""One SIGINT must interrupt overlapping solves from two different
extensions, and Python's handler must be back afterwards (scs/py_ctrlc.c)."""
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signal semantics"
)

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


def _run(body):
    return subprocess.run(
        [sys.executable, "-c", _COMMON + body],
        capture_output=True, text=True, timeout=120,
    )


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

