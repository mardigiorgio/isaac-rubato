#!/usr/bin/env python
"""Verify the isaac-rubato platform is wired correctly.

Asserts, in order:
  1. `import newton` resolves to the custom fork checkout (newton-adaptive),
     NOT Isaac Lab's stock git-pinned upstream Newton.
  2. The adaptive solver SolverMuJoCoAdaptive is importable from newton.solvers.
  3. Isaac Lab + its Newton backend extension import.

Exit 0 on success, 1 on the first failed assertion (with a diagnostic).
Run with the platform venv active, e.g.  `uv run python install/verify.py`.
"""
from __future__ import annotations

import sys
import typing


def fail(msg: str) -> typing.NoReturn:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


# 1. The Newton fork is the active import path (overrides Isaac Lab's pin).
try:
    import newton
except Exception as e:  # pragma: no cover
    fail(f"could not import newton: {e!r}")

newton_path = getattr(newton, "__file__", "") or ""
if "newton-adaptive" not in newton_path:
    fail(
        "the custom Newton fork is NOT active.\n"
        f"  newton resolves to: {newton_path}\n"
        "  expected a path under .../newton-adaptive/ (the adaptive fork).\n"
        "  Isaac Lab's stock pinned Newton is shadowing the fork -- re-run\n"
        "  `uv sync --locked` so the editable path fork is installed last and wins."
    )

# 2. The adaptive solver is present in the fork.
try:
    import newton.solvers as solvers
except Exception as e:
    fail(f"could not import newton.solvers: {e!r}")

if not hasattr(solvers, "SolverMuJoCoAdaptive"):
    fail(
        "SolverMuJoCoAdaptive is missing from newton.solvers -- this Newton\n"
        f"  build ({newton_path}) is not the adaptive fork, or it is out of date."
    )

# 3. Isaac Lab + its Newton backend extension import.
try:
    import isaaclab  # noqa: F401
    import isaaclab_newton  # noqa: F401
except Exception as e:
    fail(
        f"Isaac Lab import failed: {e!r}\n"
        "  Run `./isaaclab.sh -i` against this venv (install step 5)."
    )

print(f"newton   : {newton.__version__}  (fork active, SolverMuJoCoAdaptive present)")
print(f"           {newton_path}")
print("isaaclab : ok  (isaaclab + isaaclab_newton import)")
print("OK: isaac-rubato platform verified.")
