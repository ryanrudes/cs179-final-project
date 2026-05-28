"""Host-side helpers for fastfk ``spatial_local`` tool0 kernels."""

from __future__ import annotations

import ctypes
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pinocchio as pin

from .config import load_retarget_config
from .core import tool_frame_id

_ROOT = Path(__file__).resolve().parents[2]
_FASTFK_CPP = _ROOT / "kernels/ur3e_link_task_spatial_local_best/tool0/fk_jacobian.cpp"
_SO = _ROOT / "build/libfastfk_tool0_host.so"


@lru_cache(maxsize=1)
def _fastfk_lib() -> ctypes.CDLL:
    _SO.parent.mkdir(parents=True, exist_ok=True)
    if not _SO.exists() or _SO.stat().st_mtime < _FASTFK_CPP.stat().st_mtime:
        subprocess.check_call(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-I",
                str(_FASTFK_CPP.parent),
                str(_FASTFK_CPP),
                "-o",
                str(_SO),
            ]
        )
    lib = ctypes.CDLL(str(_SO))
    lib.fastfk_task.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4
    return lib


def max_spatial_jacobian_error(
    model: pin.Model,
    q_samples: np.ndarray,
) -> float:
    """Max ``|J_fastfk - J_pin(LOCAL)|`` over samples (for tests)."""
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim == 1:
        q_samples = q_samples.reshape(1, -1)

    lib = _fastfk_lib()
    d = model.createData()
    fid = tool_frame_id(model, load_retarget_config().frames.tool)
    max_err = 0.0
    for q_row in q_samples:
        q = pin.neutral(model)
        n = min(6, len(q_row))
        q[:n] = q_row[:n]
        qd = (ctypes.c_double * 6)(*q[:6])
        Jc = (ctypes.c_double * 36)()
        lib.fastfk_task(qd, (ctypes.c_double * 9)(), (ctypes.c_double * 3)(), Jc)
        Jf = np.array(Jc, dtype=np.float64).reshape(6, 6)
        pin.forwardKinematics(model, d, q)
        pin.updateFramePlacements(model, d)
        Jpin = pin.computeFrameJacobian(model, d, q, fid, pin.LOCAL)
        max_err = max(max_err, float(np.max(np.abs(Jf - Jpin))))
    return max_err
