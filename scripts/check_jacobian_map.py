#!/usr/bin/env python3
"""Compare fastfk 12x6 + W map vs Pinocchio LOCAL 6x6."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pinocchio as pin
from retarget.config import load_retarget_config
from retarget.core import tool_frame_id
from retarget.gpu import load_gpu_fk_model

ROOT = Path(__file__).resolve().parents[1]
W = np.array(
    [
        -7.11953789e-02,
        7.20940605e-02,
        6.27905652e-02,
        3.84199955e-02,
        -1.26523394e-02,
        -6.16128668e-02,
        6.06575459e-02,
        4.97104973e-02,
        3.21939528e-01,
        1.83290467e-02,
        -4.49216552e-02,
        1.13663651e-01,
        1.80079471e-02,
        -3.24067734e-02,
        1.58473500e-03,
        -1.18064724e-01,
        -8.72577727e-02,
        2.40868442e-02,
        1.90967750e-02,
        -5.16201044e-03,
        1.23245083e-02,
        6.42828271e-02,
        1.26887932e-01,
        -1.18171625e-01,
        1.16003461e-01,
        -6.52980357e-02,
        -2.26836521e-02,
        -7.23167881e-02,
        -7.42145861e-03,
        -1.40890628e-01,
        1.58964798e-01,
        1.08246952e-01,
        -2.40554065e-01,
        -2.07736924e-01,
        -1.23732366e-01,
        1.64110605e-02,
        4.09788191e-02,
        8.18059295e-02,
        -4.05878723e-02,
        7.01086044e-01,
        7.90642679e-01,
        -7.02548027e-02,
        -7.34263882e-02,
        1.58809163e-02,
        -5.28312385e-01,
        -3.43412548e-01,
        -5.91144383e-01,
        2.83791989e-01,
        1.82884246e-01,
        8.11265707e-02,
        1.06914371e-01,
        -2.31165618e-01,
        6.80357397e-01,
        -8.78673077e-01,
        -2.01899648e-01,
        4.26700525e-02,
        3.77202362e-01,
        -1.48131654e-01,
        3.65204960e-01,
        3.72086056e-02,
        2.61837864e00,
        -2.34477067e00,
        -1.01788245e-01,
        -1.09807568e01,
        -6.78412676e00,
        1.93229020e00,
        1.21493042e00,
        1.45346773e00,
        -2.40022421e00,
        1.53367147e-01,
        2.59442639e00,
        -6.11331749e00,
    ],
    dtype=np.float64,
).reshape(6, 12)

so_path = ROOT / "build/libfastfk_tool0.so"
if not so_path.exists():
    subprocess.check_call(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT / "kernels/ur3e_link_task_pose_best/tool0"),
            str(ROOT / "kernels/ur3e_link_task_pose_best/tool0/fk_jacobian.cpp"),
            "-o",
            str(so_path),
        ]
    )

lib = ctypes.CDLL(str(so_path))
lib.fastfk_task.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4

m = load_gpu_fk_model()
d = m.createData()
fid = tool_frame_id(m, load_retarget_config().frames.tool)
rng = np.random.default_rng(0)

for trial in range(5):
    q = pin.neutral(m)[:6] + rng.normal(0, 0.3, 6)
    qd = (ctypes.c_double * 6)(*q)
    Jc = (ctypes.c_double * 72)()
    lib.fastfk_task(qd, (ctypes.c_double * 9)(), (ctypes.c_double * 3)(), Jc)
    J = np.array(Jc, dtype=np.float64).reshape(12, 6)

    qq = pin.neutral(m)
    qq[:6] = q
    pin.forwardKinematics(m, d, qq)
    pin.updateFramePlacements(m, d)
    Jpin = pin.computeFrameJacobian(m, d, qq, fid, pin.LOCAL)
    err = np.max(np.abs(W @ J - Jpin))
    print(f"trial {trial}: max |W@J - Jpin| = {err:.3e}")
