#!/usr/bin/env python3
"""Align fastfk tool0 FK/Jacobian with Pinocchio tool0 (same URDF as codegen)."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pinocchio as pin

from retarget.config import load_retarget_config
from retarget.core import tool_frame_id
from retarget.gpu import load_gpu_fk_model

ROOT = Path(__file__).resolve().parents[1]
SO = ROOT / "build/libfastfk_tool0.so"
if not SO.exists():
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
            str(SO),
        ]
    )

lib = ctypes.CDLL(str(SO))
lib.fastfk_task.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4

m = load_gpu_fk_model()
d = m.createData()
fid = tool_frame_id(m, load_retarget_config().frames.tool)
q = pin.neutral(m)
qd = (ctypes.c_double * 6)(*q[:6])
Rc = (ctypes.c_double * 9)()
pc = (ctypes.c_double * 3)()
Jc = (ctypes.c_double * 72)()
lib.fastfk_task(qd, Rc, pc, Jc)
R = np.array(Rc).reshape(3, 3)
p = np.array(pc)
J = np.array(Jc).reshape(12, 6)

pin.forwardKinematics(m, d, q)
pin.updateFramePlacements(m, d)
oM = d.oMf[fid]
Rp = oM.rotation
pp = oM.translation
Jpin = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL)

print("FK position |fastfk - pin|", np.linalg.norm(p - pp))
print("FK rotation max |fastfk - pin|", np.max(np.abs(R - Rp)))
print("J pos rows max |Jf[0:3]-Jpin[0:3]|", np.max(np.abs(J[0:3] - Jpin[0:3])))
print("max |Jf[3:6]-Jpin[3:6]|", np.max(np.abs(J[3:6] - Jpin[3:6])))

W, *_ = np.linalg.lstsq(J.T, Jpin.T, rcond=None)
W = W.T
print("lstsq max |W@Jf - Jpin|", np.max(np.abs(W @ J - Jpin)))
