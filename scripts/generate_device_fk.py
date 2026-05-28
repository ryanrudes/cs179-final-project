#!/usr/bin/env python3
"""Regenerate CUDA device FK headers from kernels/<robot>_link_task_spatial_local_best/<link>/fk_jacobian.cpp."""

from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
from pathlib import Path

import numpy as np
import pinocchio as pin

from retarget.config import load_retarget_config
from retarget.core import tool_frame_id
from retarget.gpu import load_gpu_fk_model


def convert_cpp_to_device_header(cpp_path: Path, out_path: Path, func_name: str) -> None:
    src = cpp_path.read_text()
    m = re.search(r"void fastfk_task\([^)]*\)\s*\{", src)
    if not m:
        raise SystemExit(f"fastfk_task not found in {cpp_path}")
    body = src[m.start() :]
    body = re.sub(
        r"void fastfk_task\(\s*const double\* q,\s*double\* R,\s*double\* p,\s*double\* J\)\s*\{",
        "",
        body,
        count=1,
    )
    body = re.sub(r"#ifdef __cplusplus.*", "", body, flags=re.DOTALL)
    body = body.rstrip()
    # Drop only the outer ``fastfk_task`` closing brace (keep inner frame ``}``).
    if body.endswith("}"):
        body = body[:-1].rstrip()

    body = body.replace("double", "float")
    body = body.replace("fastfk_fast_sincos", "fastfk_device_sincos")

    header = f'''#pragma once
#include "fastfk_device_math.cuh"

namespace cs179::gpu {{
namespace ur3e {{

constexpr int kTool0JacRows = 6;
constexpr int kTool0JacCols = 6;
constexpr int kTool0JacSize = kTool0JacRows * kTool0JacCols;

/// FK + Pinocchio ``LOCAL`` spatial Jacobian (6 x 6); zero ``J`` before call.
__device__ inline void {func_name}(
    const float* q,
    float* R,
    float* p,
    float* J)
{{
'''
    footer = """
}

}  // namespace ur3e
}  // namespace cs179::gpu
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body + footer)
    print(f"wrote {out_path}")


def verify_spatial_jacobian(cpp_path: Path, *, n_samples: int = 32) -> None:
    so_path = cpp_path.resolve().parents[2] / "build" / "libfastfk_tool0_verify.so"
    so_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-I",
            str(cpp_path.parent),
            str(cpp_path),
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

    max_p = max_r = max_j = 0.0
    for _ in range(n_samples):
        q = pin.neutral(m) + rng.normal(0, 0.5, m.nq)
        qd = (ctypes.c_double * 6)(*q[:6])
        Rc = (ctypes.c_double * 9)()
        pc = (ctypes.c_double * 3)()
        Jc = (ctypes.c_double * 36)()
        lib.fastfk_task(qd, Rc, pc, Jc)
        Rf = np.array(Rc, dtype=np.float64).reshape(3, 3)
        pf = np.array(pc, dtype=np.float64)
        Jf = np.array(Jc, dtype=np.float64).reshape(6, 6)
        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        oM = d.oMf[fid]
        Jpin = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL)
        max_p = max(max_p, float(np.linalg.norm(pf - oM.translation)))
        max_r = max(max_r, float(np.max(np.abs(Rf - oM.rotation))))
        max_j = max(max_j, float(np.max(np.abs(Jf - Jpin))))

    print(
        f"verify spatial_local ({n_samples} samples): "
        f"max |p|={max_p:.3e} max |R|={max_r:.3e} max |J-Jpin|={max_j:.3e}"
    )
    if max_j > 1.0e-5:
        raise SystemExit("spatial_local Jacobian verification failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        default="ur3e_link_task_spatial_local_best",
        help="kernels subdirectory name",
    )
    parser.add_argument("--link", default="tool0")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cuda/include/generated/ur3e_tool0_fk_device.cuh"),
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cpp = root / "kernels" / args.robot / args.link / "fk_jacobian.cpp"
    if not cpp.is_file():
        raise SystemExit(f"kernel not found: {cpp}")
    if not args.no_verify:
        verify_spatial_jacobian(cpp)
    convert_cpp_to_device_header(cpp, args.out, f"{args.link.replace('-', '_')}_fk_jacobian")


if __name__ == "__main__":
    main()
