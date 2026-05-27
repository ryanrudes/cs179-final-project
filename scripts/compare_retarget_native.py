#!/usr/bin/env python3
"""Compare Python (SciPy) vs native (NLopt) retargeting outputs in detail."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.spatial.transform import Rotation as R

from retarget.core import (
    Retargeter,
    _PythonRetargeter,
    native_retarget_built,
    seed_ik,
    set_use_native_retarget,
    target_to_se3,
    unwrap_euler_targets,
)
from reachability import DirectionalReachEnvelope, scale_cartesian_to_robot
from retarget.core import demo_elbow_side_targets
from rlds import RldsObservationLoader, RoboticsRldsDatasetUrl


@dataclass
class FrameDiff:
    frame: int
    dq: float
    dpos: float
    drot_fro: float
    dpos_err: float
    drot_err: float
    cost_py: float
    cost_native: float
    success_py: bool
    success_native: bool
    nit_py: int
    nit_native: int


def _run_trajectory(
    robot,
    cartesian: np.ndarray,
    radial_scales: np.ndarray,
    elbow_sides: np.ndarray,
    *,
    use_native: bool,
) -> list[tuple]:
    set_use_native_retarget(use_native)
    rt = Retargeter(robot, control_hz=15.0)
    rt.reset_episode(cartesian[0])
    out = []
    for i in range(len(cartesian)):
        rt.set_position_scale(float(radial_scales[i]))
        rt.set_elbow_side_target(float(elbow_sides[i]))
        out.append(rt(cartesian[i]))
    return out


def compare_trajectory(
    robot,
    cartesian: np.ndarray,
    radial_scales: np.ndarray,
    elbow_sides: np.ndarray,
) -> list[FrameDiff]:
    py_out = _run_trajectory(robot, cartesian, radial_scales, elbow_sides, use_native=False)
    native_out = _run_trajectory(robot, cartesian, radial_scales, elbow_sides, use_native=True)

    model = robot.model
    py_rt = _PythonRetargeter(robot, control_hz=15.0)

    diffs: list[FrameDiff] = []
    for i, (py, nat) in enumerate(zip(py_out, native_out, strict=True)):
        py_q, py_pos, py_rot, py_pe, py_re, py_ok, py_nit = py
        n_q, n_pos, n_rot, n_pe, n_re, n_ok, n_nit = nat

        py_rt.set_position_scale(float(radial_scales[i]))
        py_rt.set_elbow_side_target(float(elbow_sides[i]))
        cost_py = py_rt.compute_cost(py_q, cartesian[i])

        dq = float(np.linalg.norm(pin.difference(model, py_q, n_q)))
        dpos = float(np.linalg.norm(py_pos - n_pos))
        R_py = R.from_euler("xyz", py_rot).as_matrix()
        R_nat = R.from_euler("xyz", n_rot).as_matrix()
        drot_fro = float(np.linalg.norm(R_py - R_nat, ord="fro"))

        diffs.append(
            FrameDiff(
                frame=i,
                dq=dq,
                dpos=dpos,
                drot_fro=drot_fro,
                dpos_err=abs(py_pe - n_pe),
                drot_err=abs(py_re - n_re),
                cost_py=cost_py,
                cost_native=float("nan"),  # filled below if we can evaluate
                success_py=bool(py_ok),
                success_native=bool(n_ok),
                nit_py=int(py_nit),
                nit_native=int(n_nit),
            )
        )
        # Evaluate Python cost at native q for apples-to-apples cost gap
        diffs[-1].cost_native = py_rt.compute_cost(n_q, cartesian[i])

    return diffs


def summarize(name: str, diffs: list[FrameDiff]) -> None:
    dq = np.array([d.dq for d in diffs])
    dpos = np.array([d.dpos for d in diffs])
    drot = np.array([d.drot_fro for d in diffs])
    dpe = np.array([d.dpos_err for d in diffs])
    dre = np.array([d.drot_err for d in diffs])
    dcost = np.array([abs(d.cost_py - d.cost_native) for d in diffs])
    success_mismatch = sum(d.success_py != d.success_native for d in diffs)

    print(f"\n=== {name} ({len(diffs)} frames) ===")
    print(f"  dq (rad):     max={dq.max():.6f}  mean={dq.mean():.6f}  p99={np.percentile(dq, 99):.6f}")
    print(f"  dpos (m):     max={dpos.max():.6f}  mean={dpos.mean():.6f}")
    print(f"  drot (fro):   max={drot.max():.6f}  mean={drot.mean():.6f}")
    print(f"  d pos_err:    max={dpe.max():.6f}  mean={dpe.mean():.6f}")
    print(f"  d rot_err:    max={dre.max():.6f}  mean={dre.mean():.6f}")
    print(f"  |cost gap|:   max={dcost.max():.6e}  mean={dcost.mean():.6e}")
    print(f"  success mismatch: {success_mismatch}/{len(diffs)}")
    nit_py = np.array([d.nit_py for d in diffs])
    nit_nat = np.array([d.nit_native for d in diffs])
    print(f"  nit:          py mean={nit_py.mean():.1f}  native mean={nit_nat.mean():.1f}")

    # Flag critical frames
    critical = [
        d
        for d in diffs
        if d.dpos > 0.01 or d.dq > 0.1 or d.dpos_err > 0.01 or d.success_py != d.success_native
    ]
    if critical:
        print(f"  CRITICAL frames (dpos>1cm or dq>0.1rad or dpos_err>1cm or success mismatch): {len(critical)}")
        for d in critical[:8]:
            print(
                f"    frame {d.frame}: dq={d.dq:.4f} dpos={d.dpos:.4f} "
                f"dpe={d.dpos_err:.4f} ok py={d.success_py} nat={d.success_native} "
                f"nit {d.nit_py}/{d.nit_native}"
            )
        if len(critical) > 8:
            print(f"    ... and {len(critical) - 8} more")
    else:
        print("  No critical frame-level divergences under thresholds.")


def compare_seed(robot, targets: np.ndarray) -> None:
    model = robot.model
    data = model.createData()
    q0 = pin.neutral(model)
    fid = model.getFrameId("tool0")
    print("\n=== IK seed (reset_episode) ===")
    seeds = []
    for t in targets:
        q_py = seed_ik(model, data, q0, t, fid)
        set_use_native_retarget(True)
        rt = Retargeter(robot)
        rt.reset_episode(t)
        q_nat = rt.q
        seeds.append(float(np.linalg.norm(pin.difference(model, q_py, q_nat))))
    seeds = np.array(seeds)
    print(f"  seed dq: max={seeds.max():.6f} mean={seeds.mean():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=int, default=0, help="DROID demo index")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames per demo")
    parser.add_argument("--synthetic-frames", type=int, default=20, help="Synthetic trajectory length")
    parser.add_argument(
        "--num-demos",
        type=int,
        default=0,
        help="If >0, sweep DROID demos 0..num_demos-1 (full trajectory, max-frames each).",
    )
    args = parser.parse_args()

    if not native_retarget_built():
        raise SystemExit("Native retargeter not built; run ./scripts/build_loader.sh")

    robot = load_robot_description("ur3e_description")
    panda = load_robot_description("panda_description")

    # Synthetic smooth trajectory in workspace
    t = np.linspace(0, 1, args.synthetic_frames)
    cart = np.column_stack(
        [
            0.35 + 0.05 * np.sin(2 * np.pi * t),
            -0.15 + 0.03 * np.cos(2 * np.pi * t),
            0.45 + 0.02 * t,
            0.1 + 0.05 * t,
            -0.2 + 0.02 * np.sin(np.pi * t),
            0.3 - 0.05 * t,
        ]
    )
    cart[:, 3:6] = unwrap_euler_targets(cart[:, 3:6])
    scales = np.linspace(0.7, 1.0, len(cart))
    elbows = np.ones(len(cart))

    compare_seed(robot, cart[:5])
    summarize("Synthetic trajectory", compare_trajectory(robot, cart, scales, elbows))

    # Cached DROID demo with reach scaling (matches run.py pipeline)
    data_dir = "data"
    loader = RldsObservationLoader(data_dir, str(RoboticsRldsDatasetUrl.DROID_100))
    demo = loader.get_demo(args.demo)
    cart_raw = np.asarray(demo["cartesian_position"], dtype=float)
    joint = np.asarray(demo["joint_position"], dtype=float)
    n = min(args.max_frames, len(cart_raw))
    cart_raw = cart_raw[:n]
    joint = joint[:n]

    ur3e = load_robot_description("ur3e_description")
    reach = DirectionalReachEnvelope.from_robot_cached(
        ur3e.model,
        ur3e.data,
        ur3e.model.getFrameId("tool0"),
        robot_key="ur3e_description",
        n_samples=100_000,
        show_progress=False,
    )
    cart, scales = scale_cartesian_to_robot(cart_raw.copy(), reach, safety=0.9)
    cart[:, 3:6] = unwrap_euler_targets(cart[:, 3:6])
    elbows = demo_elbow_side_targets(joint, panda.model, panda.data)

    summarize(
        f"DROID demo {args.demo} (scaled, {n} frames)",
        compare_trajectory(robot, cart, scales, elbows),
    )

    if args.num_demos > 0:
        print("\n=== DROID demo sweep (full warm-started trajectories) ===")
        print("demo | max_dq | max_dpos(mm) | max_dpe(mm) | frames_dq>0.05")
        ur3e = load_robot_description("ur3e_description")
        reach = DirectionalReachEnvelope.from_robot_cached(
            ur3e.model,
            ur3e.data,
            ur3e.model.getFrameId("tool0"),
            robot_key="ur3e_description",
            n_samples=100_000,
            show_progress=False,
        )
        for demo_id in range(args.num_demos):
            demo = loader.get_demo(demo_id)
            cart_d = np.asarray(demo["cartesian_position"], dtype=float)
            joint_d = np.asarray(demo["joint_position"], dtype=float)
            n_d = min(args.max_frames, len(cart_d))
            cart_d = cart_d[:n_d]
            joint_d = joint_d[:n_d]
            cart_d, scales_d = scale_cartesian_to_robot(cart_d.copy(), reach, safety=0.9)
            cart_d[:, 3:6] = unwrap_euler_targets(cart_d[:, 3:6])
            elbows_d = demo_elbow_side_targets(joint_d, panda.model, panda.data)
            diffs = compare_trajectory(robot, cart_d, scales_d, elbows_d)
            dq = np.array([d.dq for d in diffs])
            dpos = np.array([d.dpos for d in diffs])
            dpe = np.array([d.dpos_err for d in diffs])
            n_bad = int((dq > 0.05).sum())
            print(
                f"{demo_id:4d} | {dq.max():.5f} | {dpos.max()*1000:.2f} | {dpe.max()*1000:.2f} | {n_bad}"
            )


if __name__ == "__main__":
    main()
