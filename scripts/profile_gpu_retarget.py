#!/usr/bin/env python3
"""Profile batched GPU retarget phases on cached RLDS demos."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from reachability import (
    REACH_BINS_PHI,
    REACH_BINS_THETA,
    REACH_SAMPLE_COUNT_RETARGET,
    REACH_SAFETY_MARGIN,
    DirectionalReachEnvelope,
)
from retarget.config import load_retarget_config
from retarget.core import demo_elbow_side_targets, tool_frame_id
from retarget.demos import iter_retarget_demos
from retarget.gpu import (
    gpu_retarget_built,
    load_gpu_fk_model,
    max_gpu_trajectory_frames,
    pack_targets,
    pack_initial_gpu_q,
    pack_targets,
    retarget_cartesian_trajectories,
    retarget_params_from_config,
    retarget_trajectories_gpu,
    trajectory_fits_gpu_shmem,
)
from retarget.run import _evaluate_retargeted_demo, _prepare_demo_cartesian
from rlds import RldsObservationLoader, RoboticsRldsDatasetUrl


def _load_batch(
    *,
    data_dir: Path,
    start_demo: int,
    end_demo: int | None,
    n_dof: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    loader = RldsObservationLoader(data_dir=data_dir, dataset_url=RoboticsRldsDatasetUrl.DROID_100)
    demos: list[tuple[int, np.ndarray, np.ndarray]] = []
    skipped = 0
    for demo_idx, (joint_positions, _gripper, cartesian_positions) in enumerate(
        iter_retarget_demos(loader=loader, start_demo=start_demo, end_demo=end_demo),
        start=start_demo,
    ):
        if not trajectory_fits_gpu_shmem(len(cartesian_positions), n_dof):
            skipped += 1
            continue
        demos.append((demo_idx, joint_positions, cartesian_positions))
    if skipped:
        print(f"(profile) skipped {skipped} demo(s) over T_pad limit {max_gpu_trajectory_frames(n_dof)}")
    return demos


def _timed_phases(
    demos: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    reach_safety: float,
) -> dict[str, float]:
    if not gpu_retarget_built():
        raise RuntimeError("CUDA _native not built")

    config = load_retarget_config()
    gpu_model = load_gpu_fk_model()
    panda = load_robot_description("panda_description")
    panda_data = panda.model.createData()
    reach_data = gpu_model.createData()
    tool_fid = tool_frame_id(gpu_model, config.frames.tool)
    ur3e_reach = DirectionalReachEnvelope.from_robot_cached(
        gpu_model,
        reach_data,
        tool_fid,
        robot_key="ur3e_description",
        n_samples=REACH_SAMPLE_COUNT_RETARGET,
        n_theta=REACH_BINS_THETA,
        n_phi=REACH_BINS_PHI,
        force_rebuild=False,
    )

    cartesian_list: list[np.ndarray] = []
    scales_list: list[np.ndarray] = []
    elbow_list: list[np.ndarray] = []
    total_frames = 0

    t0 = time.perf_counter()
    for _demo_idx, joint_positions, cartesian_positions in demos:
        cartesian_positions, radial_scales = _prepare_demo_cartesian(
            cartesian_positions, ur3e_reach, reach_safety
        )
        elbow_sides = demo_elbow_side_targets(
            joint_positions,
            panda.model,
            panda_data,
            frame_names=config.frames.panda_elbow,
        )
        cartesian_list.append(cartesian_positions)
        scales_list.append(radial_scales)
        elbow_list.append(elbow_sides)
        total_frames += len(cartesian_positions)
    t_prepare = time.perf_counter() - t0

    t_lens = [int(t.shape[0]) for t in cartesian_list]
    q_in, lengths, _d_pad, t_pad = pack_initial_gpu_q(gpu_model, t_lens)
    tgt_batch = pack_targets(cartesian_list, t_pad=t_pad)

    t_seed = 0.0

    n_dof = gpu_model.nv
    kwargs = retarget_params_from_config(gpu_model, config, t_pad=t_pad, n_dof=n_dof)
    scales_batch = np.zeros((len(cartesian_list), t_pad), dtype=np.float32)
    for i, (scales, t_len) in enumerate(zip(scales_list, t_lens, strict=True)):
        scales_batch[i, :t_len] = np.asarray(scales[:t_len], dtype=np.float32)

    t0 = time.perf_counter()
    q_out = retarget_trajectories_gpu(
        q_in,
        tgt_batch,
        lengths,
        scales_batch,
        **kwargs,
    )
    t_kernel = time.perf_counter() - t0

    q_trajs = [
        np.asarray(q_out[i, :n_dof, :t_len], dtype=np.float32).T
        for i, t_len in enumerate(t_lens)
    ]

    t0 = time.perf_counter()
    for (_demo_idx, _joint, cartesian_positions), joint_traj, radial_scales in zip(
        demos, q_trajs, scales_list, strict=True
    ):
        _evaluate_retargeted_demo(
            model=gpu_model,
            joint_traj=joint_traj,
            cartesian_positions=cartesian_positions,
            radial_scales=radial_scales,
            control_hz=float(config.control_hz),
            config=config,
            viz=None,
            on_frame=None,
        )
    t_eval = time.perf_counter() - t0

    return {
        "prepare_s": t_prepare,
        "seed_ik_s": t_seed,  # always 0 (neutral init on device path)
        "gpu_kernel_s": t_kernel,
        "evaluate_s": t_eval,
        "total_frames": float(total_frames),
        "n_demos": float(len(demos)),
        "t_pad": float(t_pad),
    }


def _print_timings(stats: dict[str, float]) -> None:
    n = int(stats["n_demos"])
    frames = int(stats["total_frames"])
    parts = {
        "prepare (reach+elbow)": stats["prepare_s"],
        "initial q pack": stats["seed_ik_s"],
        "GPU kernel": stats["gpu_kernel_s"],
        "post eval FK": stats["evaluate_s"],
    }
    total = sum(parts.values())
    print(f"\n=== GPU retarget profile ({n} demos, {frames} frames, batch T_pad={int(stats['t_pad'])}) ===")
    for name, sec in parts.items():
        pct = 100.0 * sec / total if total > 0 else 0.0
        per_frame_ms = 1000.0 * sec / frames if frames else 0.0
        print(f"  {name:24s} {sec:8.2f}s  ({pct:5.1f}%)  {per_frame_ms:6.2f} ms/frame")
    print(f"  {'TOTAL (timed phases)':24s} {total:8.2f}s")
    print(f"  seed_ik per frame: {1000.0 * stats['seed_ik_s'] / frames:.2f} ms")
    print(f"  GPU kernel per demo: {1000.0 * stats['gpu_kernel_s'] / n:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start-demo", type=int, default=0)
    parser.add_argument("--end-demo", type=int, default=None)
    parser.add_argument("--cprofile", action="store_true", help="Run cProfile on prepare+seed")
    args = parser.parse_args()

    gpu_model = load_gpu_fk_model()
    demos = _load_batch(
        data_dir=args.data_dir,
        start_demo=args.start_demo,
        end_demo=args.end_demo,
        n_dof=gpu_model.nv,
    )
    if not demos:
        print("No demos to profile.")
        return

    if args.cprofile:
        pr = cProfile.Profile()

        def _run() -> None:
            _timed_phases(demos, reach_safety=REACH_SAFETY_MARGIN)

        pr.enable()
        _run()
        pr.disable()
        buf = StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
        print(buf.getvalue())
    else:
        # warmup one short trajectory (CUDA JIT / first Pinocchio)
        if demos:
            warm = [demos[0]]
            _timed_phases(warm, reach_safety=REACH_SAFETY_MARGIN)

        stats = _timed_phases(demos, reach_safety=REACH_SAFETY_MARGIN)
        _print_timings(stats)

        # Compare 1-demo vs full batch kernel scaling
        one = [demos[0]]
        s1 = _timed_phases(one, reach_safety=REACH_SAFETY_MARGIN)
        print(
            f"\n=== 1-demo reference (demo {one[0][0]}, {int(s1['total_frames'])} frames) ==="
        )
        _print_timings(s1)
        print(
            f"\nKernel scaling: {s1['gpu_kernel_s']:.3f}s (1 demo) vs "
            f"{stats['gpu_kernel_s']:.3f}s ({int(stats['n_demos'])} demos) — "
            f"ratio {stats['gpu_kernel_s'] / max(s1['gpu_kernel_s'], 1e-9):.2f}x"
        )


if __name__ == "__main__":
    main()
