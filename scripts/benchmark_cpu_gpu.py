#!/usr/bin/env python3
"""Benchmark CPU vs GPU motion retargeting throughput on cached DROID RLDS demos."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

# Rich imports for beautiful output formatting
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from reachability import (
    REACH_BINS_PHI,
    REACH_BINS_THETA,
    REACH_SAMPLE_COUNT_RETARGET,
    REACH_SAFETY_MARGIN,
    DirectionalReachEnvelope,
    scale_cartesian_to_robot,
)
from retarget.config import load_retarget_config
from retarget.core import (
    Retargeter,
    demo_elbow_side_targets,
    set_use_native_retarget,
    tool_frame_id,
    unwrap_euler_targets,
)
from retarget.demos import iter_retarget_demos
from retarget.gpu import (
    gpu_retarget_built,
    iter_gpu_demo_batches,
    load_gpu_fk_model,
    max_gpu_trajectory_frames,
    prepare_cartesian_for_gpu_batch,
    query_gpu_free_bytes,
    retarget_cartesian_trajectories,
    trajectory_fits_gpu_shmem,
)
from retarget.run import _evaluate_retargeted_demo
from rlds import RldsObservationLoader, RoboticsRldsDatasetUrl

console = Console()


def get_cpu_name() -> str:
    """Query CPU model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        return platform.processor()
    except Exception:
        return "Unknown CPU"


def get_gpu_info() -> dict[str, str]:
    """Query GPU model, driver version, and total VRAM via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().split(",")
        return {
            "name": out[0].strip(),
            "driver": out[1].strip(),
            "memory": f"{int(out[2].strip()):,} MB",
        }
    except Exception:
        return {"name": "NVIDIA GPU (CUDA)", "driver": "N/A", "memory": "N/A"}


# --- Multiprocessing CPU Worker ---
_worker_retargeter = None
_worker_panda_model = None
_worker_panda_data = None
_worker_ur3e_reach = None
_worker_reach_safety = None
_worker_config = None

def _init_cpu_worker(control_hz: float, reach_safety: float, use_native: bool):
    global _worker_retargeter, _worker_panda_model, _worker_panda_data
    global _worker_ur3e_reach, _worker_reach_safety, _worker_config

    set_use_native_retarget(use_native)
    config = load_retarget_config()
    robot = load_robot_description("ur3e_description")
    panda_robot = load_robot_description("panda_description")

    _worker_retargeter = Retargeter(robot, control_hz=control_hz, config=config)
    _worker_panda_model = panda_robot.model
    _worker_panda_data = panda_robot.model.createData()
    _worker_config = config
    _worker_reach_safety = reach_safety

    reach_tool_frame_id = tool_frame_id(robot.model, config.frames.tool)
    reach_data = robot.model.createData()
    _worker_ur3e_reach = DirectionalReachEnvelope.from_robot_cached(
        robot.model,
        reach_data,
        reach_tool_frame_id,
        robot_key="ur3e_description",
        n_samples=REACH_SAMPLE_COUNT_RETARGET,
        n_theta=REACH_BINS_THETA,
        n_phi=REACH_BINS_PHI,
        force_rebuild=False,
    )

def _worker_process_demo(demo_data):
    idx, joint_positions, _, cartesian_positions = demo_data
    
    t0 = time.perf_counter()
    cart = cartesian_positions.copy()
    cart[:, 3:6] = unwrap_euler_targets(cart[:, 3:6])
    cart_scaled, radial_scales = scale_cartesian_to_robot(
        cart, _worker_ur3e_reach, safety=_worker_reach_safety
    )
    demo_elbow_sides = demo_elbow_side_targets(
        joint_positions,
        _worker_panda_model,
        _worker_panda_data,
        frame_names=_worker_config.frames.panda_elbow,
    )
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    _worker_retargeter.reset_episode(cart_scaled[0])
    for frame in range(len(cart_scaled)):
        target = cart_scaled[frame]
        _worker_retargeter.set_position_scale(radial_scales[frame])
        _worker_retargeter.set_elbow_side_target(demo_elbow_sides[frame])
        _worker_retargeter(target)
    t_solve = time.perf_counter() - t0

    return len(cart_scaled), t_prep, t_solve
# ----------------------------------


def run_cpu_benchmark(
    demos: list[tuple[int, np.ndarray, np.ndarray]],
    control_hz: float,
    reach_safety: float,
    use_native: bool,
    desc: str,
) -> dict[str, float]:
    """Benchmark CPU retargeter utilizing all available cores."""
    total_frames = 0
    t_prep_total = 0.0
    t_solve_total = 0.0

    n_workers = os.cpu_count() or 1
    demo_args = [(i, j, g, c) for i, (j, g, c) in enumerate(demos)]

    t_start = time.perf_counter()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Benchmarking {desc} ({n_workers} cores)...", total=len(demos))

        # Use ProcessPoolExecutor to map demos across processes
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_cpu_worker,
            initargs=(control_hz, reach_safety, use_native)
        ) as executor:
            for frames, t_prep, t_solve in executor.map(_worker_process_demo, demo_args):
                total_frames += frames
                t_prep_total += t_prep
                t_solve_total += t_solve
                progress.advance(task)

    t_total = time.perf_counter() - t_start
    return {
        "demos": len(demos),
        "frames": total_frames,
        "prep_s": t_prep_total,
        "solve_s": t_solve_total,
        "total_s": t_total,
    }


def run_gpu_benchmark_subset(
    demos: list[tuple[int, np.ndarray, np.ndarray]],
    gpu_fk_model: pin.Model,
    panda_robot: pin.RobotWrapper,
    ur3e_reach: DirectionalReachEnvelope,
    config,
    control_hz: float,
    reach_safety: float,
    do_eval: bool,
) -> dict[str, float]:
    """Benchmark GPU batch retargeter on the preloaded subset (allowing direct speed comparison)."""
    panda_data = panda_robot.model.createData()

    t_prep_total = 0.0
    t_solve_total = 0.0
    t_eval_total = 0.0
    total_frames = sum(len(c) for _, _, c in demos)

    # Re-package list format for GPU batch helper
    # (iter_gpu_demo_batches accepts (idx, joint, cartesian) tuples)
    gpu_demos_input = [(i, j, c) for i, (j, _, c) in enumerate(demos)]

    t_start = time.perf_counter()

    # 1. Prep Cartesian scaling & Elbow side targets in one batch
    t0 = time.perf_counter()
    cartesian_list, scales_list = prepare_cartesian_for_gpu_batch(
        gpu_demos_input, ur3e_reach, reach_safety=reach_safety
    )

    elbow_list = None
    if config.cost.weights.elbow_branch > 0.0:
        elbow_list = [
            demo_elbow_side_targets(
                joint_positions,
                panda_robot.model,
                panda_data,
                frame_names=config.frames.panda_elbow,
            )
            for _idx, joint_positions, _cartesian in gpu_demos_input
        ]
    t_prep_total += time.perf_counter() - t0

    # 2. Batch GPU Kernel launch (Jacobi projected gradient descent in shared memory)
    t0 = time.perf_counter()
    q_trajs = retarget_cartesian_trajectories(
        gpu_fk_model,
        cartesian_list,
        config,
        position_scales_list=scales_list,
        elbow_targets_list=elbow_list,
    )
    t_solve_total += time.perf_counter() - t0

    # 3. Optional CPU Evaluation FK
    if do_eval:
        t0 = time.perf_counter()
        for (
            _demo_idx,
            _joint_positions,
            _raw_cartesian,
        ), joint_traj, cart_scaled, radial_scales in zip(
            gpu_demos_input, q_trajs, cartesian_list, scales_list, strict=True
        ):
            _evaluate_retargeted_demo(
                model=gpu_fk_model,
                joint_traj=joint_traj,
                cartesian_positions=cart_scaled,
                radial_scales=radial_scales,
                control_hz=control_hz,
                config=config,
                viz=None,
                on_frame=None,
            )
        t_eval_total += time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    return {
        "demos": len(demos),
        "frames": total_frames,
        "prep_s": t_prep_total,
        "solve_s": t_solve_total,
        "eval_s": t_eval_total,
        "total_s": t_total,
    }


def run_gpu_benchmark_stream(
    loader: RldsObservationLoader,
    start_demo: int,
    end_demo: int | None,
    gpu_fk_model: pin.Model,
    panda_robot: pin.RobotWrapper,
    ur3e_reach: DirectionalReachEnvelope,
    config,
    control_hz: float,
    reach_safety: float,
    do_eval: bool,
) -> dict[str, float]:
    """Benchmark GPU batch retargeter streaming through all available demos in the dataset."""
    panda_data = panda_robot.model.createData()
    mem_budget = int(query_gpu_free_bytes() * 0.5)

    n_demos = len(loader)
    stop = n_demos if end_demo is None else min(end_demo, n_demos)
    total_to_process = stop - start_demo

    def demo_stream():
        for demo_idx, demo in enumerate(
            iter_retarget_demos(
                loader=loader, start_demo=start_demo, end_demo=end_demo
            ),
            start=start_demo,
        ):
            joint_positions, _gripper, cartesian = demo
            if trajectory_fits_gpu_shmem(len(cartesian), gpu_fk_model.nv):
                yield demo_idx, joint_positions, cartesian

    t_prep_total = 0.0
    t_solve_total = 0.0
    t_eval_total = 0.0
    total_frames = 0
    total_demos = 0

    t_start = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[green]Benchmarking GPU (Streaming)...", total=total_to_process
        )

        for batch in iter_gpu_demo_batches(
            demo_stream(), n_dof=gpu_fk_model.nv, mem_budget_bytes=mem_budget
        ):
            # 1. Prep Cartesian scaling & Elbow side targets in one batch
            t0 = time.perf_counter()
            cartesian_list, scales_list = prepare_cartesian_for_gpu_batch(
                batch, ur3e_reach, reach_safety=reach_safety
            )

            elbow_list = None
            if config.cost.weights.elbow_branch > 0.0:
                elbow_list = [
                    demo_elbow_side_targets(
                        joint_positions,
                        panda_robot.model,
                        panda_data,
                        frame_names=config.frames.panda_elbow,
                    )
                    for _idx, joint_positions, _cartesian in batch
                ]
            t_prep_total += time.perf_counter() - t0

            # 2. Batch GPU Kernel launch (Jacobi projected gradient descent in shared memory)
            t0 = time.perf_counter()
            q_trajs = retarget_cartesian_trajectories(
                gpu_fk_model,
                cartesian_list,
                config,
                position_scales_list=scales_list,
                elbow_targets_list=elbow_list,
            )
            t_solve_total += time.perf_counter() - t0

            # 3. Optional CPU Evaluation FK
            if do_eval:
                t0 = time.perf_counter()
                for (
                    _demo_idx,
                    _joint_positions,
                    _raw_cartesian,
                ), joint_traj, cart_scaled, radial_scales in zip(
                    batch, q_trajs, cartesian_list, scales_list, strict=True
                ):
                    _evaluate_retargeted_demo(
                        model=gpu_fk_model,
                        joint_traj=joint_traj,
                        cartesian_positions=cart_scaled,
                        radial_scales=radial_scales,
                        control_hz=control_hz,
                        config=config,
                        viz=None,
                        on_frame=None,
                    )
                t_eval_total += time.perf_counter() - t0

            batch_frames = sum(len(c) for c in cartesian_list)
            total_frames += batch_frames
            total_demos += len(batch)
            progress.advance(task, len(batch))

    t_total = time.perf_counter() - t_start
    return {
        "demos": total_demos,
        "frames": total_frames,
        "prep_s": t_prep_total,
        "solve_s": t_solve_total,
        "eval_s": t_eval_total,
        "total_s": t_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="droid",
        help="Dataset directory name (under data-dir).",
    )
    parser.add_argument(
        "--cpu-limit",
        type=int,
        default=50,
        help="Maximum demos to run for sequential CPU benchmarks.",
    )
    parser.add_argument(
        "--gpu-limit",
        type=int,
        default=None,
        help="Maximum demos to run for GPU streaming (default: all).",
    )
    parser.add_argument(
        "--skip-scipy",
        action="store_true",
        help="Skip the slow Python SciPy CPU backend benchmark.",
    )
    args = parser.parse_args()

    # Verify native build options
    if not gpu_retarget_built():
        console.print(
            "[bold red]Error: GPU retarget is not built. "
            "Please rebuild with CUDA support enabled: ./scripts/build_native.sh[/bold red]"
        )
        return

    # Load target robot description structures
    console.print("[yellow]Loading robot descriptions & configurations...[/yellow]")
    robot = load_robot_description("ur3e_description")
    panda_robot = load_robot_description("panda_description")
    gpu_fk_model = load_gpu_fk_model()
    config = load_retarget_config()

    # Load cache loader
    loader = RldsObservationLoader(
        data_dir=args.data_dir, dataset_url=args.dataset
    )
    control_hz = loader.control_hz
    reach_safety = REACH_SAFETY_MARGIN

    # Resolve reach envelope
    reach_tool_frame_id = tool_frame_id(robot.model, config.frames.tool)
    reach_data = robot.model.createData()
    ur3e_reach = DirectionalReachEnvelope.from_robot_cached(
        robot.model,
        reach_data,
        reach_tool_frame_id,
        robot_key="ur3e_description",
        n_samples=REACH_SAMPLE_COUNT_RETARGET,
        n_theta=REACH_BINS_THETA,
        n_phi=REACH_BINS_PHI,
        force_rebuild=False,
    )

    # Print hardware specs and metadata
    cpu_name = get_cpu_name()
    gpu_info = get_gpu_info()

    system_info_table = Table(title="System & Dataset Information", show_header=False)
    system_info_table.add_column("Property", style="bold cyan")
    system_info_table.add_column("Value", style="green")
    system_info_table.add_row("CPU Model", cpu_name)
    system_info_table.add_row("GPU Model", gpu_info["name"])
    system_info_table.add_row("GPU Driver Version", gpu_info["driver"])
    system_info_table.add_row("GPU VRAM Total", gpu_info["memory"])
    system_info_table.add_row("Dataset Name", args.dataset)
    system_info_table.add_row("Total Demos in Cache", f"{len(loader):,}")
    system_info_table.add_row("Control Rate (Hz)", f"{control_hz} Hz")

    console.print(Panel(system_info_table, border_style="bold blue"))

    # 1. Preload demos for CPU & GPU Subset Comparison
    console.print(
        f"[yellow]Preloading first {args.cpu_limit} demos into memory to ensure fair CPU comparisons (no IO overhead)...[/yellow]"
    )
    subset_demos = []
    skipped_subset = 0
    for joint_positions, gripper_positions, cartesian_positions in iter_retarget_demos(
        loader=loader, start_demo=0, end_demo=args.cpu_limit + skipped_subset
    ):
        if not trajectory_fits_gpu_shmem(len(cartesian_positions), gpu_fk_model.nv):
            skipped_subset += 1
            continue
        subset_demos.append((joint_positions, gripper_positions, cartesian_positions))
        if len(subset_demos) >= args.cpu_limit:
            break

    subset_total_frames = sum(len(c) for _, _, c in subset_demos)
    console.print(
        f"[green]Successfully loaded {len(subset_demos)} demos containing {subset_total_frames:,} frames.[/green]"
    )

    results = {}

    # Benchmark: Python CPU (SciPy)
    if not args.skip_scipy:
        # Run 2 demos as warmup
        run_cpu_benchmark(
            subset_demos[:2],
            control_hz,
            reach_safety,
            use_native=False,
            desc="Python CPU Warmup",
        )
        # Run main benchmark
        results["SciPy CPU (1 Core)"] = run_cpu_benchmark(
            subset_demos,
            control_hz,
            reach_safety,
            use_native=False,
            desc="SciPy CPU (Python fallback)",
        )

    # Benchmark: C++ CPU (NLopt)
    # Run 2 demos as warmup
    run_cpu_benchmark(
        subset_demos[:2],
        control_hz,
        reach_safety,
        use_native=True,
        desc="C++ CPU Warmup",
    )
    # Run main benchmark
    n_cores = os.cpu_count() or 1
    results[f"C++ CPU (NLopt) {n_cores} Cores"] = run_cpu_benchmark(
        subset_demos,
        control_hz,
        reach_safety,
        use_native=True,
        desc="C++ CPU (NLopt)",
    )

    # Benchmark: GPU CUDA (Pure Batch)
    # Warmup GPU
    run_gpu_benchmark_subset(
        subset_demos[:2],
        gpu_fk_model,
        panda_robot,
        ur3e_reach,
        config,
        control_hz,
        reach_safety,
        do_eval=False,
    )
    results["GPU CUDA (Pure)"] = run_gpu_benchmark_subset(
        subset_demos,
        gpu_fk_model,
        panda_robot,
        ur3e_reach,
        config,
        control_hz,
        reach_safety,
        do_eval=False,
    )

    # Benchmark: GPU CUDA (with CPU Eval)
    results["GPU CUDA (with CPU Eval)"] = run_gpu_benchmark_subset(
        subset_demos,
        gpu_fk_model,
        panda_robot,
        ur3e_reach,
        config,
        control_hz,
        reach_safety,
        do_eval=True,
    )

    # --- Print Subset Results ---
    subset_table = Table(
        title=f"Direct Comparison on Subset ({len(subset_demos)} demos, {subset_total_frames:,} frames)",
        border_style="cyan",
    )
    subset_table.add_column("Backend", style="bold cyan")
    subset_table.add_column("Prep Time (s)", justify="right")
    subset_table.add_column("Solve Time (s)", justify="right")
    subset_table.add_column("CPU Eval (s)", justify="right")
    subset_table.add_column("Total Time (s)", justify="right")
    subset_table.add_column("Demos/Sec", justify="right", style="green")
    subset_table.add_column("Frames/Sec", justify="right", style="bold green")
    subset_table.add_column("Speedup (vs C++)", justify="right", style="yellow")

    cpp_solve = results[f"C++ CPU (NLopt) {n_cores} Cores"]["solve_s"]

    for name, res in results.items():
        demos_sec = res["demos"] / res["total_s"]
        frames_sec = res["frames"] / res["total_s"]
        speedup = cpp_solve / max(res["solve_s"], 1e-9)

        subset_table.add_row(
            name,
            f"{res['prep_s']:.3f}",
            f"{res['solve_s']:.3f}",
            f"{res.get('eval_s', 0.0):.3f}",
            f"{res['total_s']:.3f}",
            f"{demos_sec:.2f}",
            f"{frames_sec:.1f}",
            f"{speedup:.2f}x (solver)" if name != "SciPy CPU" else "N/A",
        )

    console.print("\n")
    console.print(subset_table)

    # 2. GPU Streaming Benchmark (Complete Dataset or Large Limit)
    console.print("\n")
    max_gpu_str = f"all {len(loader):,}" if args.gpu_limit is None else f"first {args.gpu_limit:,}"
    console.print(
        f"[yellow]Running GPU streaming benchmark over {max_gpu_str} demos of the complete dataset...[/yellow]"
    )

    gpu_stream_res = run_gpu_benchmark_stream(
        loader=loader,
        start_demo=0,
        end_demo=args.gpu_limit,
        gpu_fk_model=gpu_fk_model,
        panda_robot=panda_robot,
        ur3e_reach=ur3e_reach,
        config=config,
        control_hz=control_hz,
        reach_safety=reach_safety,
        do_eval=False,
    )

    # Print GPU Streaming results
    stream_table = Table(
        title=f"GPU Streaming Throughput on Complete Dataset ({gpu_stream_res['demos']:,} demos)",
        border_style="green",
    )
    stream_table.add_column("Metric", style="bold cyan")
    stream_table.add_column("Value", justify="right", style="green")

    stream_table.add_row("Demos Processed", f"{gpu_stream_res['demos']:,}")
    stream_table.add_row("Frames Processed", f"{gpu_stream_res['frames']:,}")
    stream_table.add_row("Prep Time (s)", f"{gpu_stream_res['prep_s']:.3f}")
    stream_table.add_row("Solve Time (s)", f"{gpu_stream_res['solve_s']:.3f}")
    stream_table.add_row("Total Time (s)", f"{gpu_stream_res['total_s']:.3f}")
    stream_table.add_row("Demos / Second", f"{gpu_stream_res['demos'] / gpu_stream_res['total_s']:.2f}")
    stream_table.add_row("Frames / Second", f"{gpu_stream_res['frames'] / gpu_stream_res['total_s']:.1f}")

    console.print(stream_table)

    # Overall Summary Panel
    gpu_solver_only_fps = gpu_stream_res["frames"] / gpu_stream_res["solve_s"]
    cpp_solver_fps = results[f"C++ CPU (NLopt) {n_cores} Cores"]["frames"] / results[f"C++ CPU (NLopt) {n_cores} Cores"]["solve_s"]
    ratio = gpu_solver_only_fps / max(cpp_solver_fps, 1e-9)

    summary_text = (
        f"[bold cyan]Performance Summary:[/bold cyan]\n\n"
        f"• [bold]C++ CPU (NLopt) Solver Rate:[/bold] {cpp_solver_fps:,.1f} frames/sec\n"
        f"• [bold]GPU CUDA (Pure Kernel) Solver Rate:[/bold] {gpu_solver_only_fps:,.1f} frames/sec\n"
        f"• [bold]GPU Solver Speedup:[/bold] [bold green]{ratio:,.1f}x[/bold green] faster than CPU NLopt solver.\n\n"
        f"[yellow]Key Insight:[/yellow] The GPU version leverages Jacobi projected gradient descent "
        f"implemented inside shared memory, computing the full trajectory dynamics in parallel inside a CUDA block. "
        f"This allows for massive speedups. However, running post-retargeting forward kinematics on CPU "
        f"(evaluation mode) will dominate the runtime, so it should be skipped in pure trajectory generation pipelines."
    )
    console.print(Panel(summary_text, title="Conclusions", border_style="bold green"))


if __name__ == "__main__":
    main()
