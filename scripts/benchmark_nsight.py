#!/usr/bin/env python3
"""Nsight Compute Benchmarking for GPU Retargeting Kernel.

Usage:
    uv run scripts/benchmark_nsight.py                # profile a 10-demo batch
    uv run scripts/benchmark_nsight.py --demos 100    # profile a custom batch size
    uv run scripts/benchmark_nsight.py --sweep        # profile 10/100/1000 demos and
                                                      # print a utilization scaling table
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DEFAULT_DEMOS = 10
SWEEP_BATCH_SIZES = (10, 100, 1000)

# The metrics we want to collect
METRICS_MAPPING = {
    "gpu__time_duration.sum": "Total Kernel Duration",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "Compute (SM) Throughput %",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy %",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed": "Shared Mem Throughput %",
    "launch__registers_per_thread": "Registers Per Thread",
    "launch__shared_mem_per_block_dynamic": "Dynamic Shared Memory",
}


def run_minimal_benchmark(n_demos: int):
    """Minimal run to isolate the GPU kernel for profiling."""
    from reachability import (
        REACH_BINS_PHI,
        REACH_BINS_THETA,
        REACH_SAFETY_MARGIN,
        REACH_SAMPLE_COUNT_RETARGET,
        DirectionalReachEnvelope,
    )
    from retarget.config import load_retarget_config
    from retarget.core import tool_frame_id
    from retarget.demos import iter_retarget_demos
    from retarget.gpu import (
        load_gpu_fk_model,
        prepare_cartesian_for_gpu_batch,
        retarget_cartesian_trajectories,
        trajectory_fits_gpu_shmem,
    )
    from rlds import RldsObservationLoader
    from robot_descriptions.loaders.pinocchio import load_robot_description

    # Load resources
    loader = RldsObservationLoader(data_dir="data", dataset_url="droid")
    gpu_fk_model = load_gpu_fk_model()
    config = load_retarget_config()
    robot = load_robot_description("ur3e_description")
    reach_data = robot.model.createData()
    ur3e_reach = DirectionalReachEnvelope.from_robot_cached(
        robot.model,
        reach_data,
        tool_frame_id(robot.model, config.frames.tool),
        robot_key="ur3e_description",
        n_samples=REACH_SAMPLE_COUNT_RETARGET,
        n_theta=REACH_BINS_THETA,
        n_phi=REACH_BINS_PHI,
    )

    # Collect exactly n_demos valid demos
    batch = []
    for demo_idx, demo in enumerate(iter_retarget_demos(loader=loader)):
        joint_positions, gripper, cartesian = demo
        if trajectory_fits_gpu_shmem(len(cartesian), gpu_fk_model.nv):
            batch.append((demo_idx, joint_positions, cartesian))
        if len(batch) >= n_demos:
            break

    if len(batch) < n_demos:
        console.print(
            f"[yellow]Only {len(batch)} demos fit GPU shared memory "
            f"(requested {n_demos}); profiling with {len(batch)}.[/yellow]"
        )

    # Prepare data
    cartesian_list, scales_list = prepare_cartesian_for_gpu_batch(
        batch, ur3e_reach, reach_safety=REACH_SAFETY_MARGIN
    )

    # Execute exactly one batch on the GPU (this is the target kernel)
    retarget_cartesian_trajectories(
        gpu_fk_model, cartesian_list, config, position_scales_list=scales_list
    )


def profile_batch(n_demos: int) -> dict | None:
    """Run ncu on a single batch size and return the parsed kernel metrics."""
    metrics_str = ",".join(METRICS_MAPPING.keys())

    python_exe = (
        ".venv/bin/python" if Path(".venv/bin/python").exists() else sys.executable
    )

    cmd = [
        "ncu",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        metrics_str,
        python_exe,
        __file__,
        "--ncu-target",
        "--demos",
        str(n_demos),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        console.print("[bold red]Error running Nsight Compute (ncu)[/bold red]")
        console.print(e.stderr)
        return None

    output = result.stdout

    # Find the CSV section
    csv_lines = []
    in_csv = False
    for line in output.splitlines():
        if line.startswith("ID,") or '"ID"' in line:
            in_csv = True

        if in_csv:
            if line.startswith("==") or not line.strip():
                if csv_lines:  # We reached the end of the CSV block
                    break
                else:
                    continue
            csv_lines.append(line)

    if not csv_lines:
        console.print("[bold red]Could not find CSV output in ncu payload.[/bold red]")
        console.print(output)
        return None

    # Parse the CSV
    reader = csv.DictReader(csv_lines)
    rows = list(reader)

    target_kernel = "retarget_trajectory_kernel"

    target_row = None
    for row in rows:
        kernel_name = row.get("Kernel Name", row.get("Name", ""))
        if target_kernel in kernel_name:
            target_row = row
            break

    if target_row is None:
        console.print(
            f"[bold red]Could not find metrics for kernel: {target_kernel}[/bold red]"
        )
        console.print(
            "[dim]The kernel might not have been executed or was optimized away.[/dim]"
        )
        unique_kernels = sorted(
            list(set(row.get("Kernel Name", row.get("Name", "")) for row in rows))
        )
        console.print(
            f"[yellow]Distinct kernels found in NCU output: {unique_kernels}[/yellow]"
        )
        return None

    # Extract metrics from the target row
    kernel_stats = {}
    for metric_name, display_name in METRICS_MAPPING.items():
        val = target_row.get(metric_name)
        if val is None:
            for k in target_row.keys():
                if metric_name in k:
                    val = target_row[k]
                    break

        if val is not None:
            raw_val = val.replace(",", "")
            if metric_name == "gpu__time_duration.sum":
                try:
                    # Convert nanoseconds to milliseconds
                    ms = float(raw_val) / 1e6
                    val = f"{ms:.3f} ms"
                except ValueError:
                    pass
            elif "pct_of_peak_sustained" in metric_name:
                try:
                    val = f"{float(raw_val):.2f}%"
                except ValueError:
                    val = f"{val}%"
            elif "shared_mem" in metric_name:
                try:
                    kib = float(raw_val) / 1024.0
                    val = f"{val} bytes ({kib:.2f} KiB)"
                except ValueError:
                    pass

            kernel_stats[display_name] = val

    return kernel_stats


def display_single(kernel_stats: dict):
    table = Table(
        title="CUDA Kernel Hardware Metrics (Nsight Compute)", border_style="cyan"
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Kernel Name", "retarget_trajectory_kernel")
    for name, value in kernel_stats.items():
        table.add_row(name, value)

    console.print()
    console.print(Panel(table, border_style="bold blue"))


def display_sweep(results: dict[int, dict]):
    table = Table(
        title="CUDA Kernel Utilization vs Batch Size (Nsight Compute)",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold cyan")
    for n_demos in results:
        table.add_column(f"{n_demos} demos", style="green", justify="right")

    metric_names = list(METRICS_MAPPING.values())
    for name in metric_names:
        table.add_row(name, *[results[n].get(name, "n/a") for n in results])

    console.print()
    console.print(Panel(table, border_style="bold blue"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu-target", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--demos",
        type=int,
        default=DEFAULT_DEMOS,
        help=f"Number of demos in the profiled batch (default {DEFAULT_DEMOS}).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=f"Profile batch sizes {list(SWEEP_BATCH_SIZES)} and print a scaling table.",
    )
    args = parser.parse_args()

    if args.ncu_target:
        run_minimal_benchmark(args.demos)
        return

    batch_sizes = list(SWEEP_BATCH_SIZES) if args.sweep else [args.demos]

    console.print(
        "[cyan]Launching Nsight Compute (ncu) to profile GPU retargeting kernel...[/cyan]"
    )
    console.print(
        "[dim]This may take a minute as ncu serializes kernel execution to collect metrics.[/dim]"
    )

    results: dict[int, dict] = {}
    for n_demos in batch_sizes:
        if len(batch_sizes) > 1:
            console.print(f"[cyan]Profiling batch of {n_demos} demos...[/cyan]")
        stats = profile_batch(n_demos)
        if stats is None:
            return
        results[n_demos] = stats

    if args.sweep:
        display_sweep(results)
    else:
        display_single(results[batch_sizes[0]])


if __name__ == "__main__":
    main()
