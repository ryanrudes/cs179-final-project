"""Aggregate retargeting metrics for headless batch runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

LIVE_REFRESH_EVERY_N_FRAMES = 20

from .core import (
    JOINT_ACC_WEIGHT,
    JOINT_VEL_WEIGHT,
    POS_WEIGHT,
    ROT_WEIGHT,
    ROT_WEIGHT_MIN_SCALE,
)

# Reference thresholds when reporting fraction of bad frames (tuning guide).
POS_WARN_M = 0.02
POS_BAD_M = 0.05
ROT_WARN_RAD = float(np.deg2rad(10.0))
ROT_BAD_RAD = float(np.deg2rad(20.0))
JOINT_SPEED_WARN_RAD_S = float(np.pi)


@dataclass(frozen=True)
class ScalarStats:
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_array(cls, values: np.ndarray) -> ScalarStats:
        x = np.asarray(values, dtype=np.float64).ravel()
        if x.size == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            mean=float(np.mean(x)),
            std=float(np.std(x)),
            min=float(np.min(x)),
            max=float(np.max(x)),
            p50=float(np.percentile(x, 50)),
            p95=float(np.percentile(x, 95)),
            p99=float(np.percentile(x, 99)),
        )


@dataclass
class DemoStats:
    demo_idx: int
    n_frames: int
    position_error_m: ScalarStats
    rotation_error_rad: ScalarStats
    radial_scale: ScalarStats
    joint_speed_rad_s: ScalarStats
    ik_iterations: ScalarStats
    ik_success_rate: float
    frac_pos_above_warn: float
    frac_pos_above_bad: float
    frac_rot_above_warn: float
    frac_rot_above_bad: float
    frac_joint_speed_above_warn: float


def _frac_above(values: np.ndarray, threshold: float) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(np.mean(x > threshold))


def compute_demo_stats(
    *,
    demo_idx: int,
    position_errors: np.ndarray,
    rotation_errors: np.ndarray,
    radial_scales: np.ndarray,
    joint_speeds: np.ndarray,
    ik_success: np.ndarray,
    ik_iterations: np.ndarray,
) -> DemoStats:
    return DemoStats(
        demo_idx=demo_idx,
        n_frames=int(len(position_errors)),
        position_error_m=ScalarStats.from_array(position_errors),
        rotation_error_rad=ScalarStats.from_array(rotation_errors),
        radial_scale=ScalarStats.from_array(radial_scales),
        joint_speed_rad_s=ScalarStats.from_array(joint_speeds),
        ik_iterations=ScalarStats.from_array(ik_iterations),
        ik_success_rate=float(np.mean(ik_success)) if len(ik_success) else 0.0,
        frac_pos_above_warn=_frac_above(position_errors, POS_WARN_M),
        frac_pos_above_bad=_frac_above(position_errors, POS_BAD_M),
        frac_rot_above_warn=_frac_above(rotation_errors, ROT_WARN_RAD),
        frac_rot_above_bad=_frac_above(rotation_errors, ROT_BAD_RAD),
        frac_joint_speed_above_warn=_frac_above(joint_speeds, JOINT_SPEED_WARN_RAD_S),
    )


@dataclass
class FrameArrays:
    position_errors: np.ndarray
    rotation_errors: np.ndarray
    radial_scales: np.ndarray
    joint_speeds: np.ndarray
    ik_success: np.ndarray
    ik_iterations: np.ndarray


@dataclass
class BatchStatsAccumulator:
    """Concatenate frame-level metrics across demos for global summaries."""

    _frames: list[FrameArrays] = field(default_factory=list)
    demo_stats: list[DemoStats] = field(default_factory=list)

    def add_demo(self, demo: DemoStats, frames: FrameArrays) -> None:
        self.demo_stats.append(demo)
        self._frames.append(frames)

    @property
    def total_frames(self) -> int:
        return sum(f.position_errors.size for f in self._frames)

    @property
    def n_demos(self) -> int:
        return len(self.demo_stats)

    def pooled(self, name: str) -> np.ndarray:
        return np.concatenate([getattr(f, name) for f in self._frames])

    def global_demo_stats(self) -> DemoStats:
        return self.compute_running_stats()

    def compute_running_stats(self, partial: PartialFrameBuffer | None = None) -> DemoStats | None:
        arrays = _concat_frame_metrics(self._frames, partial.to_arrays() if partial else None)
        if arrays is None:
            return None
        return compute_demo_stats(demo_idx=-1, **arrays)


@dataclass
class PartialFrameBuffer:
    """Frame metrics for the demo currently being retargeted."""

    position_errors: list[float] = field(default_factory=list)
    rotation_errors: list[float] = field(default_factory=list)
    radial_scales: list[float] = field(default_factory=list)
    joint_speeds: list[float] = field(default_factory=list)
    ik_success: list[bool] = field(default_factory=list)
    ik_iterations: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.position_errors)

    def clear(self) -> None:
        self.position_errors.clear()
        self.rotation_errors.clear()
        self.radial_scales.clear()
        self.joint_speeds.clear()
        self.ik_success.clear()
        self.ik_iterations.clear()

    def push(
        self,
        *,
        position_error: float,
        rotation_error: float,
        radial_scale: float,
        joint_speed: float,
        ik_success: bool,
        ik_iterations: int,
    ) -> None:
        self.position_errors.append(position_error)
        self.rotation_errors.append(rotation_error)
        self.radial_scales.append(radial_scale)
        self.joint_speeds.append(joint_speed)
        self.ik_success.append(ik_success)
        self.ik_iterations.append(ik_iterations)

    def to_arrays(self) -> FrameArrays | None:
        if not self.position_errors:
            return None
        return FrameArrays(
            position_errors=np.asarray(self.position_errors, dtype=np.float64),
            rotation_errors=np.asarray(self.rotation_errors, dtype=np.float64),
            radial_scales=np.asarray(self.radial_scales, dtype=np.float64),
            joint_speeds=np.asarray(self.joint_speeds, dtype=np.float64),
            ik_success=np.asarray(self.ik_success, dtype=bool),
            ik_iterations=np.asarray(self.ik_iterations, dtype=np.int64),
        )


def _concat_frame_metrics(
    completed: list[FrameArrays],
    partial: FrameArrays | None,
) -> dict[str, np.ndarray] | None:
    chunks: dict[str, list[np.ndarray]] = {
        "position_errors": [],
        "rotation_errors": [],
        "radial_scales": [],
        "joint_speeds": [],
        "ik_success": [],
        "ik_iterations": [],
    }
    for frame in completed:
        for key in chunks:
            chunks[key].append(getattr(frame, key))
    if partial is not None:
        for key in chunks:
            chunks[key].append(getattr(partial, key))
    if not chunks["position_errors"]:
        return None
    return {key: np.concatenate(values) for key, values in chunks.items()}


def _format_compact_stats_table(title: str, stats: DemoStats) -> Table:
    table = Table(title=title, show_header=True, header_style="bold", expand=True)
    table.add_column("Metric", ratio=2)
    table.add_column("Mean", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Max", justify="right")
    table.add_row(
        "Position err",
        f"{stats.position_error_m.mean:.4f} m",
        f"{stats.position_error_m.p95:.4f} m",
        f"{stats.position_error_m.max:.4f} m",
    )
    table.add_row(
        "Rotation err",
        f"{stats.rotation_error_rad.mean:.4f} rad",
        f"{stats.rotation_error_rad.p95:.4f} rad",
        f"{stats.rotation_error_rad.max:.4f} rad",
    )
    table.add_row(
        "Workspace scale",
        f"{stats.radial_scale.mean:.3f}×",
        f"{stats.radial_scale.p95:.3f}×",
        f"{stats.radial_scale.min:.3f}× min",
    )
    table.add_row(
        "Joint speed",
        f"{stats.joint_speed_rad_s.mean:.3f} rad/s",
        f"{stats.joint_speed_rad_s.p95:.3f} rad/s",
        f"{stats.joint_speed_rad_s.max:.3f} rad/s",
    )
    table.add_row(
        "IK iterations",
        f"{stats.ik_iterations.mean:.1f}",
        f"{stats.ik_iterations.p95:.0f}",
        f"{stats.ik_iterations.max:.0f}",
    )
    return table


def format_live_summary_panel(
    running: DemoStats,
    *,
    demos_completed: int,
    demo_count: int,
    total_frames: int,
    control_hz: float,
    current_demo_idx: int | None = None,
    current_frame: int | None = None,
    current_demo_frames: int | None = None,
    last_demo: DemoStats | None = None,
) -> Panel:
    if current_demo_idx is not None and current_frame is not None and current_demo_frames is not None:
        status = (
            f"Demo [cyan]{current_demo_idx}[/] frame {current_frame + 1}/{current_demo_frames} · "
            f"Completed {demos_completed}/{demo_count} demos · {total_frames:,} frames · hz={control_hz:g}"
        )
    else:
        status = (
            f"Completed {demos_completed}/{demo_count} demos · {total_frames:,} frames · hz={control_hz:g}"
        )

    tables = Group(
        _format_compact_stats_table("Running totals (all finished frames + current demo)", running),
        _format_live_rates_table(running),
    )
    if last_demo is not None:
        tables = Group(
            tables,
            _format_compact_stats_table(f"Last demo {last_demo.demo_idx}", last_demo),
        )
    return Panel(tables, title="[bold]Live retarget stats[/bold]", subtitle=status)


def _format_live_rates_table(stats: DemoStats) -> Table:
    table = Table(title="Warning rates (running)", show_header=True, header_style="bold", expand=True)
    table.add_column("Check", ratio=3)
    table.add_column("Rate", justify="right")
    table.add_row("IK success", f"{100.0 * stats.ik_success_rate:.1f}%")
    table.add_row(f"Pos > {POS_WARN_M * 1000:.0f} mm", f"{100.0 * stats.frac_pos_above_warn:.1f}%")
    table.add_row(f"Pos > {POS_BAD_M * 1000:.0f} mm", f"{100.0 * stats.frac_pos_above_bad:.1f}%")
    table.add_row(f"Rot > {np.rad2deg(ROT_WARN_RAD):.0f}°", f"{100.0 * stats.frac_rot_above_warn:.1f}%")
    table.add_row(
        f"‖dq‖·hz > {JOINT_SPEED_WARN_RAD_S:.2f} rad/s",
        f"{100.0 * stats.frac_joint_speed_above_warn:.1f}%",
    )
    return table


class LiveRetargetDisplay:
    """Rich Live layout: progress bars plus running pooled statistics."""

    def __init__(
        self,
        *,
        batch: BatchStatsAccumulator,
        demo_count: int,
        control_hz: float,
    ) -> None:
        self.batch = batch
        self.demo_count = demo_count
        self.control_hz = control_hz
        self.partial = PartialFrameBuffer()
        self.last_demo: DemoStats | None = None
        self._current_demo_idx: int | None = None
        self._current_demo_frames: int = 0
        self._frames_since_refresh = 0
        self.console = Console()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._live: Live | None = None

    def __enter__(self) -> LiveRetargetDisplay:
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        self.progress.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.progress.__exit__(*args)
        if self._live is not None:
            self._live.__exit__(*args)

    def add_demos_task(self) -> int:
        return self.progress.add_task("Retargeting demos", total=self.demo_count)

    def add_frames_task(self, demo_idx: int, total: int) -> int:
        self._current_demo_idx = demo_idx
        self._current_demo_frames = total
        self.partial.clear()
        self._frames_since_refresh = 0
        return self.progress.add_task(f"Demo {demo_idx}", total=total)

    def remove_frames_task(self, task_id: int) -> None:
        self.progress.remove_task(task_id)
        self._current_demo_idx = None
        self.partial.clear()

    def advance_frame(
        self,
        frames_task: int,
        *,
        position_error: float,
        rotation_error: float,
        radial_scale: float,
        joint_speed: float,
        ik_success: bool,
        ik_iterations: int,
    ) -> None:
        self.progress.advance(frames_task)
        self.partial.push(
            position_error=position_error,
            rotation_error=rotation_error,
            radial_scale=radial_scale,
            joint_speed=joint_speed,
            ik_success=ik_success,
            ik_iterations=ik_iterations,
        )
        self._frames_since_refresh += 1
        if len(self.partial) == 1 or self._frames_since_refresh >= LIVE_REFRESH_EVERY_N_FRAMES:
            self._frames_since_refresh = 0
            self.refresh()

    def finish_demo(
        self,
        demo_stats: DemoStats,
        frames: FrameArrays,
        *,
        refresh: bool = True,
    ) -> None:
        self.batch.add_demo(demo_stats, frames)
        self.last_demo = demo_stats
        self.partial.clear()
        self._current_demo_idx = None
        self._current_demo_frames = 0
        if refresh:
            self.refresh()

    def refresh(self) -> None:
        if self._live is None:
            return
        self._live.update(self._render(), refresh=True)

    def _render(self) -> RenderableType:
        running = self.batch.compute_running_stats(self.partial)
        summary: RenderableType
        if running is None:
            summary = Panel(
                "[dim]Waiting for first frame…[/dim]",
                title="[bold]Live retarget stats[/bold]",
            )
        else:
            summary = format_live_summary_panel(
                running,
                demos_completed=self.batch.n_demos,
                demo_count=self.demo_count,
                total_frames=self.batch.total_frames + len(self.partial),
                control_hz=self.control_hz,
                current_demo_idx=self._current_demo_idx,
                current_frame=len(self.partial) if len(self.partial) else None,
                current_demo_frames=self._current_demo_frames or None,
                last_demo=self.last_demo,
            )
        return Group(self.progress, summary)


def _add_scalar_row(table: Table, label: str, stats: ScalarStats, unit: str) -> None:
    table.add_row(
        label,
        f"{stats.mean:.4f}",
        f"{stats.std:.4f}",
        f"{stats.min:.4f}",
        f"{stats.max:.4f}",
        f"{stats.p50:.4f}",
        f"{stats.p95:.4f}",
        f"{stats.p99:.4f}",
        unit,
    )


def _format_demo_stats_table(title: str, stats: DemoStats) -> Table:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Mean")
    table.add_column("Std")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("P50")
    table.add_column("P95")
    table.add_column("P99")
    table.add_column("Unit")
    _add_scalar_row(table, "Position error", stats.position_error_m, "m")
    _add_scalar_row(table, "Rotation error", stats.rotation_error_rad, "rad")
    _add_scalar_row(table, "Workspace scale", stats.radial_scale, "×")
    _add_scalar_row(table, "Joint speed ‖dq‖·hz", stats.joint_speed_rad_s, "rad/s")
    _add_scalar_row(table, "IK iterations", stats.ik_iterations, "iters")
    return table


def _format_rates_table(title: str, stats: DemoStats) -> Table:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("IK success rate", f"{100.0 * stats.ik_success_rate:.2f}%")
    table.add_row(
        f"Frames with pos error > {POS_WARN_M:.3f} m (warn)",
        f"{100.0 * stats.frac_pos_above_warn:.2f}%",
    )
    table.add_row(
        f"Frames with pos error > {POS_BAD_M:.3f} m (bad)",
        f"{100.0 * stats.frac_pos_above_bad:.2f}%",
    )
    table.add_row(
        f"Frames with rot error > {np.rad2deg(ROT_WARN_RAD):.1f}° (warn)",
        f"{100.0 * stats.frac_rot_above_warn:.2f}%",
    )
    table.add_row(
        f"Frames with rot error > {np.rad2deg(ROT_BAD_RAD):.1f}° (bad)",
        f"{100.0 * stats.frac_rot_above_bad:.2f}%",
    )
    table.add_row(
        f"Frames with joint speed > {JOINT_SPEED_WARN_RAD_S:.3f} rad/s (warn)",
        f"{100.0 * stats.frac_joint_speed_above_warn:.2f}%",
    )
    return table


def _format_worst_demos_table(batch: BatchStatsAccumulator, *, top_k: int = 5) -> Table | None:
    if not batch.demo_stats:
        return None
    ranked = sorted(
        batch.demo_stats,
        key=lambda d: d.position_error_m.p95,
        reverse=True,
    )[:top_k]
    table = Table(
        title=f"Worst demos by P95 position error (top {len(ranked)})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Demo")
    table.add_column("Frames")
    table.add_column("Pos P95 (m)")
    table.add_column("Pos max (m)")
    table.add_column("Rot P95 (rad)")
    table.add_column("Scale min")
    table.add_column("IK ok %")
    for d in ranked:
        table.add_row(
            str(d.demo_idx),
            str(d.n_frames),
            f"{d.position_error_m.p95:.4f}",
            f"{d.position_error_m.max:.4f}",
            f"{d.rotation_error_rad.p95:.4f}",
            f"{d.radial_scale.min:.3f}",
            f"{100.0 * d.ik_success_rate:.1f}",
        )
    return table


def print_batch_summary(
    batch: BatchStatsAccumulator,
    *,
    control_hz: float,
    start_demo: int,
    end_demo: int | None,
) -> None:
    if batch.n_demos == 0:
        return

    console = Console()
    global_stats = batch.global_demo_stats()
    range_label = f"{start_demo}..{end_demo}" if end_demo is not None else f"{start_demo}..end"

    console.print()
    console.print(
        f"[bold]Retarget summary[/bold] — {batch.n_demos} demo(s), "
        f"{batch.total_frames:,} frame(s), demos {range_label}, control_hz={control_hz:g}"
    )
    console.print(_format_demo_stats_table("All frames (pooled)", global_stats))
    console.print(_format_rates_table("Failure / warning rates (pooled)", global_stats))

    worst = _format_worst_demos_table(batch)
    if worst is not None:
        console.print(worst)

    console.print(
        "[dim]Tuning knobs in config/default.yaml → cost.weights "
        f"(cs179 retarget --config): position={POS_WEIGHT}, rotation={ROT_WEIGHT} "
        f"(min scale {ROT_WEIGHT_MIN_SCALE}), joint_velocity={JOINT_VEL_WEIGHT}, "
        f"joint_acceleration={JOINT_ACC_WEIGHT}. "
        "Raise rotation or lower position if rotation error is high; "
        "raise joint_velocity / joint_acceleration if joint_speed warnings are frequent; "
        "check radial_scale min if targets are heavily clipped inward.[/dim]"
    )
    console.print()
