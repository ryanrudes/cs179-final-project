"""Run UR3e retargeting over cached DROID proprio demos."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from reachability import (
    REACH_BINS_PHI,
    REACH_BINS_THETA,
    REACH_SAMPLE_COUNT_RETARGET,
    REACH_SAFETY_MARGIN,
    DirectionalReachEnvelope,
    scale_cartesian_to_robot,
)
from .config import RetargetConfig, load_retarget_config
from .core import (
    Retargeter,
    demo_elbow_side_targets,
    pose_log6_error,
    target_to_se3,
    tool_frame_id as resolve_tool_frame_id,
    unwrap_euler_targets,
)
from .demos import iter_retarget_demos
from .stats import (
    BatchStatsAccumulator,
    FrameArrays,
    LiveRetargetDisplay,
    compute_demo_stats,
    print_batch_summary,
)
from rlds import RldsObservationLoader, RoboticsRldsDatasetUrl

if TYPE_CHECKING:
    import pinocchio as pin


def _demo_count(loader: RldsObservationLoader, start_demo: int, end_demo: int | None) -> int:
    n_demos = len(loader)
    stop = n_demos if end_demo is None else min(end_demo, n_demos)
    return max(0, stop - start_demo)


def _retarget_demo_frames(
    *,
    joint_positions: np.ndarray,
    cartesian_positions: np.ndarray,
    retargeter: Retargeter,
    ur3e_reach: DirectionalReachEnvelope,
    panda_model: pin.Model,
    panda_data: pin.Data,
    control_hz: float,
    reach_safety: float,
    config: RetargetConfig,
    viz,
    on_frame: Callable[[int, float, float, float, float, bool, int], None] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    FrameArrays,
]:
    import time

    num_frames = len(joint_positions)
    cartesian_positions = cartesian_positions.copy()
    cartesian_positions[:, 3:6] = unwrap_euler_targets(cartesian_positions[:, 3:6])
    cartesian_positions, radial_scales = scale_cartesian_to_robot(
        cartesian_positions, ur3e_reach, safety=reach_safety
    )
    demo_elbow_sides = demo_elbow_side_targets(
        joint_positions, panda_model, panda_data, frame_names=config.frames.panda_elbow
    )

    retargeter.reset_episode(cartesian_positions[0])
    retargeter.set_position_scale(radial_scales[0])
    retargeter.set_elbow_side_target(demo_elbow_sides[0])

    positions = []
    position_errors = []
    rotation_errors = []
    rotation_error_vectors = []
    joint_speeds = []
    ik_success = []
    ik_iterations = []

    model = retargeter.model
    data = model.createData()
    reach_tool_frame_id = resolve_tool_frame_id(model, config.frames.tool)
    for frame in range(num_frames):
        target = cartesian_positions[frame]
        retargeter.set_position_scale(radial_scales[frame])
        retargeter.set_elbow_side_target(demo_elbow_sides[frame])
        q, pos, _rot, pos_error, rot_error, success, nit = retargeter(target)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        err6 = pose_log6_error(data.oMf[reach_tool_frame_id], target_to_se3(target))
        if retargeter.q_prev is not None:
            dq = pin.difference(model, retargeter.q_prev, retargeter.q)
            joint_speeds.append(float(np.linalg.norm(dq * control_hz)))
        else:
            joint_speeds.append(0.0)
        if viz is not None:
            viz.display(q)
            time.sleep(1 / config.display_fps)
        positions.append(pos)
        position_errors.append(pos_error)
        rotation_errors.append(rot_error)
        rotation_error_vectors.append(err6[3:].copy())
        ik_success.append(success)
        ik_iterations.append(nit)
        if on_frame is not None:
            on_frame(
                frame,
                pos_error,
                rot_error,
                float(radial_scales[frame]),
                joint_speeds[-1],
                success,
                nit,
            )

    frame_metrics = FrameArrays(
        position_errors=np.asarray(position_errors, dtype=np.float64),
        rotation_errors=np.asarray(rotation_errors, dtype=np.float64),
        radial_scales=np.asarray(radial_scales, dtype=np.float64),
        joint_speeds=np.asarray(joint_speeds, dtype=np.float64),
        ik_success=np.asarray(ik_success, dtype=bool),
        ik_iterations=np.asarray(ik_iterations, dtype=np.int64),
    )

    return (
        cartesian_positions,
        np.asarray(positions),
        np.asarray(rotation_error_vectors, dtype=np.float64),
        frame_metrics.position_errors,
        frame_metrics.rotation_errors,
        frame_metrics,
    )


def _plot_demo_errors(
    *,
    num_frames: int,
    cartesian_positions: np.ndarray,
    positions: np.ndarray,
    rotation_error_vectors: np.ndarray,
    position_errors: np.ndarray,
    rotation_errors: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(positions[:, 0], label="X", color="red")
    ax1.plot(positions[:, 1], label="Y", color="green")
    ax1.plot(positions[:, 2], label="Z", color="blue")
    ax1.plot(cartesian_positions[:, 0], label="Target X", color="red", linestyle="--")
    ax1.plot(cartesian_positions[:, 1], label="Target Y", color="green", linestyle="--")
    ax1.plot(cartesian_positions[:, 2], label="Target Z", color="blue", linestyle="--")
    # log6 rotational residual (same convention as retarget cost); target is zero.
    ax2.plot(rotation_error_vectors[:, 0], label="ex", color="red")
    ax2.plot(rotation_error_vectors[:, 1], label="ey", color="green")
    ax2.plot(rotation_error_vectors[:, 2], label="ez", color="blue")
    ax2.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax3.plot(position_errors, label="Position Error", color="orange")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(rotation_errors, label="Rotation Error", color="purple")
    ax1.set_xlabel("Frame")
    ax2.set_xlabel("Frame")
    ax3.set_xlabel("Frame")
    ax1.set_ylabel("Position (m)")
    ax2.set_ylabel("Rotation error (rad)")
    ax3.set_ylabel("Position Error (m)")
    ax3_twin.set_ylabel("Rotation Error (rad)")
    ax1.set_xlim(0, num_frames)
    ax2.set_xlim(0, num_frames)
    ax3.set_xlim(0, num_frames)
    ax1.set_ylim(0,)
    ax3.set_ylim(0,)
    ax3_twin.set_ylim(0,)
    ax1.legend()
    ax2.legend()
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2)
    plt.show()


def run(
    *,
    data_dir: Path = Path("data"),
    dataset_url: str | RoboticsRldsDatasetUrl = RoboticsRldsDatasetUrl.DROID_100,
    start_demo: int = 0,
    end_demo: int | None = None,
    robot_description: str = "ur3e_description",
    panda_description: str = "panda_description",
    control_hz: float | None = None,
    enable_visualization: bool = True,
    show_plots: bool = True,
    show_progress: bool | None = None,
    reach_n_samples: int = REACH_SAMPLE_COUNT_RETARGET,
    reach_n_theta: int = REACH_BINS_THETA,
    reach_n_phi: int = REACH_BINS_PHI,
    reach_force_rebuild: bool = False,
    reach_safety: float = REACH_SAFETY_MARGIN,
    config_path: Path | None = None,
) -> BatchStatsAccumulator | None:
    if not 0.0 < reach_safety <= 1.0:
        raise ValueError(f"reach_safety must be in (0, 1], got {reach_safety}")

    retarget_config = load_retarget_config(config_path)

    robot = load_robot_description(robot_description)
    panda_robot = load_robot_description(panda_description)
    panda_data = panda_robot.model.createData()

    reach_tool_frame_id = resolve_tool_frame_id(robot.model, retarget_config.frames.tool)
    reach_data = robot.model.createData()
    ur3e_reach = DirectionalReachEnvelope.from_robot_cached(
        robot.model,
        reach_data,
        reach_tool_frame_id,
        robot_key=robot_description,
        n_samples=reach_n_samples,
        n_theta=reach_n_theta,
        n_phi=reach_n_phi,
        force_rebuild=reach_force_rebuild,
    )

    loader = RldsObservationLoader(data_dir=data_dir, dataset_url=dataset_url)
    demo_control_hz = control_hz if control_hz is not None else loader.control_hz
    print(f"Using control_hz={demo_control_hz} for velocity/acceleration costs")
    print(f"Using reach_safety={reach_safety} for workspace scaling")
    if config_path is not None:
        print(f"Retarget config: {Path(config_path).resolve()}")
    else:
        from .config import default_config_path

        print(f"Retarget config: {default_config_path()}")

    retargeter = Retargeter(robot, control_hz=demo_control_hz, config=retarget_config)

    viz = None
    if enable_visualization:
        from pinocchio.visualize import MeshcatVisualizer

        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    if show_progress is None:
        show_progress = not enable_visualization and not show_plots

    report_stats = show_progress
    stats_batch = BatchStatsAccumulator() if report_stats else None

    demo_count = _demo_count(loader, start_demo, end_demo)
    if demo_count == 0:
        print("No demos in the requested range.")
        return stats_batch

    def retarget_one_demo(
        demo_idx: int,
        joint_positions: np.ndarray,
        cartesian_positions: np.ndarray,
        on_frame: Callable[[int, float, float, float, float, bool, int], None] | None,
        live_display: LiveRetargetDisplay | None = None,
    ) -> None:
        num_frames = len(joint_positions)
        (
            cartesian_positions,
            positions,
            rotation_error_vectors,
            position_errors,
            rotation_errors,
            frame_metrics,
        ) = _retarget_demo_frames(
                joint_positions=joint_positions,
                cartesian_positions=cartesian_positions,
                retargeter=retargeter,
                ur3e_reach=ur3e_reach,
                panda_model=panda_robot.model,
                panda_data=panda_data,
                control_hz=demo_control_hz,
                reach_safety=reach_safety,
                config=retarget_config,
                viz=viz,
                on_frame=on_frame,
            )
        if stats_batch is not None:
            demo_stats = compute_demo_stats(
                demo_idx=demo_idx,
                position_errors=frame_metrics.position_errors,
                rotation_errors=frame_metrics.rotation_errors,
                radial_scales=frame_metrics.radial_scales,
                joint_speeds=frame_metrics.joint_speeds,
                ik_success=frame_metrics.ik_success,
                ik_iterations=frame_metrics.ik_iterations,
            )
            if live_display is not None:
                live_display.finish_demo(demo_stats, frame_metrics)
            else:
                stats_batch.add_demo(demo_stats, frame_metrics)
        if show_plots:
            _plot_demo_errors(
                num_frames=num_frames,
                cartesian_positions=cartesian_positions,
                positions=positions,
                rotation_error_vectors=rotation_error_vectors,
                position_errors=position_errors,
                rotation_errors=rotation_errors,
            )
        elif not show_progress:
            print(f"Demo {demo_idx}: done ({num_frames} frames).")

    if not show_progress:
        for demo_idx, demo in enumerate(
            iter_retarget_demos(loader=loader, start_demo=start_demo, end_demo=end_demo),
            start=start_demo,
        ):
            joint_positions, _gripper_positions, cartesian_positions = demo
            retarget_one_demo(demo_idx, joint_positions, cartesian_positions, on_frame=None)
        if stats_batch is not None:
            print_batch_summary(
                stats_batch,
                control_hz=demo_control_hz,
                start_demo=start_demo,
                end_demo=end_demo,
            )
        else:
            print(f"Retargeted {demo_count} demo(s).")
        return stats_batch

    assert stats_batch is not None
    with LiveRetargetDisplay(
        batch=stats_batch,
        demo_count=demo_count,
        control_hz=demo_control_hz,
    ) as display:
        demos_task = display.add_demos_task()
        for demo_idx, demo in enumerate(
            iter_retarget_demos(loader=loader, start_demo=start_demo, end_demo=end_demo),
            start=start_demo,
        ):
            joint_positions, _gripper_positions, cartesian_positions = demo
            num_frames = len(joint_positions)
            frames_task = display.add_frames_task(demo_idx, num_frames)

            def on_frame(
                _frame: int,
                pos_error: float,
                rot_error: float,
                radial_scale: float,
                joint_speed: float,
                ik_success: bool,
                ik_iterations: int,
                *,
                task_id: int = frames_task,
            ) -> None:
                display.advance_frame(
                    task_id,
                    position_error=pos_error,
                    rotation_error=rot_error,
                    radial_scale=radial_scale,
                    joint_speed=joint_speed,
                    ik_success=ik_success,
                    ik_iterations=ik_iterations,
                )

            retarget_one_demo(
                demo_idx,
                joint_positions,
                cartesian_positions,
                on_frame=on_frame,
                live_display=display,
            )
            display.remove_frames_task(frames_task)
            display.progress.advance(demos_task)
        display.refresh()

    if stats_batch is not None:
        print_batch_summary(
            stats_batch,
            control_hz=demo_control_hz,
            start_demo=start_demo,
            end_demo=end_demo,
        )
    else:
        print(f"Retargeted {demo_count} demo(s).")
    return stats_batch
