"""Replay cached retarget joint trajectories in Meshcat (e.g. over SSH port forward)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from rlds import RldsObservationLoader

from .cache import RetargetOutputIndex, load_joint_trajectory, load_metadata
from .config import load_retarget_config
from .gpu import load_gpu_fk_model
from .viz import DualRobotMeshcatDisplay, display_retarget_frame


def _replay_viz_models(index: RetargetOutputIndex) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    """Kinematics model for ``q`` (GPU URDF when cached with ``use_gpu``) + mesh from description."""
    robot = load_robot_description(index.robot_description)
    if index.use_gpu:
        return load_gpu_fk_model(), robot.collision_model, robot.visual_model
    return robot.model, robot.collision_model, robot.visual_model


def _load_source_joint_positions(
    *,
    data_dir: Path,
    index: RetargetOutputIndex,
    demo_idx: int,
) -> np.ndarray:
    loader = RldsObservationLoader(data_dir=data_dir, dataset_url=index.dataset_url)
    demo = loader.get_demo(demo_idx)
    return np.asarray(demo["joint_position"], dtype=np.float64)


def _make_replay_viz(
    index: RetargetOutputIndex,
    *,
    compare_source: bool,
    compare_separation: float,
    panda_description: str,
) -> tuple[object, pin.Model]:
    """One Meshcat server for the whole session; returns (viz, kinematics model for ``q``)."""
    model, collision_model, visual_model = _replay_viz_models(index)
    if compare_source:
        source_robot = load_robot_description(panda_description)
        # Use the same kinematics model the playback ``q`` is built for (the
        # GPU FK model when the cache was solved with --gpu) — the description
        # model can have a different nq.
        viz = DualRobotMeshcatDisplay(
            model,
            collision_model,
            visual_model,
            source_robot.model,
            source_robot.collision_model,
            source_robot.visual_model,
            separation=compare_separation,
        )
        print(
            "Side-by-side Meshcat: source arm (left, tinted) follows the original demo; "
            "target arm (right) shows the retargeted trajectory."
        )
    else:
        from pinocchio.visualize import MeshcatVisualizer

        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
    print("Meshcat: open http://127.0.0.1:7000/static/ (SSH: ssh -L 7000:localhost:7000 …)")
    return viz, model


def _play_demo_frames(
    viz,
    model: pin.Model,
    joint_traj: np.ndarray,
    *,
    display_fps: float,
    loop: bool,
    source_joint_positions: np.ndarray | None,
) -> None:
    while True:
        for frame in range(joint_traj.shape[0]):
            q = pin.neutral(model)
            q[: model.nv] = joint_traj[frame]
            q = pin.normalize(model, q)
            display_retarget_frame(
                viz,
                q,
                source_joint_row=(
                    None if source_joint_positions is None else source_joint_positions[frame]
                ),
            )
            time.sleep(1.0 / display_fps)
        if not loop:
            return


def _source_joints_for_demo(
    index: RetargetOutputIndex,
    demo_idx: int,
    joint_traj: np.ndarray,
    *,
    compare_source: bool,
    data_dir: Path,
) -> np.ndarray | None:
    if not compare_source:
        return None
    source = _load_source_joint_positions(data_dir=data_dir, index=index, demo_idx=demo_idx)
    if len(source) != joint_traj.shape[0]:
        raise ValueError(
            f"demo {demo_idx}: source ({len(source)} frames) and "
            f"retarget ({joint_traj.shape[0]} frames) length mismatch"
        )
    return source


def replay_demo(
    output_dir: Path,
    demo_idx: int,
    *,
    fps: float | None = None,
    loop: bool = False,
    compare_source: bool = False,
    compare_separation: float = 1.0,
    data_dir: Path = Path("data"),
    panda_description: str = "panda_description",
) -> None:
    """Play one cached demo in Meshcat; blocks until interrupted when *loop* is False."""
    index = load_metadata(output_dir)
    joint_traj = load_joint_trajectory(output_dir, demo_idx)
    config = load_retarget_config()
    display_fps = float(config.display_fps if fps is None else fps)

    source_joint_positions = _source_joints_for_demo(
        index, demo_idx, joint_traj, compare_source=compare_source, data_dir=data_dir
    )
    viz, model = _make_replay_viz(
        index,
        compare_source=compare_source,
        compare_separation=compare_separation,
        panda_description=panda_description,
    )
    print(f"Replaying demo {demo_idx} ({joint_traj.shape[0]} frames) from {output_dir}")
    try:
        _play_demo_frames(
            viz,
            model,
            joint_traj,
            display_fps=display_fps,
            loop=loop,
            source_joint_positions=source_joint_positions,
        )
    except KeyboardInterrupt:
        print("Stopped.")


def replay_range(
    output_dir: Path,
    *,
    start_demo: int,
    end_demo: int | None,
    fps: float | None = None,
    compare_source: bool = False,
    compare_separation: float = 1.0,
    data_dir: Path = Path("data"),
    panda_description: str = "panda_description",
) -> None:
    """Play cached demos back to back, reusing one Meshcat server for the whole range."""
    index = load_metadata(output_dir)
    demo_ids = sorted({d.demo_idx for d in index.demos})
    if end_demo is None:
        targets = [i for i in demo_ids if i >= start_demo]
    else:
        targets = [i for i in demo_ids if start_demo <= i < end_demo]
    if not targets:
        raise ValueError(f"no cached demos in range [{start_demo}, {end_demo})")

    config = load_retarget_config()
    display_fps = float(config.display_fps if fps is None else fps)
    viz, model = _make_replay_viz(
        index,
        compare_source=compare_source,
        compare_separation=compare_separation,
        panda_description=panda_description,
    )
    try:
        for demo_idx in targets:
            joint_traj = load_joint_trajectory(output_dir, demo_idx)
            source_joint_positions = _source_joints_for_demo(
                index, demo_idx, joint_traj, compare_source=compare_source, data_dir=data_dir
            )
            print(f"Replaying demo {demo_idx} ({joint_traj.shape[0]} frames) from {output_dir}")
            _play_demo_frames(
                viz,
                model,
                joint_traj,
                display_fps=display_fps,
                loop=False,
                source_joint_positions=source_joint_positions,
            )
    except KeyboardInterrupt:
        print("Stopped.")
