"""Replay cached retarget joint trajectories in Meshcat (e.g. over SSH port forward)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from .cache import RetargetOutputIndex, load_joint_trajectory, load_metadata
from .config import load_retarget_config
from .gpu import load_gpu_fk_model


def _replay_viz_models(index: RetargetOutputIndex) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    """Kinematics model for ``q`` (GPU URDF when cached with ``use_gpu``) + mesh from description."""
    robot = load_robot_description(index.robot_description)
    if index.use_gpu:
        return load_gpu_fk_model(), robot.collision_model, robot.visual_model
    return robot.model, robot.collision_model, robot.visual_model


def replay_demo(
    output_dir: Path,
    demo_idx: int,
    *,
    fps: float | None = None,
    loop: bool = False,
) -> None:
    """Play one cached demo in Meshcat; blocks until interrupted when *loop* is False."""
    index = load_metadata(output_dir)
    joint_traj = load_joint_trajectory(output_dir, demo_idx)
    config = load_retarget_config()
    display_fps = float(config.display_fps if fps is None else fps)

    model, collision_model, visual_model = _replay_viz_models(index)

    from pinocchio.visualize import MeshcatVisualizer

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    print(f"Meshcat: open http://127.0.0.1:7000/static/ (SSH: ssh -L 7000:localhost:7000 …)")
    print(f"Replaying demo {demo_idx} ({joint_traj.shape[0]} frames) from {output_dir}")

    try:
        while True:
            for frame in range(joint_traj.shape[0]):
                q = pin.neutral(model)
                q[: model.nv] = joint_traj[frame]
                viz.display(pin.normalize(model, q))
                time.sleep(1.0 / display_fps)
            if not loop:
                break
    except KeyboardInterrupt:
        print("Stopped.")


def replay_range(
    output_dir: Path,
    *,
    start_demo: int,
    end_demo: int | None,
    fps: float | None = None,
) -> None:
    index = load_metadata(output_dir)
    demo_ids = sorted({d.demo_idx for d in index.demos})
    if end_demo is None:
        targets = [i for i in demo_ids if i >= start_demo]
    else:
        targets = [i for i in demo_ids if start_demo <= i < end_demo]
    if not targets:
        raise ValueError(f"no cached demos in range [{start_demo}, {end_demo})")
    for demo_idx in targets:
        replay_demo(output_dir, demo_idx, fps=fps, loop=False)
