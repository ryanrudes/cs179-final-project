"""Meshcat visualization helpers for retargeting."""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from .core import panda_q_from_demo


class DualRobotMeshcatDisplay:
    """Side-by-side Meshcat view: source arm (original demo) and target arm (retargeted)."""

    def __init__(
        self,
        target_model: pin.Model,
        target_collision: pin.GeometryModel,
        target_visual: pin.GeometryModel,
        source_model: pin.Model,
        source_collision: pin.GeometryModel,
        source_visual: pin.GeometryModel,
        *,
        target_root: str = "target",
        source_root: str = "source",
        separation: float = 1.0,
    ) -> None:
        from pinocchio.visualize import MeshcatVisualizer

        self.source_model = source_model
        self.target_viz = MeshcatVisualizer(target_model, target_collision, target_visual)
        self.target_viz.initViewer(open=True)
        self.target_viz.loadViewerModel(rootNodeName=target_root)

        self.source_viz = MeshcatVisualizer(source_model, source_collision, source_visual)
        self.source_viz.initViewer(viewer=self.target_viz.viewer)
        self.source_viz.loadViewerModel(
            rootNodeName=source_root,
            visual_color=[0.75, 0.78, 0.92, 1.0],
        )

        half = separation / 2.0
        target_offset = np.eye(4)
        target_offset[1, 3] = half
        source_offset = np.eye(4)
        source_offset[1, 3] = -half
        self.target_viz.viewer[target_root].set_transform(target_offset)
        self.source_viz.viewer[source_root].set_transform(source_offset)

        # Camera helpers need meshcat's set_cam_target/set_cam_pos, which not
        # all releases provide (e.g. meshcat 0.3.2). Skip on older versions —
        # the user can orbit manually.
        try:
            self.target_viz.setCameraTarget(np.array([0.4, 0.0, 0.2]))
            self.target_viz.setCameraPosition(np.array([2.2, 0.0, 1.1]))
        except AttributeError:
            pass

    def display(self, target_q: np.ndarray, source_joint_row: np.ndarray) -> None:
        self.target_viz.display(pin.normalize(self.target_viz.model, target_q))
        source_q = panda_q_from_demo(source_joint_row, self.source_model)
        self.source_viz.display(pin.normalize(self.source_model, source_q))


def create_retarget_visualizer(
    target_robot,
    *,
    compare_source: bool = False,
    source_robot=None,
    separation: float = 1.0,
):
    """Return a Meshcat visualizer, optionally with the source robot for side-by-side compare."""
    if compare_source:
        if source_robot is None:
            raise ValueError("source_robot is required when compare_source=True")
        return DualRobotMeshcatDisplay(
            target_robot.model,
            target_robot.collision_model,
            target_robot.visual_model,
            source_robot.model,
            source_robot.collision_model,
            source_robot.visual_model,
            separation=separation,
        )

    from pinocchio.visualize import MeshcatVisualizer

    viz = MeshcatVisualizer(target_robot.model, target_robot.collision_model, target_robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    return viz


def display_retarget_frame(
    viz,
    target_q: np.ndarray,
    *,
    source_joint_row: np.ndarray | None = None,
) -> None:
    """Update Meshcat for one frame (single robot or side-by-side compare)."""
    if isinstance(viz, DualRobotMeshcatDisplay):
        if source_joint_row is None:
            raise ValueError("source_joint_row is required for side-by-side visualization")
        viz.display(target_q, source_joint_row)
        return
    viz.display(target_q)
