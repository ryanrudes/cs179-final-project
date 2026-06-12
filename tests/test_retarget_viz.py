"""Tests for retarget Meshcat helpers."""

from __future__ import annotations

import numpy as np
import pytest

from retarget.core import panda_q_from_demo
from retarget.viz import display_retarget_frame


class _SingleViz:
    def __init__(self) -> None:
        self.last_q = None

    def display(self, q: np.ndarray) -> None:
        self.last_q = q


def test_display_retarget_frame_single_robot() -> None:
    viz = _SingleViz()
    q = np.array([0.1, 0.2, 0.3])
    display_retarget_frame(viz, q, source_joint_row=np.zeros(7))
    assert viz.last_q is q


def test_panda_q_from_demo_fills_neutral() -> None:
    from robot_descriptions.loaders.pinocchio import load_robot_description

    model = load_robot_description("panda_description").model
    q = panda_q_from_demo(np.arange(7, dtype=np.float64), model)
    assert q.shape == (model.nq,)
    assert np.allclose(q[:7], np.arange(7))


def test_create_retarget_visualizer_requires_source_robot() -> None:
    from robot_descriptions.loaders.pinocchio import load_robot_description

    from retarget.viz import create_retarget_visualizer

    robot = load_robot_description("ur3e_description")
    with pytest.raises(ValueError, match="source_robot"):
        create_retarget_visualizer(robot, compare_source=True)
