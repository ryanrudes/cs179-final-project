"""Native retargeter parity vs Python SciPy path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.spatial.transform import Rotation as R

from retarget.config import load_retarget_config
from retarget.core import (
    Retargeter,
    _NATIVE_RETARGETER,
    native_retarget_params,
    pose_log6_error,
    resolve_tool_frame,
    seed_ik,
    set_use_native_retarget,
    target_to_se3,
)

pytestmark = pytest.mark.skipif(_NATIVE_RETARGETER is None, reason="native retargeter not built")

# DROID demo 0 frame 0 after 1024×1024 reach scaling (ill-conditioned damped LS IK).
# Must match float64(cast(scaled cart[0])) exactly; ~1e-8 target drift changes the IK basin.
_HARD_IK_TARGET = np.array(
    [0.35868366, 0.06870935, 0.51446164, -2.89348388, -0.19573815, 0.12700166],
    dtype=np.float64,
)


def test_native_seed_ik_matches_python() -> None:
    from cs179 import _native as native_mod

    robot = load_robot_description("ur3e_description")
    cfg = load_retarget_config()
    model = robot.model
    data = model.createData()
    frame_id = model.getFrameId("tool0")
    q0 = pin.neutral(model)

    q_py = seed_ik(model, data, q0, _HARD_IK_TARGET, frame_id, config=cfg)
    params = dict(cfg.to_native_dict())
    q_cpp = np.asarray(
        native_mod.seed_ik(model, q0, _HARD_IK_TARGET, cfg.frames.tool, params),
        dtype=np.float64,
    )

    # Early iterations should be bit-identical; full 200-iter paths can differ by 2π joint wraps.
    cfg_10 = replace(cfg, ik_seed=replace(cfg.ik_seed, max_iterations=10))
    params_short = {**params, "seed_ik_n_iter": 10}
    q_py_10 = seed_ik(model, data, q0, _HARD_IK_TARGET, frame_id, config=cfg_10)
    q_cpp_10 = np.asarray(
        native_mod.seed_ik(model, q0, _HARD_IK_TARGET, cfg.frames.tool, params_short),
        dtype=np.float64,
    )
    assert np.allclose(q_py_10, q_cpp_10, atol=1e-6, rtol=0.0)

    oMdes = target_to_se3(_HARD_IK_TARGET)

    def tool_log6_error(q: np.ndarray) -> float:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return float(np.linalg.norm(pose_log6_error(data.oMf[frame_id], oMdes)))

    assert tool_log6_error(q_py) < 0.05
    assert abs(tool_log6_error(q_py) - tool_log6_error(q_cpp)) < 0.01


def test_native_retarget_matches_python_on_single_frame() -> None:
    robot = load_robot_description("ur3e_description")
    target = np.array([0.35, -0.15, 0.45, 0.1, -0.2, 0.3], dtype=float)

    set_use_native_retarget(False)
    py_rt = Retargeter(robot, control_hz=15.0)
    py_rt.reset_episode(target)
    py_rt.set_position_scale(0.85)
    py_rt.set_elbow_side_target(1.0)
    py_q, py_pos, py_rot, py_pe, py_re, _, _ = py_rt(target)

    set_use_native_retarget(True)
    native_rt = Retargeter(robot, control_hz=15.0)
    native_rt.reset_episode(target)
    native_rt.set_position_scale(0.85)
    native_rt.set_elbow_side_target(1.0)
    n_q, n_pos, n_rot, n_pe, n_re, _, _ = native_rt(target)

    assert np.allclose(n_q, py_q, atol=0.05, rtol=0.05)
    assert np.allclose(n_pos, py_pos, atol=0.01)
    assert np.allclose(R.from_euler("xyz", n_rot).as_matrix(), R.from_euler("xyz", py_rot).as_matrix(), atol=1e-2)
    assert abs(n_pe - py_pe) < 0.02
    assert abs(n_re - py_re) < 0.1


def test_native_seed_ik_panda_nv() -> None:
    """Panda has nv=9; native seed IK must not truncate the velocity to 6."""
    from cs179 import _native as native_mod

    robot = load_robot_description("panda_description")
    model = robot.model
    cfg = load_retarget_config()
    q0 = pin.neutral(model)
    target = np.array([0.4, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
    tool = resolve_tool_frame(model, cfg.frames.tool)
    q = native_mod.seed_ik(model, q0, target, tool, native_retarget_params(cfg, model))
    assert q.shape == (model.nq,)


def test_native_retarget_warm_start_sequence() -> None:
    robot = load_robot_description("ur3e_description")
    targets = np.array(
        [
            [0.35, -0.15, 0.45, 0.1, -0.2, 0.3],
            [0.36, -0.14, 0.46, 0.11, -0.19, 0.31],
            [0.37, -0.13, 0.47, 0.12, -0.18, 0.32],
        ],
        dtype=float,
    )

    set_use_native_retarget(True)
    rt = Retargeter(robot, control_hz=15.0)
    rt.reset_episode(targets[0])
    for target in targets:
        q, *_ = rt(target)
        assert q.shape == (robot.model.nq,)
        assert np.all(np.isfinite(q))
