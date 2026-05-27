"""Slow regression: native retarget pose quality vs Python (SciPy) on cached demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from robot_descriptions.loaders.pinocchio import load_robot_description

from reachability import DirectionalReachEnvelope, scale_cartesian_to_robot
from retarget.core import (
    Retargeter,
    _NATIVE_RETARGETER,
    demo_elbow_side_targets,
    set_use_native_retarget,
    unwrap_euler_targets,
)
from rlds import RldsObservationLoader, RoboticsRldsDatasetUrl

# Native must not exceed Python position error by more than this (metres).
PER_FRAME_MARGIN_M = 0.005
# Mean native position error allowed above Python mean.
MEAN_MARGIN_M = 0.004
# Share of frames where native is worse than Python + PER_FRAME_MARGIN_M.
MAX_WORSE_FRAME_FRAC = 0.10

NUM_DEMOS = 10
MAX_FRAMES_PER_DEMO = 80
REACH_N_SAMPLES = 100_000

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(_NATIVE_RETARGETER is None, reason="native retargeter not built"),
]

_DATASET_DIR = Path("data") / "droid_100"
_HAS_DROID_CACHE = (_DATASET_DIR / "metadata" / "metadata.json").is_file()


def _run_trajectory_pose_errors(
    robot,
    cartesian: np.ndarray,
    radial_scales: np.ndarray,
    elbow_sides: np.ndarray,
    *,
    use_native: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Warm-started trajectory; returns (position_errors, rotation_errors) per frame."""
    set_use_native_retarget(use_native)
    rt = Retargeter(robot, control_hz=15.0)
    rt.reset_episode(cartesian[0])
    pos_errs: list[float] = []
    rot_errs: list[float] = []
    for i in range(len(cartesian)):
        rt.set_position_scale(float(radial_scales[i]))
        rt.set_elbow_side_target(float(elbow_sides[i]))
        _q, _pos, _rot, pe, re, _ok, _nit = rt(cartesian[i])
        pos_errs.append(float(pe))
        rot_errs.append(float(re))
    return np.asarray(pos_errs), np.asarray(rot_errs)


def _prepare_demo(
    loader: RldsObservationLoader,
    demo_id: int,
    reach: DirectionalReachEnvelope,
    panda_model,
    panda_data,
    *,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demo = loader.get_demo(demo_id)
    cart = np.asarray(demo["cartesian_position"], dtype=float)
    joint = np.asarray(demo["joint_position"], dtype=float)
    n = min(max_frames, len(cart))
    cart = cart[:n]
    joint = joint[:n]
    cart, scales = scale_cartesian_to_robot(cart.copy(), reach, safety=0.9)
    cart[:, 3:6] = unwrap_euler_targets(cart[:, 3:6])
    elbows = demo_elbow_side_targets(joint, panda_model, panda_data)
    return cart, scales, elbows


@pytest.fixture(scope="module")
def retarget_quality_context():
    if not _HAS_DROID_CACHE:
        pytest.skip(f"DROID cache missing under {_DATASET_DIR}")

    robot = load_robot_description("ur3e_description")
    panda = load_robot_description("panda_description")
    ur3e = load_robot_description("ur3e_description")
    reach = DirectionalReachEnvelope.from_robot_cached(
        ur3e.model,
        ur3e.data,
        ur3e.model.getFrameId("tool0"),
        robot_key="ur3e_description",
        n_samples=REACH_N_SAMPLES,
        show_progress=False,
    )
    loader = RldsObservationLoader(Path("data"), str(RoboticsRldsDatasetUrl.DROID_100))
    return {
        "robot": robot,
        "panda_model": panda.model,
        "panda_data": panda.data,
        "reach": reach,
        "loader": loader,
    }


@pytest.mark.parametrize("demo_id", list(range(NUM_DEMOS)))
def test_native_pose_error_not_worse_than_python_per_demo(
    retarget_quality_context,
    demo_id: int,
) -> None:
    """Native NLopt path must not regress task-space position error vs SciPy on real demos."""
    ctx = retarget_quality_context
    cart, scales, elbows = _prepare_demo(
        ctx["loader"],
        demo_id,
        ctx["reach"],
        ctx["panda_model"],
        ctx["panda_data"],
        max_frames=MAX_FRAMES_PER_DEMO,
    )
    pe_py, _re_py = _run_trajectory_pose_errors(
        ctx["robot"], cart, scales, elbows, use_native=False
    )
    pe_nat, _re_nat = _run_trajectory_pose_errors(
        ctx["robot"], cart, scales, elbows, use_native=True
    )

    worse = pe_nat > pe_py + PER_FRAME_MARGIN_M
    worse_frac = float(np.mean(worse))

    assert worse_frac <= MAX_WORSE_FRAME_FRAC, (
        f"demo {demo_id}: native worse on {worse.sum()}/{len(pe_py)} frames "
        f"(frac={worse_frac:.3f}, max allowed={MAX_WORSE_FRAME_FRAC}); "
        f"worst excess={(pe_nat - pe_py).max():.4f} m"
    )
    assert float(pe_nat.mean()) <= float(pe_py.mean()) + MEAN_MARGIN_M, (
        f"demo {demo_id}: mean pos err native={pe_nat.mean()*1000:.2f} mm "
        f"python={pe_py.mean()*1000:.2f} mm (margin={MEAN_MARGIN_M*1000:.1f} mm)"
    )


def test_native_pose_error_aggregate_over_demos(retarget_quality_context) -> None:
    """Across demos 0..N-1, native must stay competitive on average position error."""
    ctx = retarget_quality_context
    all_py: list[float] = []
    all_nat: list[float] = []
    total_worse = 0
    total_frames = 0

    for demo_id in range(NUM_DEMOS):
        cart, scales, elbows = _prepare_demo(
            ctx["loader"],
            demo_id,
            ctx["reach"],
            ctx["panda_model"],
            ctx["panda_data"],
            max_frames=MAX_FRAMES_PER_DEMO,
        )
        pe_py, _ = _run_trajectory_pose_errors(
            ctx["robot"], cart, scales, elbows, use_native=False
        )
        pe_nat, _ = _run_trajectory_pose_errors(
            ctx["robot"], cart, scales, elbows, use_native=True
        )
        all_py.extend(pe_py.tolist())
        all_nat.extend(pe_nat.tolist())
        total_worse += int(np.sum(pe_nat > pe_py + PER_FRAME_MARGIN_M))
        total_frames += len(pe_py)

    worse_frac = total_worse / total_frames
    assert worse_frac <= MAX_WORSE_FRAME_FRAC, (
        f"aggregate: native worse on {total_worse}/{total_frames} frames (frac={worse_frac:.3f})"
    )
    assert np.mean(all_nat) <= np.mean(all_py) + MEAN_MARGIN_M, (
        f"aggregate mean pos err native={np.mean(all_nat)*1000:.2f} mm "
        f"python={np.mean(all_py)*1000:.2f} mm"
    )
