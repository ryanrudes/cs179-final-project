"""GPU trajectory retarget smoke tests (requires CUDA build)."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest
from retarget.config import load_retarget_config
from retarget.core import Retargeter, pose_log6_error, target_to_se3, tool_frame_id
from retarget.gpu import (
    estimate_gpu_launch_bytes,
    gpu_retarget_built,
    iter_gpu_demo_batches,
    load_gpu_fk_model,
    max_gpu_batch_demoes,
    max_gpu_trajectory_frames,
    trajectory_fits_gpu_shmem,
    pad_time,
    pack_targets,
    pack_trajectories,
    retarget_cartesian_trajectory,
    retarget_params_from_config,
    retarget_trajectories_gpu,
    pack_initial_gpu_q,
)

pytestmark = pytest.mark.skipif(not gpu_retarget_built(), reason="CUDA GPU retarget not built")


@pytest.fixture(scope="module")
def gpu_fk():
    return load_gpu_fk_model()


def test_retarget_gpu_short_trajectory(gpu_fk):
    model = gpu_fk
    data = model.createData()
    config = load_retarget_config()

    # Targets near the neutral tool pose with a small spatial drift.
    t_len = 48
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    targets = np.zeros((t_len, 6), dtype=np.float32)
    targets[:, 0] = np.linspace(float(oM0.translation[0]) - 0.02, float(oM0.translation[0]) + 0.02, t_len)
    targets[:, 1] = float(oM0.translation[1])
    targets[:, 2] = float(oM0.translation[2])
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    targets[:, 3:6] = e0

    q_in, lengths, d_pad, t_pad = pack_initial_gpu_q(model, [t_len])
    tgt_batch = pack_targets([targets], t_pad=t_pad)
    n_dof = model.nv

    kwargs = retarget_params_from_config(model, config, t_pad=t_pad, n_dof=n_dof)
    q_out = retarget_trajectories_gpu(
        q_in,
        tgt_batch,
        lengths,
        **kwargs,
    )

    q_traj = q_out[0, :n_dof, :t_len]
    errs = []
    for t in range(t_len):
        q_pin = pin.neutral(model)
        q_pin[:n_dof] = q_traj[:, t]
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        err6 = pose_log6_error(data.oMf[frame_id], target_to_se3(targets[t]))
        errs.append(float(np.linalg.norm(err6[:3])))
    mean_pos_err = float(np.mean(errs))
    assert mean_pos_err < 0.02, f"mean position error {mean_pos_err:.4f} m too large"


def test_retarget_gpu_batch_mixed_lengths(gpu_fk):
    model = gpu_fk
    config = load_retarget_config()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    base = np.array(
        [oM0.translation[0], oM0.translation[1], oM0.translation[2], e0[0], e0[1], e0[2]],
        dtype=np.float32,
    )
    tgt_short = np.tile(base, (24, 1))
    tgt_long = np.tile(base, (48, 1))
    tgt_long[:, 0] += np.linspace(-0.02, 0.02, 48, dtype=np.float32)

    from retarget.gpu import retarget_cartesian_trajectories

    q_list = retarget_cartesian_trajectories(model, [tgt_short, tgt_long], config)
    assert len(q_list) == 2
    assert q_list[0].shape == (24, model.nv)
    assert q_list[1].shape == (48, model.nv)


def test_retarget_gpu_batch_two_trajectories(gpu_fk):
    model = gpu_fk
    config = load_retarget_config()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    base = np.array(
        [oM0.translation[0], oM0.translation[1], oM0.translation[2], e0[0], e0[1], e0[2]],
        dtype=np.float32,
    )
    t_len = 32
    tgt_a = np.tile(base, (t_len, 1))
    tgt_b = tgt_a.copy()
    tgt_b[:, 0] += np.linspace(-0.01, 0.01, t_len, dtype=np.float32)

    from retarget.gpu import retarget_cartesian_trajectories

    q_list = retarget_cartesian_trajectories(model, [tgt_a, tgt_b], config)
    assert len(q_list) == 2
    assert q_list[0].shape == (t_len, model.nv)
    assert not np.allclose(q_list[0], q_list[1])


def test_refine_elbow_trajectory_noop_when_weight_zero(gpu_fk):
    from retarget.gpu import refine_elbow_trajectory

    model = gpu_fk
    config = load_retarget_config()
    q = np.zeros((8, model.nv), dtype=np.float32)
    sides = np.ones(8, dtype=np.float32)
    out = refine_elbow_trajectory(model, q, sides, config)
    assert np.allclose(out, q)


def test_use_rotation_dls_matches_pose(gpu_fk):
    """6D DLS uses fastfk ``spatial_local`` Jacobian (matches Pinocchio LOCAL)."""
    model = gpu_fk
    data = model.createData()
    config = load_retarget_config()
    t_len = 32
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    targets = np.zeros((t_len, 6), dtype=np.float32)
    targets[:, 0] = np.linspace(float(oM0.translation[0]) - 0.02, float(oM0.translation[0]) + 0.02, t_len)
    targets[:, 1] = float(oM0.translation[1])
    targets[:, 2] = float(oM0.translation[2])
    targets[:, 3:6] = e0

    q_out = retarget_cartesian_trajectory(model, targets, config, rot_nu_clamp=0.5)
    assert q_out.shape == (t_len, model.nv)
    pos_errs = []
    rot_errs = []
    for t in range(t_len):
        q_pin = pin.neutral(model)
        q_pin[: model.nv] = q_out[t]
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        err6 = pose_log6_error(data.oMf[frame_id], target_to_se3(targets[t]))
        pos_errs.append(float(np.linalg.norm(err6[:3])))
        rot_errs.append(float(np.linalg.norm(err6[3:])))
    assert float(np.mean(pos_errs)) < 0.02
    assert float(np.mean(rot_errs)) < 0.15


def test_retarget_cartesian_trajectory_api(gpu_fk):
    model = gpu_fk
    config = load_retarget_config()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    targets = np.tile(
        np.array(
            [oM0.translation[0], oM0.translation[1], oM0.translation[2], e0[0], e0[1], e0[2]],
            dtype=np.float32,
        ),
        (16, 1),
    )
    q_traj = retarget_cartesian_trajectory(model, targets, config)
    assert q_traj.shape == (16, model.nv)
    max_frames = max_gpu_trajectory_frames(model.nv)
    assert max_frames >= 48


def test_gpu_retarget_two_frame_trajectory(gpu_fk):
    """Regression: t_len=2 must not OOB-read t-2 in temporal acceleration stencil."""
    model = gpu_fk
    config = load_retarget_config()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    base = np.array(
        [oM0.translation[0], oM0.translation[1], oM0.translation[2], e0[0], e0[1], e0[2]],
        dtype=np.float32,
    )
    targets = np.tile(base, (2, 1))
    q_traj = retarget_cartesian_trajectory(model, targets, config)
    assert q_traj.shape == (2, model.nv)


def test_gpu_trajectory_over_shmem_limit_rejected(gpu_fk):
    """Trajectories longer than the SMEM budget are rejected (not windowed)."""
    model = gpu_fk
    config = load_retarget_config()
    max_f = max_gpu_trajectory_frames(model.nv)
    if max_f < 80:
        pytest.skip("GPU shared-memory budget too small for limit test")
    t_len = max_f + 24
    assert not trajectory_fits_gpu_shmem(t_len, model.nv)
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    targets = np.tile(
        np.array(
            [oM0.translation[0], oM0.translation[1], oM0.translation[2], e0[0], e0[1], e0[2]],
            dtype=np.float32,
        ),
        (t_len, 1),
    )
    with pytest.raises(ValueError, match="exceeds GPU"):
        retarget_cartesian_trajectory(model, targets, config)


def test_spatial_jacobian_matches_pinocchio(gpu_fk):
    from retarget.gpu_jacobian import max_spatial_jacobian_error

    rng = np.random.default_rng(0)
    q_samples = rng.normal(0, 0.4, (16, 6))
    assert max_spatial_jacobian_error(gpu_fk, q_samples) < 1.0e-5


def test_iter_gpu_demo_batches_respects_memory_budget(gpu_fk):
    """Greedy batching splits when padded T_pad grows the launch allocation."""
    model = gpu_fk
    n_dof = model.nv
    budget = estimate_gpu_launch_bytes(2, pad_time(64), n_dof)

    def stream():
        for i in range(6):
            cart = np.zeros((32 + 8 * i, 6), dtype=np.float32)
            yield i, np.zeros((len(cart), 7), dtype=np.float32), cart

    batches = list(
        iter_gpu_demo_batches(
            stream(),
            n_dof=n_dof,
            mem_budget_bytes=budget,
            max_trajectories=2,
        )
    )
    assert len(batches) >= 2
    assert sum(len(b) for b in batches) == 6
    for batch in batches:
        assert len(batch) <= 2
        max_len = max(len(c) for _i, _j, c in batch)
        t_pad = pad_time(max_len)
        assert estimate_gpu_launch_bytes(len(batch), t_pad, n_dof) <= estimate_gpu_launch_bytes(
            2, t_pad, n_dof
        )


def test_max_gpu_batch_demoes_positive(gpu_fk):
    n = max_gpu_batch_demoes(pad_time(128), gpu_fk.nv, mem_budget_bytes=512 * 1024**2)
    assert n >= 1


def test_max_gpu_batch_demoes_capped_by_launch_limit(gpu_fk):
    n = max_gpu_batch_demoes(pad_time(32), gpu_fk.nv, mem_budget_bytes=1024**4)
    from retarget.gpu import _MAX_GPU_TRAJECTORIES_PER_LAUNCH

    assert n == _MAX_GPU_TRAJECTORIES_PER_LAUNCH


def test_iter_gpu_demo_batches_splits_at_trajectory_cap(gpu_fk):
    from retarget.gpu import _MAX_GPU_TRAJECTORIES_PER_LAUNCH

    model = gpu_fk
    n_dof = model.nv
    big_budget = 1024**4

    def stream():
        for i in range(_MAX_GPU_TRAJECTORIES_PER_LAUNCH + 4):
            cart = np.zeros((32, 6), dtype=np.float32)
            yield i, np.zeros((32, 7), dtype=np.float32), cart

    batches = list(
        iter_gpu_demo_batches(stream(), n_dof=n_dof, mem_budget_bytes=big_budget)
    )
    assert len(batches) >= 2
    assert all(len(b) <= _MAX_GPU_TRAJECTORIES_PER_LAUNCH for b in batches)
    assert sum(len(b) for b in batches) == _MAX_GPU_TRAJECTORIES_PER_LAUNCH + 4


def test_gpu_vs_cpu_synthetic_trajectory(gpu_fk):
    """GPU path vs CPU retargeter on the same URDF as the fastfk kernels."""

    model = gpu_fk

    class _MinimalRobot:
        def __init__(self, pin_model: pin.Model) -> None:
            self.model = pin_model
            self.data = pin_model.createData()

    cpu_robot = _MinimalRobot(model)
    data = model.createData()
    config = load_retarget_config()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    frame_id = tool_frame_id(model, config.frames.tool)
    oM0 = data.oMf[frame_id]
    from scipy.spatial.transform import Rotation as R

    e0 = R.from_matrix(oM0.rotation).as_euler("xyz")
    t_len = 32
    targets = np.zeros((t_len, 6), dtype=np.float32)
    targets[:, 0] = np.linspace(float(oM0.translation[0]) - 0.02, float(oM0.translation[0]) + 0.02, t_len)
    targets[:, 1] = float(oM0.translation[1])
    targets[:, 2] = float(oM0.translation[2])
    targets[:, 3:6] = e0

    q_gpu = retarget_cartesian_trajectory(model, targets, config)

    retargeter = Retargeter(cpu_robot, control_hz=config.control_hz, config=config)
    retargeter.reset_episode(targets[0])
    q_cpu = []
    for t in range(t_len):
        q, *_ = retargeter(targets[t])
        q_cpu.append(q[: model.nv])
    q_cpu = np.stack(q_cpu)

    joint_gaps = [float(np.linalg.norm(q_gpu[t] - q_cpu[t])) for t in range(t_len)]
    mean_joint_gap = float(np.mean(joint_gaps))
    assert mean_joint_gap < 0.5, f"mean joint L2 gap {mean_joint_gap:.3f} rad vs CPU"
