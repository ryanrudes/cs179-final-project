"""GPU trajectory retargeting (Jacobi projected gradient descent in shared memory)."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin

from .config import RetargetConfig, load_retarget_config
from .core import (
    clamp_configuration,
    elbow_side_scalar,
    seed_ik,
    target_elbow_frames,
    tool_frame_id,
    unwrap_euler_targets,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GPU_NATIVE = None
_GPU_SHMEM_BYTES = None
try:
    from cs179._native import (
        retarget_gpu_shmem_bytes as _retarget_gpu_shmem_bytes,
        retarget_trajectories_gpu as _retarget_trajectories_gpu,
    )

    _GPU_NATIVE = _retarget_trajectories_gpu
    _GPU_SHMEM_BYTES = _retarget_gpu_shmem_bytes
except ImportError:
    pass

# Conservative default when CUDA device props are unavailable from Python.
_DEFAULT_GPU_SHMEM_LIMIT = 48 * 1024

_ROOT = Path(__file__).resolve().parents[2]
# fastfk kernels under ``kernels/`` were generated from this URDF.
GPU_FK_URDF = _ROOT / "urdf" / "ur3e.urdf"


@lru_cache(maxsize=1)
def load_gpu_fk_model() -> pin.Model:
    """Pinocchio model matching the bundled fastfk tool0 codegen."""
    if not GPU_FK_URDF.is_file():
        raise FileNotFoundError(f"GPU FK URDF not found: {GPU_FK_URDF}")
    return pin.buildModelFromUrdf(str(GPU_FK_URDF))


def gpu_retarget_built() -> bool:
    return _GPU_NATIVE is not None


def max_gpu_trajectory_frames(
    n_dof: int = 6,
    *,
    shmem_limit: int | None = None,
) -> int:
    """Max ``T_pad`` (32-aligned) that fits in one block's shared memory."""
    if _GPU_SHMEM_BYTES is None:
        return 0
    from cs179._native import RetargetGpuParams

    limit = shmem_limit if shmem_limit is not None else _DEFAULT_GPU_SHMEM_LIMIT
    params = RetargetGpuParams()
    params.n_dof = n_dof
    params.d_pad = pad_dof(n_dof)
    for t in range(32, 8192 + 1, 32):
        params.t_pad = t
        if int(_GPU_SHMEM_BYTES(params)) > limit:
            return max(32, t - 32)
    return 8192


def trajectory_fits_gpu_shmem(t_len: int, n_dof: int = 6) -> bool:
    """True when ``pad_time(t_len)`` fits the GPU block shared-memory budget."""
    max_t_pad = max_gpu_trajectory_frames(n_dof)
    if max_t_pad <= 0:
        return False
    return pad_time(int(t_len)) <= max_t_pad


def pad_dof(n_dof: int) -> int:
    if n_dof <= 4:
        return 4
    if n_dof <= 8:
        return 8
    if n_dof <= 16:
        return 16
    return 32


def pad_time(length: int) -> int:
    return int(math.ceil(length / 32) * 32)


def pack_trajectories(
    q_list: list[NDArray[np.floating]],
    *,
    n_dof: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], int, int]:
    """Pack variable-length joint trajectories into padded ``(n_traj, d_pad, t_pad)``."""
    if not q_list:
        raise ValueError("q_list must be non-empty")
    n_dof = n_dof or int(q_list[0].shape[-1])
    d_pad = pad_dof(n_dof)
    t_pad = pad_time(max(q.shape[0] for q in q_list))
    n_traj = len(q_list)
    out = np.zeros((n_traj, d_pad, t_pad), dtype=np.float32)
    lengths = np.zeros(n_traj, dtype=np.int32)
    for i, q in enumerate(q_list):
        t_len = int(q.shape[0])
        lengths[i] = t_len
        out[i, :n_dof, :t_len] = np.asarray(q[:t_len, :n_dof], dtype=np.float32).T
    return out, lengths, d_pad, t_pad


def pack_targets(
    targets_list: list[NDArray[np.floating]],
    *,
    t_pad: int,
) -> NDArray[np.float32]:
    n_traj = len(targets_list)
    out = np.zeros((n_traj, t_pad, 6), dtype=np.float32)
    for i, tgt in enumerate(targets_list):
        t_len = int(tgt.shape[0])
        tgt = np.asarray(tgt, dtype=np.float32)
        tgt = tgt.copy()
        tgt[:, 3:6] = unwrap_euler_targets(tgt[:, 3:6])
        out[i, :t_len, :] = tgt
    return out


def retarget_params_from_config(
    model: pin.Model,
    config: RetargetConfig,
    *,
    t_pad: int,
    n_dof: int | None = None,
) -> dict:
    """Keyword arguments for ``retarget_trajectories_gpu`` from YAML config."""
    n_dof = n_dof or model.nv
    d_pad = pad_dof(n_dof)
    u = config.cost.units
    w = config.cost.weights
    q_neutral_full = np.asarray(pin.neutral(model), dtype=np.float32)
    q_neutral = q_neutral_full[:n_dof]
    return {
        "n_dof": n_dof,
        "n_outer_iters": 6,
        "n_temporal_iters": 4,
        "n_pose_refine_iters": 4,
        "alpha": 0.15,
        "step_size": 0.0,
        "temporal_step_size": 5.0e-6,
        "frames_per_tile": 256,
        "ik_damping": 1.0e-2,
        "control_hz": float(config.control_hz),
        "position_error_unit": float(u.position),
        "rotation_error_unit": float(np.deg2rad(u.rotation)),
        "joint_acceleration_error_unit": float(np.deg2rad(u.joint_acceleration)),
        "neutral_pose_error_unit": float(u.neutral_pose),
        "neutral_pose_weight": float(w.neutral_pose),
        "pos_weight": float(w.position),
        "rot_weight": float(w.rotation),
        "rot_weight_min_scale": float(w.rotation_min_scale),
        "joint_vel_weight": float(w.joint_velocity),
        "joint_acc_weight": float(w.joint_acceleration),
        "joint_velocity_error_unit": np.deg2rad(
            np.asarray(u.joint_velocity, dtype=np.float32)
        ),
        "q_neutral": q_neutral,
        "q_lower": np.asarray(model.lowerPositionLimit[:n_dof], dtype=np.float32),
        "q_upper": np.asarray(model.upperPositionLimit[:n_dof], dtype=np.float32),
        # Pinocchio may use nq > nv; FK kernels are generated for the first ``nv`` joints.
        "t_pad": t_pad,
        "d_pad": d_pad,
        "use_rotation_dls": 1,  # spatial_local 6x6 J; position-only if set to 0
        "rot_nu_clamp": 0.5,
    }


def refine_elbow_trajectory(
    model: pin.Model,
    q_traj: NDArray[np.floating],
    elbow_targets: NDArray[np.floating],
    config: RetargetConfig,
    *,
    n_iters: int = 10,
    step_size: float = 0.08,
) -> NDArray[np.float32]:
    """Pinocchio elbow-branch refinement (fastfk link poses differ from URDF frames)."""
    w = config.cost.weights.elbow_branch
    if w <= 0.0:
        return np.asarray(q_traj, dtype=np.float32)

    frames = target_elbow_frames(config, model)
    margin = float(config.cost.elbow_margin)
    unit = float(config.cost.units.elbow_side)
    t_len = int(q_traj.shape[0])
    data = model.createData()
    out = np.asarray(q_traj, dtype=np.float64).copy()
    n_dof = model.nv

    for t in range(t_len):
        target_side = float(elbow_targets[t])
        if abs(target_side) < 1e-6:
            continue
        q = pin.neutral(model)
        q[:n_dof] = out[t, :n_dof]
        q = clamp_configuration(model, q)
        for _ in range(n_iters):
            side = elbow_side_scalar(model, data, q, frames)
            violation = margin - target_side * side
            if violation <= 0.0:
                break
            grad = np.zeros(n_dof, dtype=np.float64)
            eps = 1.0e-4
            for d in range(n_dof):
                qp = q.copy()
                qp[d] += eps
                qp = clamp_configuration(model, qp)
                qm = q.copy()
                qm[d] -= eps
                qm = clamp_configuration(model, qm)
                sp = elbow_side_scalar(model, data, qp, frames)
                sm = elbow_side_scalar(model, data, qm, frames)
                vp = margin - target_side * sp
                vm = margin - target_side * sm
                cost_p = w * (vp / unit) ** 2 if vp > 0.0 else 0.0
                cost_m = w * (vm / unit) ** 2 if vm > 0.0 else 0.0
                grad[d] = (cost_p - cost_m) / (2.0 * eps)
            q = clamp_configuration(model, pin.integrate(model, q, -step_size * grad))
        out[t, :n_dof] = q[:n_dof]

    return out.astype(np.float32)


def retarget_cartesian_trajectories(
    model: pin.Model,
    cartesian_list: list[NDArray[np.floating]],
    config: RetargetConfig,
    *,
    position_scales_list: list[NDArray[np.floating]] | None = None,
    elbow_targets_list: list[NDArray[np.floating]] | None = None,
    **gpu_kwargs,
) -> list[NDArray[np.float32]]:
    """Batch GPU retarget for many demos in one launch (mixed lengths via max ``T_pad``)."""
    if not cartesian_list:
        return []
    if position_scales_list is not None and len(position_scales_list) != len(cartesian_list):
        raise ValueError("position_scales_list length must match cartesian_list")
    if elbow_targets_list is not None and len(elbow_targets_list) != len(cartesian_list):
        raise ValueError("elbow_targets_list length must match cartesian_list")

    targets = [np.asarray(t, dtype=np.float32) for t in cartesian_list]
    return _retarget_cartesian_group(
        model,
        targets,
        config,
        position_scales_list=position_scales_list,
        elbow_targets_list=elbow_targets_list,
        **gpu_kwargs,
    )


def pack_initial_gpu_q(
    model: pin.Model,
    t_lens: list[int],
) -> tuple[NDArray[np.float32], NDArray[np.int32], int, int]:
    """Neutral pose at every frame (GPU plan: Jacobi + temporal, not per-frame host IK)."""
    n_dof = model.nv
    q_neutral = np.asarray(pin.neutral(model)[:n_dof], dtype=np.float32)
    q_list = [np.tile(q_neutral, (t_len, 1)) for t_len in t_lens]
    return pack_trajectories(q_list, n_dof=n_dof)


def _retarget_cartesian_group_once(
    model: pin.Model,
    cartesian_targets: list[NDArray[np.floating]],
    config: RetargetConfig,
    *,
    position_scales_list: list[NDArray[np.floating]] | None = None,
    **gpu_kwargs,
) -> list[NDArray[np.float32]]:
    """Single GPU launch; trajectories may differ in length (padded to common ``t_pad``)."""
    n_dof = model.nv
    t_lens = [int(t.shape[0]) for t in cartesian_targets]
    q_in, lengths, _d_pad, t_pad = pack_initial_gpu_q(model, t_lens)
    tgt_batch = pack_targets(cartesian_targets, t_pad=t_pad)
    kwargs = retarget_params_from_config(model, config, t_pad=t_pad, n_dof=n_dof)
    kwargs.update(gpu_kwargs)
    scales_batch = None
    if position_scales_list is not None:
        scales_batch = np.zeros((len(cartesian_targets), t_pad), dtype=np.float32)
        for i, (scales, t_len) in enumerate(zip(position_scales_list, t_lens, strict=True)):
            scales_batch[i, :t_len] = np.asarray(scales[:t_len], dtype=np.float32)
    q_out = retarget_trajectories_gpu(
        q_in,
        tgt_batch,
        lengths,
        scales_batch,
        **kwargs,
    )
    return [
        np.asarray(q_out[i, :n_dof, :t_len], dtype=np.float32).T
        for i, t_len in enumerate(t_lens)
    ]


def _retarget_cartesian_group(
    model: pin.Model,
    cartesian_targets: list[NDArray[np.floating]],
    config: RetargetConfig,
    *,
    position_scales_list: list[NDArray[np.floating]] | None = None,
    elbow_targets_list: list[NDArray[np.floating]] | None = None,
    **gpu_kwargs,
) -> list[NDArray[np.float32]]:
    n_dof = model.nv
    t_lens = [int(t.shape[0]) for t in cartesian_targets]
    max_t_pad = max_gpu_trajectory_frames(n_dof)
    if max_t_pad <= 0:
        raise RuntimeError("GPU retarget shared-memory limit unavailable")
    for t_len in t_lens:
        if not trajectory_fits_gpu_shmem(t_len, n_dof):
            raise ValueError(
                f"trajectory length {t_len} frames (T_pad={pad_time(t_len)}) exceeds GPU "
                f"shared-memory limit (max T_pad={max_t_pad}); exclude or shorten the demo"
            )
    q_trajs = _retarget_cartesian_group_once(
        model,
        cartesian_targets,
        config,
        position_scales_list=position_scales_list,
        **gpu_kwargs,
    )

    if elbow_targets_list is not None:
        for i, elbow_tgt in enumerate(elbow_targets_list):
            q_trajs[i] = refine_elbow_trajectory(
                model, q_trajs[i], np.asarray(elbow_tgt[: t_lens[i]]), config
            )
    return q_trajs


def retarget_cartesian_trajectory(
    model: pin.Model,
    cartesian_targets: NDArray[np.floating],
    config: RetargetConfig,
    *,
    position_scales: NDArray[np.floating] | None = None,
    elbow_targets: NDArray[np.floating] | None = None,
    **gpu_kwargs,
) -> NDArray[np.float32]:
    """Retarget one demo trajectory ``(T, 6)`` xyz + intrinsic xyz Euler → ``(T, nv)`` joints."""
    targets = np.asarray(cartesian_targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError("cartesian_targets must have shape (T, 6)")
    return _retarget_cartesian_group(
        model,
        [targets],
        config,
        position_scales_list=None if position_scales is None else [position_scales],
        elbow_targets_list=None if elbow_targets is None else [elbow_targets],
        **gpu_kwargs,
    )[0]


def seed_trajectories(
    model: pin.Model,
    targets: NDArray[np.floating],
    lengths: NDArray[np.integer],
    config: RetargetConfig,
) -> NDArray[np.float32]:
    """Per-frame damped IK seed, warm-started along the trajectory."""
    n_traj, t_pad, _ = targets.shape
    n_dof = model.nq
    d_pad = pad_dof(n_dof)
    q = np.zeros((n_traj, d_pad, t_pad), dtype=np.float32)
    data = model.createData()
    frame_id = tool_frame_id(model, config.frames.tool)
    q0_full = pin.neutral(model)
    for i in range(n_traj):
        q_prev = seed_ik(model, data, q0_full, targets[i, 0], frame_id, config=config)
        q[i, :n_dof, 0] = np.asarray(q_prev[:n_dof], dtype=np.float32)
        for t in range(1, int(lengths[i])):
            q_prev = seed_ik(
                model,
                data,
                q_prev,
                targets[i, t],
                frame_id,
                config=config,
            )
            q[i, :n_dof, t] = np.asarray(q_prev[:n_dof], dtype=np.float32)
    return q


def retarget_trajectories_gpu(
    q_in: NDArray[np.floating],
    targets: NDArray[np.floating],
    lengths: NDArray[np.integer],
    position_scales: NDArray[np.floating] | None = None,
    **kwargs,
) -> NDArray[np.float32]:
    """Batch GPU retarget; requires CUDA ``_native`` build."""
    if _GPU_NATIVE is None:
        raise RuntimeError("GPU retarget not available (rebuild with CUDA)")
    q_in_c = np.ascontiguousarray(q_in, dtype=np.float32)
    return _GPU_NATIVE(
        q_in_c,
        np.ascontiguousarray(targets, dtype=np.float32),
        np.ascontiguousarray(lengths, dtype=np.int32),
        None if position_scales is None else np.ascontiguousarray(position_scales, dtype=np.float32),
        **kwargs,
    )
