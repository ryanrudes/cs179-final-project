#pragma once

#include <cstddef>
#include <cstdint>

namespace cs179 {

/// Host-visible parameters for GPU trajectory retargeting (UR3e, 6-DoF).
struct RetargetGpuParams {
    int n_dof = 6;
    int d_pad = 8;
    int t_pad = 0;
    int n_outer_iters = 6;
    /// Extra temporal-only Jacobi sweeps after the pose pass (0 = pose only).
    int n_temporal_iters = 4;
    /// Pose Jacobi sweeps after temporal smoothing to recover task error.
    int n_pose_refine_iters = 4;
    /// Active threads per block; must be a multiple of 32 (one warp per frame in a tile).
    int frames_per_tile = 256;
    float alpha = 0.15f;
    /// Legacy coupled pose+temporal step; prefer ``temporal_step_size`` with separate passes.
    float step_size = 0.0f;
    /// Joint-space temporal gradient step (velocity/acceleration).
    float temporal_step_size = 5.0e-6f;
    float control_hz = 15.0f;

    float position_error_unit = 0.01f;
    float rotation_error_unit = 0.08726646259971647f;  // 5 deg
    float joint_acceleration_error_unit = 13.9626340154326f;
    float neutral_pose_error_unit = 6.283185307179586f;
    float neutral_pose_weight = 0.05f;
    float pos_weight = 1.0f;
    float rot_weight = 1.0f;
    float rot_weight_min_scale = 0.35f;
    float joint_vel_weight = 0.05f;
    float joint_acc_weight = 0.05f;

    float joint_velocity_error_unit[8] = {
        1.5707963267948966f,
        1.5707963267948966f,
        1.5707963267948966f,
        3.141592653589793f,
        3.141592653589793f,
        3.141592653589793f,
        1.0f,
        1.0f,
    };
    float q_neutral[8] = {};
    float q_lower[8] = {};
    float q_upper[8] = {};
    float rot_fd_eps_scale = 1.0e-4f;
    /// Damping for per-frame pose DLS (``J J^T + damp I``).
    float ik_damping = 1.0e-2f;
    /// If non-zero, use 6D pose DLS with ``spatial_local`` Jacobian.
    int use_rotation_dls = 0;
    /// Clamp each rotational ``log6`` component before 6D DLS.
    float rot_nu_clamp = 0.5f;
};

/// Copy params to ``__constant__`` memory (call before each kernel launch).
void retarget_gpu_set_params(const RetargetGpuParams& params);

/// Batch retarget on GPU: one block per trajectory.
void retarget_trajectories_gpu(
    const float* q_in,
    float* q_out,
    const float* targets,
    const int* lengths,
    const float* position_scales,
    int n_traj,
    const RetargetGpuParams& params);

[[nodiscard]] std::size_t retarget_gpu_shmem_bytes(const RetargetGpuParams& params);

/// Per-warp ``dq`` scratch in dynamic shared memory (``frames_per_tile / 32`` warps).
[[nodiscard]] std::size_t retarget_gpu_warp_scratch_bytes(int frames_per_tile);

}  // namespace cs179
