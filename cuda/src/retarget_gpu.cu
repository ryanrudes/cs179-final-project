#include "retarget_gpu.cuh"

#include "fastfk_device_math.cuh"
#include "generated/ur3e_tool0_fk_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs179 {

__constant__ RetargetGpuParams c_rt_params;

void retarget_gpu_set_params(const RetargetGpuParams& params) {
    cudaError_t err = cudaMemcpyToSymbol(c_rt_params, &params, sizeof(RetargetGpuParams));
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyToSymbol c_rt_params: ") + cudaGetErrorString(err));
    }
}

namespace {

using gpu::fastfk_device_cos;
using gpu::fastfk_device_sin;
using gpu::ur3e::kTool0JacSize;

constexpr int kMaxDofPad = 32;
constexpr int kWarpSize = 32;
constexpr int kWarpDqPad = 8;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__device__ inline int q_index(int d, int t, int t_pad) {
    return d * t_pad + t;
}

__device__ inline float read_q(const float* buf, int d, int t, int t_pad, int n_dof) {
    if (d >= n_dof) {
        return 0.0f;
    }
    return buf[q_index(d, t, t_pad)];
}

__device__ inline void write_q(float* buf, int d, int t, int t_pad, int n_dof, float v) {
    if (d < n_dof) {
        buf[q_index(d, t, t_pad)] = v;
    }
}

__device__ void euler_xyz_to_rotation(const float* euler, float* R) {
    const float cx = fastfk_device_cos(euler[0]);
    const float sx = fastfk_device_sin(euler[0]);
    const float cy = fastfk_device_cos(euler[1]);
    const float sy = fastfk_device_sin(euler[1]);
    const float cz = fastfk_device_cos(euler[2]);
    const float sz = fastfk_device_sin(euler[2]);

    R[0] = cy * cz;
    R[1] = cz * sx * sy - cx * sz;
    R[2] = cx * cz * sy + sx * sz;
    R[3] = cy * sz;
    R[4] = cx * cz + sx * sy * sz;
    R[5] = -cz * sx + cx * sy * sz;
    R[6] = -sy;
    R[7] = cy * sx;
    R[8] = cx * cy;
}

/// SO(3) log with the θ≈π branch handled like Pinocchio/Eigen. The naive
/// ``0.5 · θ/sinθ · antisym(R)`` formula divides float32 noise by ~0 near π,
/// producing unbounded rotation vectors (and, through the SE(3) V⁻¹ coupling,
/// absurd translation errors). ``‖rotvec‖ ≤ π`` always holds here.
__device__ void log3_rotvec(const float* R_rel, float* rotvec) {
    const float trace = R_rel[0] + R_rel[4] + R_rel[8];
    float w[3] = {R_rel[7] - R_rel[5], R_rel[2] - R_rel[6], R_rel[3] - R_rel[1]};
    const float cos_angle = fmaxf(-1.0f, fminf(1.0f, 0.5f * (trace - 1.0f)));
    const float angle = acosf(cos_angle);

    if (cos_angle > -0.99f) {
        // Direction from the antisymmetric part, magnitude from the trace.
        // Analytically ‖w‖ = 2·sinθ, but float32 orthonormality noise in w is
        // amplified by 1/sinθ near the branch cutoff, letting ‖rotvec‖ exceed
        // θ (and π). Normalizing w pins the magnitude to θ ≤ π exactly.
        const float w_norm = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (angle < 1.0e-2f || w_norm < 1.0e-8f) {
            // Tiny rotation: 0.5·w is more accurate than the acos-derived θ.
            rotvec[0] = 0.5f * w[0];
            rotvec[1] = 0.5f * w[1];
            rotvec[2] = 0.5f * w[2];
            return;
        }
        const float scale = angle / w_norm;
        rotvec[0] = scale * w[0];
        rotvec[1] = scale * w[1];
        rotvec[2] = scale * w[2];
        return;
    }

    // Near π: recover the axis from the diagonal of R (stable), fix relative
    // signs from the symmetric off-diagonals, and the overall sign from the
    // antisymmetric part (ambiguous at exactly π, where ±axis are equivalent).
    const float one_m_cos = 1.0f - cos_angle;
    float ax = sqrtf(fmaxf(0.0f, (R_rel[0] - cos_angle) / one_m_cos));
    float ay = sqrtf(fmaxf(0.0f, (R_rel[4] - cos_angle) / one_m_cos));
    float az = sqrtf(fmaxf(0.0f, (R_rel[8] - cos_angle) / one_m_cos));
    if (ax >= ay && ax >= az) {
        ay = copysignf(ay, R_rel[1] + R_rel[3]);
        az = copysignf(az, R_rel[2] + R_rel[6]);
    } else if (ay >= az) {
        ax = copysignf(ax, R_rel[1] + R_rel[3]);
        az = copysignf(az, R_rel[5] + R_rel[7]);
    } else {
        ax = copysignf(ax, R_rel[2] + R_rel[6]);
        ay = copysignf(ay, R_rel[5] + R_rel[7]);
    }
    if (ax * w[0] + ay * w[1] + az * w[2] < 0.0f) {
        ax = -ax;
        ay = -ay;
        az = -az;
    }
    const float inv_norm =
        rsqrtf(fmaxf(1.0e-12f, ax * ax + ay * ay + az * az));
    rotvec[0] = angle * ax * inv_norm;
    rotvec[1] = angle * ay * inv_norm;
    rotvec[2] = angle * az * inv_norm;
}

__device__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ float joint_difference(float q1, float q0) {
    return atan2f(fastfk_device_sin(q1 - q0), fastfk_device_cos(q1 - q0));
}

/// ``V(θ)⁻¹ · v`` for the SE(3) log: couples the translation component to the
/// rotation vector ``w`` so the result matches Pinocchio's ``log6`` exactly.
__device__ void se3_log_translation(const float* w, const float* v, float* out) {
    const float theta_sq = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const float theta = sqrtf(theta_sq);

    float wxv[3] = {
        w[1] * v[2] - w[2] * v[1],
        w[2] * v[0] - w[0] * v[2],
        w[0] * v[1] - w[1] * v[0],
    };
    float wxwxv[3] = {
        w[1] * wxv[2] - w[2] * wxv[1],
        w[2] * wxv[0] - w[0] * wxv[2],
        w[0] * wxv[1] - w[1] * wxv[0],
    };

    float c2;  // coefficient of [w]×² in V⁻¹
    if (theta < 1.0e-4f) {
        c2 = 1.0f / 12.0f + theta_sq / 720.0f;
    } else {
        const float half = 0.5f * theta;
        c2 = (1.0f - half * fastfk_device_cos(half) / fastfk_device_sin(half)) / theta_sq;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        out[i] = v[i] - 0.5f * wxv[i] + c2 * wxwxv[i];
    }
}

/// Full SE(3) ``log6`` error matching Pinocchio's ``pin.log(oMf⁻¹ · oMdes)``
/// (translation coupled through V⁻¹; same convention as the CPU solver).
__device__ void pose_log6_error(
    const float* R_curr,
    const float* p_curr,
    const float* target6,
    float* err6) {
    const float dp[3] = {
        target6[0] - p_curr[0],
        target6[1] - p_curr[1],
        target6[2] - p_curr[2],
    };
    float t_local[3];
    t_local[0] = R_curr[0] * dp[0] + R_curr[3] * dp[1] + R_curr[6] * dp[2];
    t_local[1] = R_curr[1] * dp[0] + R_curr[4] * dp[1] + R_curr[7] * dp[2];
    t_local[2] = R_curr[2] * dp[0] + R_curr[5] * dp[1] + R_curr[8] * dp[2];

    float R_tgt[9];
    euler_xyz_to_rotation(target6 + 3, R_tgt);
    float R_rel[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_rel[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                R_rel[i * 3 + j] += R_curr[k * 3 + i] * R_tgt[k * 3 + j];
            }
        }
    }
    log3_rotvec(R_rel, err6 + 3);
    se3_log_translation(err6 + 3, t_local, err6);
}

__device__ void fk_tool0(const float* q_frame, float* R, float* p, float* J) {
#pragma unroll
    for (int i = 0; i < kTool0JacSize; ++i) {
        J[i] = 0.0f;
    }
    gpu::ur3e::tool0_fk_jacobian(q_frame, R, p, J);
}

__device__ void damped_least_squares_delta_3(
    const float* J_lin,
    const float* nu3,
    float damp,
    float* dq) {
    float JJt[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        JJt[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                sum += J_lin[k * 3 + i] * J_lin[k * 3 + j];
            }
            JJt[i * 3 + j] = sum;
        }
        JJt[i * 3 + i] += damp;
    }

    float x[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        x[i] = nu3[i];
    }

#pragma unroll
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        float pivot_val = fabsf(JJt[pivot * 3 + col]);
#pragma unroll
        for (int r = col + 1; r < 3; ++r) {
            const float v = fabsf(JJt[r * 3 + col]);
            if (v > pivot_val) {
                pivot_val = v;
                pivot = r;
            }
        }
        if (pivot_val < 1.0e-12f) {
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                dq[i] = 0.0f;
            }
            return;
        }
        if (pivot != col) {
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float tmp = JJt[col * 3 + c];
                JJt[col * 3 + c] = JJt[pivot * 3 + c];
                JJt[pivot * 3 + c] = tmp;
            }
            const float tx = x[col];
            x[col] = x[pivot];
            x[pivot] = tx;
        }
        const float diag = JJt[col * 3 + col];
#pragma unroll
        for (int r = col + 1; r < 3; ++r) {
            const float f = JJt[r * 3 + col] / diag;
#pragma unroll
            for (int c = col; c < 3; ++c) {
                JJt[r * 3 + c] -= f * JJt[col * 3 + c];
            }
            x[r] -= f * x[col];
        }
    }

#pragma unroll
    for (int i = 2; i >= 0; --i) {
        float sum = x[i];
#pragma unroll
        for (int j = i + 1; j < 3; ++j) {
            sum -= JJt[i * 3 + j] * x[j];
        }
        x[i] = sum / JJt[i * 3 + i];
    }

#pragma unroll
    for (int d = 0; d < 6; ++d) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            sum += J_lin[d * 3 + i] * x[i];
        }
        dq[d] = sum;
    }
}

__device__ void damped_least_squares_delta_6(
    const float* J_pin,
    const float* nu,
    float damp,
    float* dq) {
    float JJt[36];
#pragma unroll
    for (int i = 0; i < 36; ++i) {
        JJt[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) {
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                sum += J_pin[i * 6 + k] * J_pin[j * 6 + k];
            }
            JJt[i * 6 + j] = sum;
        }
        JJt[i * 6 + i] += damp;
    }

    float x[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        x[i] = nu[i];
    }

#pragma unroll
    for (int col = 0; col < 6; ++col) {
        int pivot = col;
        float pivot_val = fabsf(JJt[pivot * 6 + col]);
#pragma unroll
        for (int r = col + 1; r < 6; ++r) {
            const float v = fabsf(JJt[r * 6 + col]);
            if (v > pivot_val) {
                pivot_val = v;
                pivot = r;
            }
        }
        if (pivot_val < 1.0e-12f) {
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                dq[i] = 0.0f;
            }
            return;
        }
        if (pivot != col) {
#pragma unroll
            for (int c = 0; c < 6; ++c) {
                const float tmp = JJt[col * 6 + c];
                JJt[col * 6 + c] = JJt[pivot * 6 + c];
                JJt[pivot * 6 + c] = tmp;
            }
            const float tx = x[col];
            x[col] = x[pivot];
            x[pivot] = tx;
        }
        const float diag = JJt[col * 6 + col];
#pragma unroll
        for (int r = col + 1; r < 6; ++r) {
            const float f = JJt[r * 6 + col] / diag;
#pragma unroll
            for (int c = col; c < 6; ++c) {
                JJt[r * 6 + c] -= f * JJt[col * 6 + c];
            }
            x[r] -= f * x[col];
        }
    }

#pragma unroll
    for (int i = 5; i >= 0; --i) {
        float sum = x[i];
#pragma unroll
        for (int j = i + 1; j < 6; ++j) {
            sum -= JJt[i * 6 + j] * x[j];
        }
        x[i] = sum / JJt[i * 6 + i];
    }

#pragma unroll
    for (int d = 0; d < 6; ++d) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            sum += J_pin[i * 6 + d] * x[i];
        }
        dq[d] = sum;
    }
}

/// Temporal + neutral-pose gradient w.r.t. ``q[d,t]`` (full trajectory visible in SMEM).
__device__ float temporal_grad_d(
    const float* q_read,
    int d,
    int t,
    int t_len,
    int t_pad,
    int n_dof) {
    const float qd = read_q(q_read, d, t, t_pad, n_dof);
    float grad = 0.0f;

    const float vel_unit = c_rt_params.joint_velocity_error_unit[d];
    const float acc_unit = c_rt_params.joint_acceleration_error_unit;
    const float hz = c_rt_params.control_hz;
    const float hz2 = hz * hz;
    const float neu_unit = c_rt_params.neutral_pose_error_unit;
    const float dq_neu = joint_difference(qd, c_rt_params.q_neutral[d]);
    grad += 2.0f * c_rt_params.neutral_pose_weight * dq_neu / (neu_unit * neu_unit);

  // Velocity: one-sided at boundaries, central interior (GPU_PLAN).
    if (t == 0 && t_len > 1) {
        const float qp = read_q(q_read, d, 1, t_pad, n_dof);
        const float vel = joint_difference(qp, qd) * hz / vel_unit;
        grad += 2.0f * c_rt_params.joint_vel_weight * vel * hz / vel_unit;
    } else if (t == t_len - 1 && t_len > 1) {
        const float qm = read_q(q_read, d, t - 1, t_pad, n_dof);
        const float vel = joint_difference(qd, qm) * hz / vel_unit;
        grad += 2.0f * c_rt_params.joint_vel_weight * vel * (-hz) / vel_unit;
    } else if (t > 0 && t < t_len - 1) {
        const float qm = read_q(q_read, d, t - 1, t_pad, n_dof);
        const float qp = read_q(q_read, d, t + 1, t_pad, n_dof);
        const float vel_m = joint_difference(qd, qm) * hz / vel_unit;
        const float vel_p = joint_difference(qp, qd) * hz / vel_unit;
        grad += 2.0f * c_rt_params.joint_vel_weight * vel_m * hz / vel_unit;
        grad += 2.0f * c_rt_params.joint_vel_weight * vel_p * (-hz) / vel_unit;
    }

    if (t > 1 && t < t_len - 1) {
        const float qm = read_q(q_read, d, t - 1, t_pad, n_dof);
        const float qmm = read_q(q_read, d, t - 2, t_pad, n_dof);
        const float dq_tm = joint_difference(qd, qm);
        const float dq_mm = joint_difference(qm, qmm);
        const float acc = (dq_tm - dq_mm) * hz2 / acc_unit;
        grad += 2.0f * c_rt_params.joint_acc_weight * acc * (hz2 / acc_unit);
    } else if (t == t_len - 1 && t_len > 2) {
        const float qm = read_q(q_read, d, t - 1, t_pad, n_dof);
        const float qmm = read_q(q_read, d, t - 2, t_pad, n_dof);
        const float dq_m = joint_difference(qd, qm);
        const float dq_mm = joint_difference(qm, qmm);
        const float acc = (dq_m - dq_mm) * hz2 / acc_unit;
        grad += 2.0f * c_rt_params.joint_acc_weight * acc * (hz2 / acc_unit);
    }

    return grad;
}

__device__ void load_q_frame(
    const float* q_read,
    float* q_frame,
    int t,
    int t_pad,
    int n_dof) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q_frame[i] = (i < n_dof) ? read_q(q_read, i, t, t_pad, n_dof) : 0.0f;
    }
}

__device__ void project_joint_limits(float* q, int n_dof) {
    for (int d = 0; d < n_dof; ++d) {
        q[d] = clampf(q[d], c_rt_params.q_lower[d], c_rt_params.q_upper[d]);
    }
}

/// Lane 0: pose DLS ``dq`` into ``warp_dq[warp * kWarpDqPad + d]``.
__device__ void warp_pose_dls(
    const float* q_read,
    const float* targets_traj,
    int t,
    int t_pad,
    int n_dof,
    float* warp_dq) {
    float q_frame[8];
    float R[9];
    float p[3];
    float J_pin[36];
    float frame_err6[6];
    float frame_dq[6];

    load_q_frame(q_read, q_frame, t, t_pad, n_dof);
    fk_tool0(q_frame, R, p, J_pin);
    pose_log6_error(R, p, targets_traj + t * 6, frame_err6);

    if (c_rt_params.use_rotation_dls != 0) {
        float nu6[6];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            nu6[i] = frame_err6[i];
        }
#pragma unroll
        for (int i = 3; i < 6; ++i) {
            nu6[i] = clampf(nu6[i], -c_rt_params.rot_nu_clamp, c_rt_params.rot_nu_clamp);
        }
        // Weight rotation rows relative to translation. Default (rot_row_scale=0)
        // derives the scale from the config error units (CPU cost: pos/pos_unit vs
        // rot/rot_unit); unweighted DLS treats 1 rad ≈ 1 m and over-prioritizes
        // rotation on unreachable poses. rot_row_scale=1 reproduces seed_ik.
        float w_rot = c_rt_params.rot_row_scale;
        if (w_rot <= 0.0f) {
            const float w_pos = c_rt_params.pos_weight / c_rt_params.position_error_unit;
            const float w_rot_raw = c_rt_params.rot_weight / c_rt_params.rotation_error_unit;
            w_rot = (w_pos > 0.0f) ? (w_rot_raw / w_pos) : 1.0f;
        }
#pragma unroll
        for (int i = 3; i < 6; ++i) {
            nu6[i] *= w_rot;
#pragma unroll
            for (int d = 0; d < 6; ++d) {
                J_pin[i * 6 + d] *= w_rot;
            }
        }
        damped_least_squares_delta_6(J_pin, nu6, c_rt_params.ik_damping, frame_dq);
    } else {
        float J_lin[18];
#pragma unroll
        for (int d = 0; d < 6; ++d) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                J_lin[d * 3 + i] = J_pin[(3 + i) * 6 + d];
            }
        }
        damped_least_squares_delta_3(J_lin, frame_err6, c_rt_params.ik_damping, frame_dq);
    }

#pragma unroll
    for (int d = 0; d < 6; ++d) {
        warp_dq[d] = frame_dq[d];
    }
}

__global__ void retarget_trajectory_kernel(
    const float* q_in,
    float* q_out,
    const float* targets,
    const int* lengths,
    const float* position_scales) {
    (void)position_scales;

    const int traj = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x >> 5);

    const int n_dof = c_rt_params.n_dof;
    const int d_pad = c_rt_params.d_pad;
    const int t_pad = c_rt_params.t_pad;
    const int t_len = lengths[traj];
    const int warps_per_block = blockDim.x / kWarpSize;

    extern __shared__ float smem[];
    const int q_block_elems = d_pad * t_pad;
    float* q_a = smem;
    float* q_b = smem + q_block_elems;
    float* warp_dq = smem + 2 * q_block_elems;

    float* q_curr = q_a;
    float* q_next = q_b;

    const int traj_stride = d_pad * t_pad;
    const float* q_in_traj = q_in + traj * traj_stride;
    const float* targets_traj = targets + traj * t_pad * 6;
    float* q_out_traj = q_out + traj * traj_stride;

    for (int idx = threadIdx.x; idx < q_block_elems; idx += blockDim.x) {
        const int d = idx / t_pad;
        const int t = idx % t_pad;
        const float v = (t < t_len && d < n_dof) ? q_in_traj[q_index(d, t, t_pad)] : 0.0f;
        q_curr[idx] = v;
    }
    for (int idx = threadIdx.x; idx < q_block_elems; idx += blockDim.x) {
        q_next[idx] = q_curr[idx];
    }
    __syncthreads();

    for (int pass = 0; pass < 3; ++pass) {
        const int n_pose_iters =
            (pass == 0) ? c_rt_params.n_outer_iters
                        : (pass == 2) ? c_rt_params.n_pose_refine_iters : 0;
        const int n_temporal_iters = (pass == 1) ? c_rt_params.n_temporal_iters : 0;
        const int n_iters = (pass == 1) ? n_temporal_iters : n_pose_iters;
        if (n_iters <= 0) {
            continue;
        }
        const bool temporal_only = (pass == 1);

        for (int outer = 0; outer < n_iters; ++outer) {
            for (int tile = 0; tile < t_len; tile += warps_per_block) {
                const int t = tile + warp;
                const bool active = (t < t_len);

                if (active && !temporal_only && lane == 0) {
                    warp_pose_dls(
                        q_curr,
                        targets_traj,
                        t,
                        t_pad,
                        n_dof,
                        warp_dq + warp * kWarpDqPad);
                }
                if (!temporal_only) {
                    __syncwarp();
                }

                if (active && lane < n_dof) {
                    const float q0 = read_q(q_curr, lane, t, t_pad, n_dof);
                    float q_solved = q0;
                    if (temporal_only) {
                        const float tgrad = temporal_grad_d(
                            q_curr, lane, t, t_len, t_pad, n_dof);
                        q_solved = q0 - c_rt_params.temporal_step_size * tgrad;
                    } else {
                        const float dq = warp_dq[warp * kWarpDqPad + lane];
                        q_solved = q0 + c_rt_params.alpha * dq;
                        if (c_rt_params.step_size > 0.0f) {
                            const float tgrad = temporal_grad_d(
                                q_curr, lane, t, t_len, t_pad, n_dof);
                            q_solved -= c_rt_params.step_size * tgrad;
                        }
                    }
                    q_solved = clampf(
                        q_solved, c_rt_params.q_lower[lane], c_rt_params.q_upper[lane]);
                    write_q(q_next, lane, t, t_pad, n_dof, q_solved);
                } else if (active && lane >= n_dof) {
                    write_q(q_next, lane, t, t_pad, n_dof, read_q(q_curr, lane, t, t_pad, n_dof));
                }
            }
            __syncthreads();

            float* tmp = q_curr;
            q_curr = q_next;
            q_next = tmp;
        }
    }

    for (int idx = threadIdx.x; idx < q_block_elems; idx += blockDim.x) {
        const int d = idx / t_pad;
        const int t = idx % t_pad;
        if (t < t_len && d < n_dof) {
            float qf[8];
            load_q_frame(q_curr, qf, t, t_pad, n_dof);
            project_joint_limits(qf, n_dof);
            q_out_traj[q_index(d, t, t_pad)] = qf[d];
        }
    }
}

/// One thread per frame: FK + log6 pose error + joint speed (no shared memory).
/// Matches the CPU evaluator (`pin.log(oMf⁻¹ · oMdes)`).
__global__ void evaluate_trajectory_metrics_kernel(
    const float* q,
    const float* targets,
    const int* lengths,
    float* position_errors,
    float* rotation_errors,
    float* joint_speeds,
    int d_pad,
    int t_pad,
    int n_dof,
    float control_hz) {
    const int traj = static_cast<int>(blockIdx.y);
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int t_len = lengths[traj];
    if (t >= t_pad) {
        return;
    }

    const int out_idx = traj * t_pad + t;
    if (t >= t_len) {
        position_errors[out_idx] = 0.0f;
        rotation_errors[out_idx] = 0.0f;
        joint_speeds[out_idx] = 0.0f;
        return;
    }

    const float* q_traj = q + static_cast<std::size_t>(traj) * d_pad * t_pad;
    const float* target6 = targets + (static_cast<std::size_t>(traj) * t_pad + t) * 6;

    float q_frame[8];
    float R[9];
    float p[3];
    float J_pin[36];
    float err6[6];

    load_q_frame(q_traj, q_frame, t, t_pad, n_dof);
    fk_tool0(q_frame, R, p, J_pin);
    pose_log6_error(R, p, target6, err6);

    position_errors[out_idx] =
        sqrtf(err6[0] * err6[0] + err6[1] * err6[1] + err6[2] * err6[2]);
    rotation_errors[out_idx] =
        sqrtf(err6[3] * err6[3] + err6[4] * err6[4] + err6[5] * err6[5]);

    float speed_sq = 0.0f;
    if (t > 0) {
        for (int d = 0; d < n_dof; ++d) {
            const float dq = joint_difference(
                read_q(q_traj, d, t, t_pad, n_dof),
                read_q(q_traj, d, t - 1, t_pad, n_dof));
            const float v = dq * control_hz;
            speed_sq += v * v;
        }
    }
    joint_speeds[out_idx] = sqrtf(speed_sq);
}

}  // namespace

void evaluate_trajectories_gpu(
    const float* q,
    const float* targets,
    const int* lengths,
    float* position_errors,
    float* rotation_errors,
    float* joint_speeds,
    int n_traj,
    int d_pad,
    int t_pad,
    int n_dof,
    float control_hz) {
    if (n_traj <= 0) {
        return;
    }
    if (n_dof <= 0 || n_dof > 8) {
        throw std::invalid_argument("n_dof must be in (0, 8] for trajectory evaluation");
    }
    if (d_pad < n_dof || t_pad <= 0) {
        throw std::invalid_argument("invalid d_pad/t_pad for trajectory evaluation");
    }

    float* d_q = nullptr;
    float* d_targets = nullptr;
    int* d_lengths = nullptr;
    float* d_pos_err = nullptr;
    float* d_rot_err = nullptr;
    float* d_speed = nullptr;

    const std::size_t q_elems = static_cast<std::size_t>(n_traj) * d_pad * t_pad;
    const std::size_t target_elems = static_cast<std::size_t>(n_traj) * t_pad * 6;
    const std::size_t out_elems = static_cast<std::size_t>(n_traj) * t_pad;

    check_cuda(cudaMalloc(&d_q, q_elems * sizeof(float)), "cudaMalloc eval q");
    check_cuda(cudaMalloc(&d_targets, target_elems * sizeof(float)), "cudaMalloc eval targets");
    check_cuda(
        cudaMalloc(&d_lengths, static_cast<std::size_t>(n_traj) * sizeof(int)),
        "cudaMalloc eval lengths");
    check_cuda(cudaMalloc(&d_pos_err, out_elems * sizeof(float)), "cudaMalloc eval pos_err");
    check_cuda(cudaMalloc(&d_rot_err, out_elems * sizeof(float)), "cudaMalloc eval rot_err");
    check_cuda(cudaMalloc(&d_speed, out_elems * sizeof(float)), "cudaMalloc eval speed");

    check_cuda(
        cudaMemcpy(d_q, q, q_elems * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy eval q");
    check_cuda(
        cudaMemcpy(d_targets, targets, target_elems * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy eval targets");
    check_cuda(
        cudaMemcpy(
            d_lengths,
            lengths,
            static_cast<std::size_t>(n_traj) * sizeof(int),
            cudaMemcpyHostToDevice),
        "cudaMemcpy eval lengths");

    constexpr int kEvalBlock = 256;
    const dim3 block(kEvalBlock);
    const dim3 grid((t_pad + kEvalBlock - 1) / kEvalBlock, static_cast<unsigned>(n_traj));
    evaluate_trajectory_metrics_kernel<<<grid, block>>>(
        d_q, d_targets, d_lengths, d_pos_err, d_rot_err, d_speed,
        d_pad, t_pad, n_dof, control_hz);
    check_cuda(cudaGetLastError(), "evaluate_trajectory_metrics_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize eval");

    check_cuda(
        cudaMemcpy(position_errors, d_pos_err, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy eval pos_err");
    check_cuda(
        cudaMemcpy(rotation_errors, d_rot_err, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy eval rot_err");
    check_cuda(
        cudaMemcpy(joint_speeds, d_speed, out_elems * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy eval speed");

    cudaFree(d_q);
    cudaFree(d_targets);
    cudaFree(d_lengths);
    cudaFree(d_pos_err);
    cudaFree(d_rot_err);
    cudaFree(d_speed);
}

std::size_t retarget_gpu_warp_scratch_bytes(int frames_per_tile) {
    const int warps = std::max(1, frames_per_tile / kWarpSize);
    return static_cast<std::size_t>(warps) * kWarpDqPad * sizeof(float);
}

std::size_t retarget_gpu_shmem_bytes(const RetargetGpuParams& params) {
    const int frames = params.frames_per_tile > 0 ? params.frames_per_tile : kWarpSize;
    return 2U * static_cast<std::size_t>(params.d_pad) * static_cast<std::size_t>(params.t_pad) *
               sizeof(float) +
           retarget_gpu_warp_scratch_bytes(frames);
}

std::size_t retarget_gpu_block_shmem_limit_bytes(int device_index) {
    int device = device_index;
    if (device < 0) {
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    }
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    return std::max(
        static_cast<std::size_t>(prop.sharedMemPerBlock),
        static_cast<std::size_t>(prop.sharedMemPerBlockOptin));
}

void retarget_trajectories_gpu(
    const float* q_in,
    float* q_out,
    const float* targets,
    const int* lengths,
    const float* position_scales,
    int n_traj,
    const RetargetGpuParams& params) {
    if (n_traj <= 0) {
        return;
    }
    if (params.n_dof <= 0 || params.n_dof > kMaxDofPad) {
        throw std::invalid_argument("n_dof must be in (0, 32]");
    }
    if (params.d_pad < params.n_dof) {
        throw std::invalid_argument("d_pad must be >= n_dof");
    }
    if (params.t_pad <= 0) {
        throw std::invalid_argument("t_pad must be positive");
    }
    if (params.frames_per_tile < kWarpSize || (params.frames_per_tile % kWarpSize) != 0) {
        throw std::invalid_argument("frames_per_tile must be a positive multiple of 32");
    }
    if (params.frames_per_tile > 1024) {
        throw std::invalid_argument("frames_per_tile must be <= 1024");
    }

    retarget_gpu_set_params(params);

    const std::size_t shmem = retarget_gpu_shmem_bytes(params);
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    const std::size_t shmem_limit = retarget_gpu_block_shmem_limit_bytes(device);
    if (shmem > shmem_limit) {
        throw std::runtime_error(
            "trajectory too long for GPU shared memory (need " + std::to_string(shmem) +
            " bytes, limit " + std::to_string(shmem_limit) + ")");
    }

    float* d_q_in = nullptr;
    float* d_q_out = nullptr;
    float* d_targets = nullptr;
    int* d_lengths = nullptr;
    float* d_scales = nullptr;

    const std::size_t q_elems =
        static_cast<std::size_t>(n_traj) * params.d_pad * params.t_pad;
    const std::size_t target_elems = static_cast<std::size_t>(n_traj) * params.t_pad * 6;

    check_cuda(cudaMalloc(&d_q_in, q_elems * sizeof(float)), "cudaMalloc q_in");
    check_cuda(cudaMalloc(&d_q_out, q_elems * sizeof(float)), "cudaMalloc q_out");
    check_cuda(cudaMalloc(&d_targets, target_elems * sizeof(float)), "cudaMalloc targets");
    check_cuda(cudaMalloc(&d_lengths, static_cast<std::size_t>(n_traj) * sizeof(int)), "cudaMalloc lengths");

    if (position_scales != nullptr) {
        check_cuda(
            cudaMalloc(&d_scales, static_cast<std::size_t>(n_traj) * params.t_pad * sizeof(float)),
            "cudaMalloc scales");
        check_cuda(
            cudaMemcpy(
                d_scales,
                position_scales,
                static_cast<std::size_t>(n_traj) * params.t_pad * sizeof(float),
                cudaMemcpyHostToDevice),
            "cudaMemcpy scales");
    }

    check_cuda(cudaMemcpy(d_q_in, q_in, q_elems * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy q_in");
    check_cuda(
        cudaMemcpy(d_targets, targets, target_elems * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy targets");
    check_cuda(
        cudaMemcpy(d_lengths, lengths, static_cast<std::size_t>(n_traj) * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy lengths");

    const dim3 grid(static_cast<unsigned>(n_traj));
    const dim3 block(static_cast<unsigned>(params.frames_per_tile));

    if (shmem > static_cast<std::size_t>(prop.sharedMemPerBlock)) {
        check_cuda(
            cudaFuncSetAttribute(
                retarget_trajectory_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem)),
            "cudaFuncSetAttribute shmem");
    }

    retarget_trajectory_kernel<<<grid, block, shmem>>>(
        d_q_in, d_q_out, d_targets, d_lengths, d_scales);
    check_cuda(cudaGetLastError(), "retarget_trajectory_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(q_out, d_q_out, q_elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy q_out");

    cudaFree(d_q_in);
    cudaFree(d_q_out);
    cudaFree(d_targets);
    cudaFree(d_lengths);
    cudaFree(d_scales);
}

}  // namespace cs179
