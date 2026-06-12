#include "bindings_detail.hpp"
#include "kernels.cuh"
#include "retarget_gpu.cuh"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void check_contiguous_1d(const py::array& array, const char* name) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
}

int pad_time(int length) {
    return ((length + 31) / 32) * 32;
}

int pad_dof(int n_dof) {
    if (n_dof <= 4) {
        return 4;
    }
    if (n_dof <= 8) {
        return 8;
    }
    if (n_dof <= 16) {
        return 16;
    }
    return 32;
}

cs179::RetargetGpuParams params_from_kwargs(
    int n_dof,
    int t_pad,
    py::kwargs kwargs) {
    cs179::RetargetGpuParams p;
    p.n_dof = n_dof;
    p.d_pad = pad_dof(n_dof);
    p.t_pad = t_pad;

    auto get_f = [&](const char* key, float& field) {
        if (kwargs.contains(key)) {
            field = kwargs[key].cast<float>();
        }
    };
    auto get_i = [&](const char* key, int& field) {
        if (kwargs.contains(key)) {
            field = kwargs[key].cast<int>();
        }
    };

    get_i("n_outer_iters", p.n_outer_iters);
    get_i("n_temporal_iters", p.n_temporal_iters);
    get_i("n_pose_refine_iters", p.n_pose_refine_iters);
    get_i("frames_per_tile", p.frames_per_tile);
    get_f("alpha", p.alpha);
    get_f("step_size", p.step_size);
    get_f("temporal_step_size", p.temporal_step_size);
    get_f("control_hz", p.control_hz);
    get_f("position_error_unit", p.position_error_unit);
    get_f("rotation_error_unit", p.rotation_error_unit);
    get_f("joint_acceleration_error_unit", p.joint_acceleration_error_unit);
    get_f("neutral_pose_error_unit", p.neutral_pose_error_unit);
    get_f("neutral_pose_weight", p.neutral_pose_weight);
    get_f("pos_weight", p.pos_weight);
    get_f("rot_weight", p.rot_weight);
    get_f("rot_weight_min_scale", p.rot_weight_min_scale);
    get_f("joint_vel_weight", p.joint_vel_weight);
    get_f("joint_acc_weight", p.joint_acc_weight);
    get_f("rot_fd_eps_scale", p.rot_fd_eps_scale);
    get_f("ik_damping", p.ik_damping);
    if (kwargs.contains("use_rotation_dls")) {
        p.use_rotation_dls = kwargs["use_rotation_dls"].cast<int>() != 0 ? 1 : 0;
    }
    get_f("rot_nu_clamp", p.rot_nu_clamp);
    get_f("rot_row_scale", p.rot_row_scale);

    if (kwargs.contains("joint_velocity_error_unit")) {
        const auto arr =
            kwargs["joint_velocity_error_unit"].cast<py::array_t<float, py::array::c_style>>();
        if (arr.size() < 1) {
            throw std::invalid_argument("joint_velocity_error_unit must be non-empty");
        }
        const int n = static_cast<int>(std::min(arr.size(), static_cast<py::ssize_t>(8)));
        std::memcpy(p.joint_velocity_error_unit, arr.data(), static_cast<std::size_t>(n) * sizeof(float));
    }
    if (kwargs.contains("q_neutral")) {
        const auto arr = kwargs["q_neutral"].cast<py::array_t<float, py::array::c_style>>();
        const int n = static_cast<int>(std::min(arr.size(), static_cast<py::ssize_t>(8)));
        std::memcpy(p.q_neutral, arr.data(), static_cast<std::size_t>(n) * sizeof(float));
    }
    if (kwargs.contains("q_lower")) {
        const auto arr = kwargs["q_lower"].cast<py::array_t<float, py::array::c_style>>();
        const int n = static_cast<int>(std::min(arr.size(), static_cast<py::ssize_t>(8)));
        std::memcpy(p.q_lower, arr.data(), static_cast<std::size_t>(n) * sizeof(float));
    }
    if (kwargs.contains("q_upper")) {
        const auto arr = kwargs["q_upper"].cast<py::array_t<float, py::array::c_style>>();
        const int n = static_cast<int>(std::min(arr.size(), static_cast<py::ssize_t>(8)));
        std::memcpy(p.q_upper, arr.data(), static_cast<std::size_t>(n) * sizeof(float));
    }
    return p;
}

}  // namespace

void bind_cuda(pybind11::module_& m) {
    m.def(
        "vector_add",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> a,
           py::array_t<float, py::array::c_style | py::array::forcecast> b,
           py::array_t<float, py::array::c_style | py::array::forcecast> out) {
            check_contiguous_1d(a, "a");
            check_contiguous_1d(b, "b");
            check_contiguous_1d(out, "out");

            if (a.size() != b.size() || a.size() != out.size()) {
                throw std::invalid_argument("a, b, and out must have the same length");
            }

            const auto n = static_cast<std::size_t>(a.size());
            cs179::vector_add(a.data(), b.data(), out.mutable_data(), n);
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("out"));

    py::class_<cs179::RetargetGpuParams>(m, "RetargetGpuParams")
        .def(py::init<>())
        .def_readwrite("n_dof", &cs179::RetargetGpuParams::n_dof)
        .def_readwrite("d_pad", &cs179::RetargetGpuParams::d_pad)
        .def_readwrite("t_pad", &cs179::RetargetGpuParams::t_pad)
        .def_readwrite("n_outer_iters", &cs179::RetargetGpuParams::n_outer_iters)
        .def_readwrite("frames_per_tile", &cs179::RetargetGpuParams::frames_per_tile)
        .def_readwrite("alpha", &cs179::RetargetGpuParams::alpha)
        .def_readwrite("step_size", &cs179::RetargetGpuParams::step_size)
        .def_readwrite("control_hz", &cs179::RetargetGpuParams::control_hz);

    m.def(
        "retarget_gpu_shmem_bytes",
        &cs179::retarget_gpu_shmem_bytes,
        py::arg("params"));

    m.def(
        "retarget_gpu_block_shmem_limit_bytes",
        &cs179::retarget_gpu_block_shmem_limit_bytes,
        py::arg("device_index") = 0);

    m.def(
        "retarget_trajectories_gpu",
        [](py::array_t<float, py::array::c_style> q_in,
           py::array_t<float, py::array::c_style> targets,
           py::array_t<int, py::array::c_style> lengths,
           py::object position_scales,
           py::kwargs kwargs) {
            if (q_in.ndim() != 3) {
                throw std::invalid_argument("q_in must have shape (n_traj, d_pad, t_pad)");
            }
            if (targets.ndim() != 3 || targets.shape(2) != 6) {
                throw std::invalid_argument("targets must have shape (n_traj, t_pad, 6)");
            }
            if (lengths.ndim() != 1 || lengths.shape(0) != q_in.shape(0)) {
                throw std::invalid_argument("lengths must have shape (n_traj,)");
            }
            if (targets.shape(0) != q_in.shape(0) || targets.shape(1) != q_in.shape(2)) {
                throw std::invalid_argument("targets must align with q_in on traj and time axes");
            }

            const int n_traj = static_cast<int>(q_in.shape(0));
            const int d_pad = static_cast<int>(q_in.shape(1));
            const int t_pad = static_cast<int>(q_in.shape(2));
            const int n_dof = kwargs.contains("n_dof") ? kwargs["n_dof"].cast<int>() : 6;

            auto params = params_from_kwargs(n_dof, t_pad, kwargs);
            params.d_pad = d_pad;
            params.t_pad = t_pad;

            py::array_t<float> q_out({n_traj, d_pad, t_pad});
            std::memcpy(
                q_out.mutable_data(),
                q_in.data(),
                static_cast<std::size_t>(n_traj) * d_pad * t_pad * sizeof(float));

            const float* scales_ptr = nullptr;
            py::array_t<float, py::array::c_style> scales_arr;
            if (!position_scales.is_none()) {
                scales_arr = position_scales.cast<py::array_t<float, py::array::c_style>>();
                if (scales_arr.ndim() != 2 || scales_arr.shape(0) != n_traj || scales_arr.shape(1) != t_pad) {
                    throw std::invalid_argument("position_scales must have shape (n_traj, t_pad)");
                }
                scales_ptr = scales_arr.data();
            }

            cs179::retarget_trajectories_gpu(
                q_in.data(),
                q_out.mutable_data(),
                targets.data(),
                lengths.data(),
                scales_ptr,
                n_traj,
                params);
            return q_out;
        },
        py::arg("q_in"),
        py::arg("targets"),
        py::arg("lengths"),
        py::arg("position_scales") = py::none());

    m.def(
        "evaluate_trajectories_gpu",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> q,
           py::array_t<float, py::array::c_style | py::array::forcecast> targets,
           py::array_t<int, py::array::c_style | py::array::forcecast> lengths,
           int n_dof,
           float control_hz) {
            if (q.ndim() != 3) {
                throw std::invalid_argument("q must have shape (n_traj, d_pad, t_pad)");
            }
            if (targets.ndim() != 3 || targets.shape(2) != 6) {
                throw std::invalid_argument("targets must have shape (n_traj, t_pad, 6)");
            }
            if (lengths.ndim() != 1 || lengths.shape(0) != q.shape(0)) {
                throw std::invalid_argument("lengths must have shape (n_traj,)");
            }
            if (targets.shape(0) != q.shape(0) || targets.shape(1) != q.shape(2)) {
                throw std::invalid_argument("targets must align with q on traj and time axes");
            }

            const int n_traj = static_cast<int>(q.shape(0));
            const int d_pad = static_cast<int>(q.shape(1));
            const int t_pad = static_cast<int>(q.shape(2));

            py::array_t<float> pos_err({n_traj, t_pad});
            py::array_t<float> rot_err({n_traj, t_pad});
            py::array_t<float> joint_speed({n_traj, t_pad});

            cs179::evaluate_trajectories_gpu(
                q.data(),
                targets.data(),
                lengths.data(),
                pos_err.mutable_data(),
                rot_err.mutable_data(),
                joint_speed.mutable_data(),
                n_traj,
                d_pad,
                t_pad,
                n_dof,
                control_hz);
            return py::make_tuple(pos_err, rot_err, joint_speed);
        },
        py::arg("q"),
        py::arg("targets"),
        py::arg("lengths"),
        py::arg("n_dof") = 6,
        py::arg("control_hz") = 15.0f);
}
