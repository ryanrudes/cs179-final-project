#include "bindings_detail.hpp"
#include "cs179/retarget.hpp"
#include "cs179/retarget_params.hpp"

#include <pinocchio/bindings/python/pybind11.hpp>
#include <pinocchio/multibody/model.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>

namespace py = pybind11;

namespace {

cs179::RetargetParams params_from_dict(const py::dict& overrides) {
    cs179::RetargetParams params = cs179::default_retarget_params();
    auto set_double = [&](const char* key, double& field) {
        if (overrides.contains(key)) {
            field = overrides[key].cast<double>();
        }
    };
    auto set_int = [&](const char* key, int& field) {
        if (overrides.contains(key)) {
            field = overrides[key].cast<int>();
        }
    };

    set_double("position_error_unit", params.position_error_unit);
    set_double("rotation_error_unit", params.rotation_error_unit);
    set_double("joint_acceleration_error_unit", params.joint_acceleration_error_unit);
    set_double("neutral_pose_error_unit", params.neutral_pose_error_unit);
    set_double("neutral_pose_weight", params.neutral_pose_weight);
    set_double("pos_weight", params.pos_weight);
    set_double("rot_weight", params.rot_weight);
    set_double("rot_weight_min_scale", params.rot_weight_min_scale);
    set_double("joint_vel_weight", params.joint_vel_weight);
    set_double("joint_acc_weight", params.joint_acc_weight);
    set_double("elbow_branch_weight", params.elbow_branch_weight);
    set_double("elbow_branch_margin", params.elbow_branch_margin);
    set_double("elbow_branch_error_unit", params.elbow_branch_error_unit);
    set_int("seed_ik_n_iter", params.seed_ik_n_iter);
    set_double("seed_ik_dt", params.seed_ik_dt);
    set_double("seed_ik_damp", params.seed_ik_damp);
    set_double("seed_ik_convergence_tol", params.seed_ik_convergence_tol);
    set_int("solver_max_iter", params.solver_max_iter);
    set_double("solver_ftol", params.solver_ftol);
    set_double("solver_grad_eps", params.solver_grad_eps);

    if (overrides.contains("joint_velocity_error_unit")) {
        const auto arr = overrides["joint_velocity_error_unit"].cast<py::array_t<double>>();
        if (arr.size() < 1) {
            throw std::invalid_argument("joint_velocity_error_unit must be non-empty");
        }
        params.joint_velocity_error_unit =
            Eigen::Map<const Eigen::VectorXd>(arr.data(), arr.size());
    }
    if (overrides.contains("tool_frame")) {
        params.tool_frame = overrides["tool_frame"].cast<std::string>();
    }
    auto set_frame = [&](const char* key, std::string& field) {
        if (overrides.contains(key)) {
            field = overrides[key].cast<std::string>();
        }
    };
    set_frame("elbow_shoulder_frame", params.elbow_shoulder_frame);
    set_frame("elbow_mid_frame", params.elbow_mid_frame);
    set_frame("elbow_wrist_frame", params.elbow_wrist_frame);
    return params;
}

cs179::Retargeter make_retargeter(
    py::handle model_handle,
    double control_hz,
    py::dict params) {
    auto& model = pinocchio::python::from<pinocchio::Model&>(model_handle);
    return cs179::Retargeter(model, control_hz, params_from_dict(params));
}

}  // namespace

void bind_retarget(pybind11::module_& m) {
    py::module_::import("pinocchio");

    m.def(
        "seed_ik",
        [](py::handle model_handle,
           py::array_t<double, py::array::c_style | py::array::forcecast> q0,
           py::array_t<double, py::array::c_style | py::array::forcecast> target,
           const std::string& tool_frame,
           py::dict params) {
            auto& model = pinocchio::python::from<pinocchio::Model&>(model_handle);
            if (q0.ndim() != 1 || static_cast<std::size_t>(q0.shape(0)) != static_cast<std::size_t>(model.nq)) {
                throw std::invalid_argument("q0 must have shape (model.nq,)");
            }
            if (target.ndim() != 1 || target.shape(0) != 6) {
                throw std::invalid_argument("target must have shape (6,)");
            }
            pinocchio::Data data(model);
            Eigen::Map<const Eigen::VectorXd> q0_map(q0.data(), model.nq);
            Eigen::Map<const Eigen::VectorXd> target_map(target.data(), 6);
            const auto frame_id = model.getFrameId(tool_frame);
            cs179::RetargetParams p = params_from_dict(params);
            const Eigen::VectorXd q = cs179::seed_ik(model, data, q0_map, target_map, frame_id, p);
            py::array_t<double> out(static_cast<py::ssize_t>(q.size()));
            std::memcpy(out.mutable_data(), q.data(), static_cast<std::size_t>(q.size()) * sizeof(double));
            return out;
        },
        py::arg("model"),
        py::arg("q0"),
        py::arg("target"),
        py::arg("tool_frame") = "tool0",
        py::arg("params") = py::dict());

    py::class_<cs179::Retargeter>(m, "Retargeter")
        .def(
            py::init(&make_retargeter),
            py::arg("model"),
            py::arg("control_hz") = 15.0,
            py::arg("params") = py::dict())
        .def("set_position_scale", &cs179::Retargeter::set_position_scale, py::arg("scale"))
        .def("set_elbow_side_target", &cs179::Retargeter::set_elbow_side_target, py::arg("side"))
        .def(
            "reset_episode",
            [](cs179::Retargeter& retargeter,
               py::array_t<double, py::array::c_style | py::array::forcecast> target) {
                if (target.ndim() != 1 || target.shape(0) != 6) {
                    throw std::invalid_argument("target must have shape (6,)");
                }
                Eigen::Map<const Eigen::VectorXd> target_map(target.data(), 6);
                retargeter.reset_episode(target_map);
            },
            py::arg("target"))
        .def(
            "__call__",
            [](cs179::Retargeter& retargeter,
               py::array_t<double, py::array::c_style | py::array::forcecast> target) {
                if (target.ndim() != 1 || target.shape(0) != 6) {
                    throw std::invalid_argument("target must have shape (6,)");
                }
                Eigen::Map<const Eigen::VectorXd> target_map(target.data(), 6);
                const auto result = retargeter.retarget_frame(target_map);
                py::array_t<double> q_new(static_cast<py::ssize_t>(result.q_new.size()));
                std::memcpy(q_new.mutable_data(), result.q_new.data(), result.q_new.size() * sizeof(double));
                return py::make_tuple(
                    q_new,
                    result.translation,
                    result.euler_xyz,
                    result.position_error,
                    result.rotation_error,
                    result.success,
                    result.nit);
            },
            py::arg("target"))
        .def_property_readonly(
            "q",
            [](const cs179::Retargeter& retargeter) {
                const auto& q = retargeter.q();
                py::array_t<double> arr(static_cast<py::ssize_t>(q.size()));
                std::memcpy(arr.mutable_data(), q.data(), q.size() * sizeof(double));
                return arr;
            })
        .def_property_readonly(
            "q_prev",
            [](const cs179::Retargeter& retargeter) -> py::object {
                const auto* q_prev = retargeter.q_prev();
                if (q_prev == nullptr) {
                    return py::none();
                }
                py::array_t<double> arr(static_cast<py::ssize_t>(q_prev->size()));
                std::memcpy(arr.mutable_data(), q_prev->data(), q_prev->size() * sizeof(double));
                return arr;
            });
}
