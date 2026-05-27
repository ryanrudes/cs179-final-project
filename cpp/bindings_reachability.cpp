#include "bindings_detail.hpp"
#include "cs179/reach_envelope.hpp"

#include <pinocchio/bindings/python/pybind11.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

cs179::DirectionalReachEnvelope build_from_robot_py(
    py::handle model_handle,
    py::handle data_handle,
    std::size_t frame_id,
    std::size_t n_samples,
    int n_theta,
    int n_phi,
    int batch_size,
    bool show_progress) {
    auto& model = pinocchio::python::from<pinocchio::Model&>(model_handle);
    auto& data = pinocchio::python::from<pinocchio::Data&>(data_handle);
    return cs179::DirectionalReachEnvelope::build_from_robot(
        model,
        data,
        frame_id,
        n_samples,
        n_theta,
        n_phi,
        batch_size,
        show_progress);
}

py::array_t<double> bin_radii_to_array(const cs179::DirectionalReachEnvelope& envelope) {
    const int n_theta = envelope.n_theta();
    const int n_phi = envelope.n_phi();
    py::array_t<double> arr({static_cast<py::ssize_t>(n_theta), static_cast<py::ssize_t>(n_phi)});
    std::memcpy(
        arr.mutable_data(),
        envelope.bin_radii().data(),
        envelope.bin_radii().size() * sizeof(double));
    return arr;
}

cs179::DirectionalReachEnvelope envelope_from_array(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("bin_radii must be a 2-D array");
    }
    const int n_theta = static_cast<int>(arr.shape(0));
    const int n_phi = static_cast<int>(arr.shape(1));
    std::vector<double> data(
        arr.data(),
        arr.data() + static_cast<py::ssize_t>(n_theta) * static_cast<py::ssize_t>(n_phi));
    return cs179::DirectionalReachEnvelope(std::move(data), n_theta, n_phi);
}

}  // namespace

void bind_reachability(pybind11::module_& m) {
    py::module_::import("pinocchio");

    py::class_<cs179::DirectionalReachEnvelope>(m, "DirectionalReachEnvelope")
        .def(py::init(&envelope_from_array), py::arg("bin_radii"))
        .def_static(
            "from_robot",
            &build_from_robot_py,
            py::arg("model"),
            py::arg("data"),
            py::arg("frame_id"),
            py::arg("n_samples"),
            py::arg("n_theta") = cs179::kReachBinsTheta,
            py::arg("n_phi") = cs179::kReachBinsPhi,
            py::arg("batch_size") = cs179::kReachBuildBatchSize,
            py::arg("show_progress") = true)
        .def_property_readonly(
            "n_theta",
            [](const cs179::DirectionalReachEnvelope& envelope) { return envelope.n_theta(); })
        .def_property_readonly(
            "n_phi",
            [](const cs179::DirectionalReachEnvelope& envelope) { return envelope.n_phi(); })
        .def_property_readonly(
            "bin_radii",
            [](const cs179::DirectionalReachEnvelope& envelope) { return bin_radii_to_array(envelope); })
        .def(
            "max_radius",
            &cs179::DirectionalReachEnvelope::max_radius)
        .def(
            "reach_limits",
            [](const cs179::DirectionalReachEnvelope& envelope, py::array_t<double> directions) {
                if (directions.ndim() == 1) {
                    if (directions.shape(0) != 3) {
                        throw std::invalid_argument("directions must have shape (3,) or (N, 3)");
                    }
                    directions = directions.reshape({1, 3});
                }
                if (directions.ndim() != 2 || directions.shape(1) != 3) {
                    throw std::invalid_argument("directions must have shape (N, 3)");
                }
                const auto n_dirs = static_cast<std::size_t>(directions.shape(0));
                const auto limits = envelope.reach_limits(directions.data(), n_dirs);
                py::array_t<double> out(static_cast<py::ssize_t>(n_dirs));
                std::memcpy(out.mutable_data(), limits.data(), limits.size() * sizeof(double));
                return out;
            },
            py::arg("directions"))
        .def(
            "scale_positions",
            [](const cs179::DirectionalReachEnvelope& envelope,
               py::array_t<double> positions,
               std::optional<py::array_t<double>> pivot,
               double safety) {
                if (positions.ndim() != 2 || positions.shape(1) != 3) {
                    throw std::invalid_argument("positions must have shape (N, 3)");
                }
                double pivot_xyz[3] = {0.0, 0.0, 0.0};
                if (pivot.has_value()) {
                    const auto pivot_arr = *pivot;
                    if (pivot_arr.ndim() != 1 || pivot_arr.shape(0) != 3) {
                        throw std::invalid_argument("pivot must have shape (3,)");
                    }
                    std::memcpy(pivot_xyz, pivot_arr.data(), 3 * sizeof(double));
                }
                const auto n_points = static_cast<std::size_t>(positions.shape(0));
                const auto result = envelope.scale_positions(positions.data(), n_points, pivot_xyz, safety);
                py::array_t<double> scaled({static_cast<py::ssize_t>(n_points), py::ssize_t{3}});
                py::array_t<double> scales(static_cast<py::ssize_t>(n_points));
                std::memcpy(scaled.mutable_data(), result.first.data(), result.first.size() * sizeof(double));
                std::memcpy(scales.mutable_data(), result.second.data(), result.second.size() * sizeof(double));
                return py::make_tuple(scaled, scales);
            },
            py::arg("positions"),
            py::arg("pivot") = std::nullopt,
            py::arg("safety") = 0.9);
}
