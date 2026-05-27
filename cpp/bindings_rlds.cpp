#include "bindings_detail.hpp"
#include "cs179/rlds_loader.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

py::array_t<float> array_from_demo_field(
    const std::vector<std::size_t>& shape,
    const std::vector<float>& data) {
    py::array_t<float> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(float));
    return arr;
}

py::array_t<float> array_from_view(
    const std::vector<std::size_t>& shape,
    const float* data,
    const std::shared_ptr<const cs179::NpyArray>& owner) {
    py::capsule owner_capsule(
        new std::shared_ptr<const cs179::NpyArray>(owner),
        [](void* ptr) { delete static_cast<std::shared_ptr<const cs179::NpyArray>*>(ptr); });
    return py::array_t<float>(shape, data, owner_capsule);
}

py::dict demo_arrays_to_dict(const cs179::RldsObservationLoader::DemoArrays& demo) {
    py::dict out;
    for (const auto& [key, values] : demo.fields) {
        out[py::str(key)] = array_from_demo_field(demo.shapes.at(key), values);
    }
    return out;
}

py::dict demo_views_to_dict(const cs179::RldsObservationLoader::DemoViews& views) {
    py::dict out;
    for (const auto& [key, data_ptr] : views.data) {
        const auto owner_it = views.shards.find(key);
        if (owner_it == views.shards.end()) {
            throw std::runtime_error("Missing shard owner for demo view");
        }
        out[py::str(key)] =
            array_from_view(views.shapes.at(key), data_ptr, owner_it->second);
    }
    return out;
}

}  // namespace

void bind_rlds(pybind11::module_& m) {
    py::class_<cs179::RldsObservationLoader, std::shared_ptr<cs179::RldsObservationLoader>>(
        m,
        "RldsObservationLoader")
        .def(
            py::init([](const std::string& data_dir, const std::optional<std::string>& dataset_url) {
                return std::make_shared<cs179::RldsObservationLoader>(
                    std::filesystem::path(data_dir),
                    dataset_url);
            }),
            py::arg("data_dir") = "data",
            py::arg("dataset_url") = std::nullopt)
        .def("__len__", &cs179::RldsObservationLoader::num_demos)
        .def_property_readonly("total_steps", &cs179::RldsObservationLoader::total_steps)
        .def_property_readonly("control_hz", &cs179::RldsObservationLoader::control_hz)
        .def_property_readonly(
            "data_dir",
            [](const cs179::RldsObservationLoader& loader) {
                return loader.data_dir().string();
            })
        .def_property_readonly(
            "dataset_url",
            [](const cs179::RldsObservationLoader& loader) { return loader.dataset_url(); })
        .def_property_readonly(
            "observation_keys",
            &cs179::RldsObservationLoader::observation_keys)
        .def_property_readonly(
            "field_shapes",
            [](const cs179::RldsObservationLoader& loader) {
                py::dict out;
                for (const auto& [key, shape] : loader.field_shapes()) {
                    out[py::str(key)] = shape;
                }
                return out;
            })
        .def(
            "get_demo",
            [](const std::shared_ptr<cs179::RldsObservationLoader>& loader, std::ptrdiff_t demo_id) {
                return demo_arrays_to_dict(loader->get_demo(demo_id));
            },
            py::arg("demo_id"))
        .def(
            "get_step_range",
            [](const std::shared_ptr<cs179::RldsObservationLoader>& loader,
               std::size_t start,
               std::size_t end) { return demo_arrays_to_dict(loader->get_step_range(start, end)); },
            py::arg("start"),
            py::arg("end"))
        .def(
            "get_demo_views",
            [](const std::shared_ptr<cs179::RldsObservationLoader>& loader, std::ptrdiff_t demo_id)
                -> py::object {
                const auto views = loader->get_demo_views(demo_id);
                if (!views.has_value()) {
                    return py::none();
                }
                return demo_views_to_dict(*views);
            },
            py::arg("demo_id"));
}
