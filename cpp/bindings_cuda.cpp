#include "bindings_detail.hpp"
#include "kernels.cuh"

#include <pybind11/numpy.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

void check_contiguous_1d(const py::array& array, const char* name) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
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
}
