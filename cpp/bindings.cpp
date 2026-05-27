#include "bindings_detail.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
#ifdef CS179_HAVE_CUDA
    m.doc() = "Native extension for cs179 (RLDS cache loader + CUDA kernels)";
#else
    m.doc() = "Native extension for cs179 (RLDS cache loader + reach envelope; CUDA not built)";
#endif

    bind_rlds(m);
#ifdef CS179_HAVE_REACHABILITY
    bind_reachability(m);
    bind_retarget(m);
#endif
#ifdef CS179_HAVE_CUDA
    bind_cuda(m);
#endif
}
