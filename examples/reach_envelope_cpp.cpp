// Directional reach envelope — C++ API walkthrough.
//
// Build with:
//   cmake -B build -DCS179_BUILD_EXAMPLES=ON -DCS179_BUILD_PYTHON=ON
//   cmake --build build --target reach_envelope_cpp
//
// Run:
//   ./build/reach_envelope_cpp
//
// For URDF-based robots, build the envelope from Python (examples/reach_envelope_python.py)
// or link urdfdom and use pinocchio::urdf::buildModel in your own binary.
//
// See REACH_ENVELOPE.md and examples/README.md.

#include "cs179/reach_envelope.hpp"

#include <pinocchio/multibody/sample-models.hpp>

#include <cstdio>
#include <cstdlib>

namespace {

void print_limits(const cs179::DirectionalReachEnvelope& envelope, const double* dir) {
    const auto limits = envelope.reach_limits(dir, 1);
    std::printf(
        "  limit([%.2f, %.2f, %.2f]) ≈ %.3f m\n",
        dir[0],
        dir[1],
        dir[2],
        limits[0]);
}

}  // namespace

int main() {
    pinocchio::Model model;
    pinocchio::buildModels::humanoidRandom(model);
    pinocchio::Data data(model);
    const auto frame_id = static_cast<pinocchio::FrameIndex>(model.nframes - 1);
    const std::string& frame_name = model.frames[static_cast<std::size_t>(frame_id)].name;
    std::printf("Model: humanoidRandom, tool frame: %s\n", frame_name.c_str());

    // Monte Carlo FK envelope (use more samples in production; see REACH_ENVELOPE.md).
    constexpr std::size_t n_samples = 50'000;
    constexpr int n_theta = 24;
    constexpr int n_phi = 48;
    constexpr double safety = 0.9;

    const auto envelope = cs179::DirectionalReachEnvelope::build_from_robot(
        model,
        data,
        frame_id,
        n_samples,
        n_theta,
        n_phi,
        cs179::kReachBuildBatchSize,
        true);

    std::printf(
        "Envelope: %d×%d bins, max radius %.3f m\n",
        envelope.n_theta(),
        envelope.n_phi(),
        envelope.max_radius());

    const double dirs[] = {
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.3,
        0.4,
        0.5,
    };
    std::printf("reach_limits:\n");
    print_limits(envelope, dirs + 0);
    print_limits(envelope, dirs + 3);
    print_limits(envelope, dirs + 6);

    const double positions[] = {
        0.45,
        0.0,
        0.25,
        0.55,
        0.1,
        0.15,
        0.20,
        -0.35,
        0.40,
    };
    const double pivot[3] = {0.0, 0.0, 0.0};
    const auto [scaled, scales] =
        envelope.scale_positions(positions, 3, pivot, safety);

    std::printf("scale_positions (safety=%.2f):\n", safety);
    for (std::size_t row = 0; row < 3; ++row) {
        const double* orig = positions + row * 3;
        const double* out = scaled.data() + row * 3;
        std::printf(
            "  [%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f]  (scale=%.3f)\n",
            orig[0],
            orig[1],
            orig[2],
            out[0],
            out[1],
            out[2],
            scales[row]);
    }

    std::printf(
        "\nNPZ cache I/O is Python-only; use DirectionalReachEnvelope.load/save in Python "
        "or pass bin_radii() into the C++ constructor.\n");
    return EXIT_SUCCESS;
}
