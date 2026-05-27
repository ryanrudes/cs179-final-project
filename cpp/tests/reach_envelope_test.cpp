#include "cs179/reach_envelope.hpp"

#include <pinocchio/multibody/sample-models.hpp>

#include <cstdio>
#include <cstdlib>

int main() {
    pinocchio::Model model;
    pinocchio::buildModels::humanoidRandom(model);
    pinocchio::Data data(model);
    const auto frame_id = static_cast<pinocchio::FrameIndex>(model.nframes - 1);

    const auto envelope = cs179::DirectionalReachEnvelope::build_from_robot(
        model,
        data,
        frame_id,
        10'000,
        cs179::kReachBinsTheta,
        cs179::kReachBinsPhi,
        cs179::kReachBuildBatchSize,
        true);

    std::printf(
        "reach envelope: %d x %d bins, max radius %.4f m\n",
        envelope.n_theta(),
        envelope.n_phi(),
        envelope.max_radius());
    return EXIT_SUCCESS;
}
