#pragma once

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace cs179 {

constexpr int kReachBinsTheta = 64;
constexpr int kReachBinsPhi = 128;
constexpr int kReachBuildBatchSize = 50'000;

/// Max tool reach per spherical direction (theta, phi) from the robot base.
class DirectionalReachEnvelope {
public:
    explicit DirectionalReachEnvelope(std::vector<double> bin_radii, int n_theta, int n_phi);

    static DirectionalReachEnvelope build_from_robot(
        const pinocchio::Model& model,
        pinocchio::Data& data,
        pinocchio::FrameIndex frame_id,
        std::size_t n_samples,
        int n_theta = kReachBinsTheta,
        int n_phi = kReachBinsPhi,
        int batch_size = kReachBuildBatchSize,
        bool show_progress = true);

    [[nodiscard]] int n_theta() const { return n_theta_; }
    [[nodiscard]] int n_phi() const { return n_phi_; }
    [[nodiscard]] const std::vector<double>& bin_radii() const { return bin_radii_; }

    [[nodiscard]] double max_radius() const;

    /// Unit directions (n, 3) row-major -> reach limit (n,).
    std::vector<double> reach_limits(const double* directions, std::size_t n_dirs) const;

    /// Positions (n, 3) row-major; returns scaled positions and per-row scale factors.
    std::pair<std::vector<double>, std::vector<double>> scale_positions(
        const double* positions,
        std::size_t n_points,
        const double pivot[3],
        double safety) const;

private:
    [[nodiscard]] double interpolate(double theta, double phi) const;
    [[nodiscard]] double fallback_radius() const;

    int n_theta_;
    int n_phi_;
    std::vector<double> bin_radii_;
};

}  // namespace cs179
