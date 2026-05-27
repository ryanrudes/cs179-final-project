#include "cs179/reach_envelope.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cs179 {
namespace {

constexpr double kEmptyBinThreshold = 1e-6;
constexpr double kDirectionEps = 1e-9;

inline int bin_index_theta(double theta, int n_theta) {
    const int ti = static_cast<int>(theta / M_PI * static_cast<double>(n_theta));
    return std::min(ti, n_theta - 1);
}

inline int bin_index_phi(double phi, int n_phi) {
    int pj = static_cast<int>((phi + M_PI) / (2.0 * M_PI) * static_cast<double>(n_phi));
    pj %= n_phi;
    if (pj < 0) {
        pj += n_phi;
    }
    return pj;
}

inline void accumulate_position(
    double px,
    double py,
    double pz,
    std::vector<double>& bin_radii,
    int n_theta,
    int n_phi) {
    const double radius = std::sqrt(px * px + py * py + pz * pz);
    if (radius < kEmptyBinThreshold) {
        return;
    }

    const double cos_theta = std::clamp(pz / radius, -1.0, 1.0);
    const double theta = std::acos(cos_theta);
    const double phi = std::atan2(py, px);
    const int ti = bin_index_theta(theta, n_theta);
    const int pj = bin_index_phi(phi, n_phi);
    const std::size_t idx = static_cast<std::size_t>(ti) * static_cast<std::size_t>(n_phi) + static_cast<std::size_t>(pj);
    bin_radii[idx] = std::max(bin_radii[idx], radius);
}

void fill_empty_bins(std::vector<double>& bin_radii, int n_theta, int n_phi) {
    const auto at = [&](int i, int j) -> double& {
        return bin_radii[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_phi) + static_cast<std::size_t>(j)];
    };

    std::vector<double> spread(bin_radii.size(), 0.0);
    for (int i = 0; i < n_theta; ++i) {
        for (int j = 0; j < n_phi; ++j) {
            double neighborhood_max = 0.0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = std::clamp(i + di, 0, n_theta - 1);
                    const int nj = std::clamp(j + dj, 0, n_phi - 1);
                    neighborhood_max = std::max(neighborhood_max, at(ni, nj));
                }
            }
            spread[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_phi) + static_cast<std::size_t>(j)] =
                neighborhood_max;
        }
    }

    double global_max = 0.0;
    for (double value : bin_radii) {
        global_max = std::max(global_max, value);
    }

    for (std::size_t idx = 0; idx < bin_radii.size(); ++idx) {
        if (bin_radii[idx] < kEmptyBinThreshold) {
            bin_radii[idx] = spread[idx];
        }
        if (bin_radii[idx] < kEmptyBinThreshold) {
            bin_radii[idx] = global_max;
        }
    }
}

Eigen::VectorXd sample_random_configuration(
    const pinocchio::Model& model,
    std::mt19937& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    Eigen::VectorXd q(model.nq);
    for (Eigen::Index i = 0; i < model.nq; ++i) {
        const double lo = model.lowerPositionLimit[i];
        const double hi = model.upperPositionLimit[i];
        q[i] = lo + (hi - lo) * uniform(rng);
    }
    return q;
}

int default_thread_count() {
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    return max_threads > 0 ? max_threads : 1;
#else
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? static_cast<int>(hc) : 1;
#endif
}

}  // namespace

DirectionalReachEnvelope::DirectionalReachEnvelope(
    std::vector<double> bin_radii,
    int n_theta,
    int n_phi)
    : n_theta_(n_theta),
      n_phi_(n_phi),
      bin_radii_(std::move(bin_radii)) {
    if (n_theta_ <= 0 || n_phi_ <= 0) {
        throw std::invalid_argument("DirectionalReachEnvelope requires positive bin counts");
    }
    if (static_cast<std::size_t>(n_theta_) * static_cast<std::size_t>(n_phi_) != bin_radii_.size()) {
        throw std::invalid_argument("bin_radii size does not match n_theta * n_phi");
    }
}

DirectionalReachEnvelope DirectionalReachEnvelope::build_from_robot(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const pinocchio::FrameIndex frame_id,
    const std::size_t n_samples,
    const int n_theta,
    const int n_phi,
    const int batch_size,
    const bool show_progress) {
    if (frame_id >= static_cast<pinocchio::FrameIndex>(model.nframes)) {
        throw std::out_of_range("frame_id out of range for model");
    }
    if (n_samples == 0) {
        return DirectionalReachEnvelope(std::vector<double>(static_cast<std::size_t>(n_theta * n_phi), 0.0), n_theta, n_phi);
    }

    const int effective_batch = std::max(1, std::min(batch_size, static_cast<int>(n_samples)));
    const int n_threads = default_thread_count();

    std::vector<double> bin_radii(static_cast<std::size_t>(n_theta * n_phi), 0.0);
    std::vector<std::vector<double>> thread_bins(
        static_cast<std::size_t>(n_threads),
        std::vector<double>(static_cast<std::size_t>(n_theta * n_phi), 0.0));

    std::size_t done = 0;
    std::size_t batch_idx = 0;
    const std::size_t report_every = std::max<std::size_t>(1, n_samples / 20 / static_cast<std::size_t>(effective_batch));
    const auto t0 = std::chrono::steady_clock::now();

    while (done < n_samples) {
        const std::size_t n_batch = std::min(static_cast<std::size_t>(effective_batch), n_samples - done);

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
        {
            const int tid = omp_get_thread_num();
            std::mt19937 rng(static_cast<std::uint32_t>(done + static_cast<std::size_t>(tid) * 1'046'729u + 42u));
            pinocchio::Data local_data(model);

#pragma omp for schedule(static)
            for (std::ptrdiff_t sample = 0; sample < static_cast<std::ptrdiff_t>(n_batch); ++sample) {
                const Eigen::VectorXd q = sample_random_configuration(model, rng);
                pinocchio::forwardKinematics(model, local_data, q);
                pinocchio::updateFramePlacements(model, local_data);
                const Eigen::Vector3d& t = local_data.oMf[frame_id].translation();
                accumulate_position(t.x(), t.y(), t.z(), thread_bins[static_cast<std::size_t>(tid)], n_theta, n_phi);
            }
        }
#else
        std::mt19937 rng(static_cast<std::uint32_t>(done + 42u));
        pinocchio::Data local_data(model);
        for (std::size_t sample = 0; sample < n_batch; ++sample) {
            const Eigen::VectorXd q = sample_random_configuration(model, rng);
            pinocchio::forwardKinematics(model, local_data, q);
            pinocchio::updateFramePlacements(model, local_data);
            const Eigen::Vector3d& t = local_data.oMf[frame_id].translation();
            accumulate_position(t.x(), t.y(), t.z(), thread_bins[0], n_theta, n_phi);
        }
#endif

        for (auto& local : thread_bins) {
            for (std::size_t idx = 0; idx < bin_radii.size(); ++idx) {
                bin_radii[idx] = std::max(bin_radii[idx], local[idx]);
            }
            std::fill(local.begin(), local.end(), 0.0);
        }

        done += n_batch;
        ++batch_idx;

        if (show_progress && batch_idx % report_every == 0) {
            const auto elapsed = std::chrono::steady_clock::now() - t0;
            const double seconds =
                std::chrono::duration<double>(elapsed).count();
            const double rate = static_cast<double>(done) / std::max(seconds, 1e-9);
            const double eta = static_cast<double>(n_samples - done) / std::max(rate, 1e-9);
            std::fprintf(
                stderr,
                "  reach envelope: %zu/%zu samples (%.0f/s, ETA %.0fs)\n",
                done,
                n_samples,
                rate,
                eta);
        }
    }

    fill_empty_bins(bin_radii, n_theta, n_phi);
    return DirectionalReachEnvelope(std::move(bin_radii), n_theta, n_phi);
}

double DirectionalReachEnvelope::max_radius() const {
    return *std::max_element(bin_radii_.begin(), bin_radii_.end());
}

double DirectionalReachEnvelope::fallback_radius() const {
    return max_radius();
}

double DirectionalReachEnvelope::interpolate(const double theta, const double phi) const {
    const double fi = theta / M_PI * static_cast<double>(n_theta_) - 0.5;
    const double fj = (phi + M_PI) / (2.0 * M_PI) * static_cast<double>(n_phi_) - 0.5;

    if (fi < 0.0 || fi > static_cast<double>(n_theta_ - 1) || fj < 0.0 || fj > static_cast<double>(n_phi_ - 1)) {
        return fallback_radius();
    }

    const int i0 = static_cast<int>(std::floor(fi));
    const int j0 = static_cast<int>(std::floor(fj));
    const int i1 = std::min(i0 + 1, n_theta_ - 1);
    const int j1 = std::min(j0 + 1, n_phi_ - 1);
    const double di = fi - static_cast<double>(i0);
    const double dj = fj - static_cast<double>(j0);

    const auto at = [&](int i, int j) -> double {
        return bin_radii_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_phi_) + static_cast<std::size_t>(j)];
    };

    const double v00 = at(i0, j0);
    const double v01 = at(i0, j1);
    const double v10 = at(i1, j0);
    const double v11 = at(i1, j1);
    const double v0 = v00 * (1.0 - dj) + v01 * dj;
    const double v1 = v10 * (1.0 - dj) + v11 * dj;
    return v0 * (1.0 - di) + v1 * di;
}

std::vector<double> DirectionalReachEnvelope::reach_limits(
    const double* directions,
    const std::size_t n_dirs) const {
    std::vector<double> limits(n_dirs, fallback_radius());
    for (std::size_t row = 0; row < n_dirs; ++row) {
        const double* d = directions + row * 3;
        const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (norm < kDirectionEps) {
            continue;
        }
        const double inv = 1.0 / norm;
        const double z = std::clamp(d[2] * inv, -1.0, 1.0);
        const double theta = std::acos(z);
        const double phi = std::atan2(d[1] * inv, d[0] * inv);
        limits[row] = interpolate(theta, phi);
    }
    return limits;
}

std::pair<std::vector<double>, std::vector<double>> DirectionalReachEnvelope::scale_positions(
    const double* positions,
    const std::size_t n_points,
    const double pivot[3],
    const double safety) const {
    std::vector<double> scaled(positions, positions + n_points * 3);
    std::vector<double> scales(n_points, 1.0);

    std::vector<double> directions;
    directions.reserve(n_points * 3);
    std::vector<std::size_t> valid_rows;
    valid_rows.reserve(n_points);

    for (std::size_t row = 0; row < n_points; ++row) {
        const double ox = positions[row * 3 + 0] - pivot[0];
        const double oy = positions[row * 3 + 1] - pivot[1];
        const double oz = positions[row * 3 + 2] - pivot[2];
        const double radius = std::sqrt(ox * ox + oy * oy + oz * oz);
        if (radius <= kDirectionEps) {
            continue;
        }
        valid_rows.push_back(row);
        directions.push_back(ox / radius);
        directions.push_back(oy / radius);
        directions.push_back(oz / radius);
    }

    const auto limits = reach_limits(directions.data(), valid_rows.size());
    for (std::size_t k = 0; k < valid_rows.size(); ++k) {
        const std::size_t row = valid_rows[k];
        const double ox = positions[row * 3 + 0] - pivot[0];
        const double oy = positions[row * 3 + 1] - pivot[1];
        const double oz = positions[row * 3 + 2] - pivot[2];
        const double radius = std::sqrt(ox * ox + oy * oy + oz * oz);
        const double limit = limits[k] * safety;
        const double scale = std::min(1.0, limit / radius);
        scales[row] = scale;
        scaled[row * 3 + 0] = pivot[0] + scale * ox;
        scaled[row * 3 + 1] = pivot[1] + scale * oy;
        scaled[row * 3 + 2] = pivot[2] + scale * oz;
    }

    return {std::move(scaled), std::move(scales)};
}

}  // namespace cs179
