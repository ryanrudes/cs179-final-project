#pragma once

#include "cs179/npy.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cs179 {

/// C++ counterpart of ``rlds.RldsObservationLoader`` (mmap-backed float32 shards).
class RldsObservationLoader {
public:
    RldsObservationLoader(
        const std::filesystem::path& data_dir = "data",
        const std::optional<std::string>& dataset_url = std::nullopt);

    std::size_t num_demos() const { return demo_lengths_.size(); }
    std::int64_t total_steps() const { return total_steps_; }
    double control_hz() const { return control_hz_; }
    const std::filesystem::path& data_dir() const { return data_dir_; }
    const std::vector<std::string>& observation_keys() const { return observation_keys_; }
    const std::unordered_map<std::string, std::vector<std::size_t>>& field_shapes() const {
        return field_shapes_;
    }
    const std::string& dataset_url() const { return dataset_url_; }

    std::size_t demo_length(std::size_t demo_id) const;

    struct DemoArrays {
        std::unordered_map<std::string, std::vector<float>> fields;
        std::unordered_map<std::string, std::vector<std::size_t>> shapes;
    };

    DemoArrays get_demo(std::ptrdiff_t demo_id) const;
    DemoArrays get_step_range(std::size_t start, std::size_t end) const;

    struct DemoViews {
        std::unordered_map<std::string, const float*> data;
        std::unordered_map<std::string, std::vector<std::size_t>> shapes;
        std::unordered_map<std::string, std::shared_ptr<const NpyArray>> shards;
    };

    std::optional<DemoViews> get_demo_views(std::ptrdiff_t demo_id) const;

private:
    std::ptrdiff_t normalize_demo_id(std::ptrdiff_t demo_id) const;
    const NpyArray& open_shard(const std::string& observation_key, int shard_id) const;

    std::filesystem::path data_dir_;
    std::filesystem::path metadata_dir_;

    std::vector<std::int32_t> demo_lengths_;
    std::vector<std::int64_t> demo_offsets_;
    std::vector<std::int64_t> shard_lengths_;
    std::vector<std::int64_t> shard_offsets_;
    std::int64_t total_steps_ = 0;

    std::vector<std::string> observation_keys_;
    std::unordered_map<std::string, std::vector<std::size_t>> field_shapes_;
    std::string dataset_url_;
    double control_hz_ = 0.0;

    mutable std::unordered_map<std::string, std::unordered_map<int, NpyArray>> shard_cache_;
};

}  // namespace cs179
