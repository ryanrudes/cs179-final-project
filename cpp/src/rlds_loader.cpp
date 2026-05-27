#include "cs179/rlds_loader.hpp"

#include "cs179/rlds_paths.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace cs179 {
namespace {

std::size_t field_stride(const std::vector<std::size_t>& field_shape) {
    std::size_t stride = 1;
    for (const auto dim : field_shape) {
        stride *= dim;
    }
    return stride;
}

void copy_shard_slice(
    float* dst,
    const float* shard,
    std::size_t field_stride,
    std::size_t local_start,
    std::size_t local_end) {
    const std::size_t count = local_end - local_start;
    std::memcpy(dst, shard + local_start * field_stride, count * field_stride * sizeof(float));
}

}  // namespace

RldsObservationLoader::RldsObservationLoader(
    const std::filesystem::path& data_dir,
    const std::optional<std::string>& dataset_url) {
    const auto direct_meta = metadata_dir(data_dir) / "metadata.json";
    if (std::filesystem::is_regular_file(direct_meta)) {
        data_dir_ = data_dir;
    } else {
        const std::string resolved_url =
            dataset_url.value_or("gs://gresearch/robotics/droid_100/1.0.0");
        data_dir_ = resolve_data_dir(data_dir, resolved_url);
    }

    metadata_dir_ = metadata_dir(data_dir_);
    const auto meta_path = metadata_dir_ / "metadata.json";
    if (!std::filesystem::is_regular_file(meta_path)) {
        throw std::runtime_error(
            "No cache metadata at " + meta_path.string() +
            ". Run `uv run cs179 download` for this dataset first.");
    }

    demo_lengths_ = load_npy_int32_vector(metadata_dir_ / "demo_lengths.npy");
    demo_offsets_ = load_npy_int64_vector(metadata_dir_ / "demo_offsets.npy");
    shard_lengths_ = load_npy_int64_vector(metadata_dir_ / "shard_lengths.npy");
    shard_offsets_ = load_npy_int64_vector(metadata_dir_ / "shard_offsets.npy");
    total_steps_ = load_npy_scalar_int64(metadata_dir_ / "total_steps.npy");

    std::ifstream meta_in(meta_path);
    if (!meta_in) {
        throw std::runtime_error("Failed to open metadata file: " + meta_path.string());
    }

    const auto metadata = nlohmann::json::parse(meta_in);
    observation_keys_ = metadata.at("observation_keys").get<std::vector<std::string>>();
    dataset_url_ = metadata.at("dataset_url").get<std::string>();
    control_hz_ = metadata.at("control_hz").get<double>();
    if (control_hz_ <= 0.0) {
        throw std::runtime_error("Invalid control_hz in metadata");
    }

    field_shapes_.clear();
    const auto& shapes = metadata.at("field_shapes");
    for (const auto& key : observation_keys_) {
        std::vector<std::size_t> shape;
        for (const auto& dim : shapes.at(key)) {
            shape.push_back(dim.get<std::size_t>());
        }
        field_shapes_[key] = std::move(shape);
    }
}

std::size_t RldsObservationLoader::demo_length(std::size_t demo_id) const {
    return static_cast<std::size_t>(demo_lengths_.at(demo_id));
}

std::ptrdiff_t RldsObservationLoader::normalize_demo_id(std::ptrdiff_t demo_id) const {
    if (demo_id < 0) {
        demo_id += static_cast<std::ptrdiff_t>(num_demos());
    }
    if (demo_id < 0 || static_cast<std::size_t>(demo_id) >= num_demos()) {
        throw std::out_of_range("demo_id out of range");
    }
    return demo_id;
}

const NpyArray& RldsObservationLoader::open_shard(const std::string& observation_key, int shard_id) const {
    auto& shards = shard_cache_[observation_key];
    if (const auto it = shards.find(shard_id); it != shards.end()) {
        return it->second;
    }
    const auto inserted = shards.emplace(
        shard_id,
        load_npy_mmap(observation_shard_path(data_dir_, observation_key, shard_id)));
    return inserted.first->second;
}

RldsObservationLoader::DemoArrays RldsObservationLoader::get_step_range(
    std::size_t start,
    std::size_t end) const {
    if (start > end || end > static_cast<std::size_t>(total_steps_)) {
        throw std::out_of_range("Invalid step range");
    }

    DemoArrays result;
    if (start == end) {
        for (const auto& key : observation_keys_) {
            const auto& shape = field_shapes_.at(key);
            result.shapes[key] = std::vector<std::size_t>{0};
            result.shapes[key].insert(result.shapes[key].end(), shape.begin(), shape.end());
            result.fields[key] = {};
        }
        return result;
    }

    const std::size_t length = end - start;
    for (const auto& key : observation_keys_) {
        const auto& field_shape = field_shapes_.at(key);
        const std::size_t stride = field_stride(field_shape);
        result.shapes[key] = std::vector<std::size_t>{length};
        result.shapes[key].insert(result.shapes[key].end(), field_shape.begin(), field_shape.end());
        result.fields[key].assign(length * stride, 0.0f);
    }

    std::size_t out_offset = 0;
    auto shard_it = std::upper_bound(shard_offsets_.begin(), shard_offsets_.end(), static_cast<std::int64_t>(start));
    std::size_t shard_id = static_cast<std::size_t>(std::distance(shard_offsets_.begin(), shard_it)) - 1;
    std::size_t cursor = start;

    while (cursor < end) {
        const std::size_t shard_start = static_cast<std::size_t>(shard_offsets_.at(shard_id));
        const std::size_t shard_end = static_cast<std::size_t>(shard_offsets_.at(shard_id + 1));
        const std::size_t take_end = std::min(end, shard_end);

        const std::size_t local_start = cursor - shard_start;
        const std::size_t local_end = take_end - shard_start;
        const std::size_t take = take_end - cursor;

        for (const auto& key : observation_keys_) {
            const auto& shard = open_shard(key, static_cast<int>(shard_id));
            const auto stride = field_stride(field_shapes_.at(key));
            const auto* shard_f = static_cast<const float*>(shard.data);
            copy_shard_slice(
                result.fields[key].data() + out_offset * stride,
                shard_f,
                stride,
                local_start,
                local_end);
        }

        cursor = take_end;
        out_offset += take;
        ++shard_id;
    }

    return result;
}

RldsObservationLoader::DemoArrays RldsObservationLoader::get_demo(std::ptrdiff_t demo_id) const {
    const auto normalized = static_cast<std::size_t>(normalize_demo_id(demo_id));
    const std::size_t start = static_cast<std::size_t>(demo_offsets_.at(normalized));
    const std::size_t end = static_cast<std::size_t>(demo_offsets_.at(normalized + 1));
    return get_step_range(start, end);
}

std::optional<RldsObservationLoader::DemoViews> RldsObservationLoader::get_demo_views(
    std::ptrdiff_t demo_id) const {
    const auto normalized = static_cast<std::size_t>(normalize_demo_id(demo_id));
    const std::size_t start = static_cast<std::size_t>(demo_offsets_.at(normalized));
    const std::size_t end = static_cast<std::size_t>(demo_offsets_.at(normalized + 1));

    auto shard_it = std::upper_bound(shard_offsets_.begin(), shard_offsets_.end(), static_cast<std::int64_t>(start));
    const std::size_t shard_id = static_cast<std::size_t>(std::distance(shard_offsets_.begin(), shard_it)) - 1;

    if (end > static_cast<std::size_t>(shard_offsets_.at(shard_id + 1))) {
        return std::nullopt;
    }

    const std::size_t shard_start = static_cast<std::size_t>(shard_offsets_.at(shard_id));
    const std::size_t local_start = start - shard_start;
    const std::size_t local_end = end - shard_start;
    const std::size_t length = local_end - local_start;

    DemoViews views;
    for (const auto& key : observation_keys_) {
        const auto& shard = open_shard(key, static_cast<int>(shard_id));
        views.shards[key] = std::make_shared<NpyArray>(shard);
        const auto stride = field_stride(field_shapes_.at(key));
        const auto* shard_f = static_cast<const float*>(shard.data);

        views.shapes[key] = std::vector<std::size_t>{length};
        views.shapes[key].insert(
            views.shapes[key].end(),
            field_shapes_.at(key).begin(),
            field_shapes_.at(key).end());
        views.data[key] = shard_f + local_start * stride;
    }

    return views;
}

}  // namespace cs179
