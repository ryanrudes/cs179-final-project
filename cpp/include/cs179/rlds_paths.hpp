#pragma once

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cs179 {

inline const std::unordered_map<std::string, std::string>& dataset_url_table() {
    static const std::unordered_map<std::string, std::string> table = {
        {"DROID", "gs://gresearch/robotics/droid/1.0.0"},
        {"DROID_100", "gs://gresearch/robotics/droid_100/1.0.0"},
        {"KUKA", "gs://gresearch/robotics/kuka/0.1.0"},
    };
    return table;
}

inline std::string normalize_dataset_url(const std::string& dataset_url) {
    const auto& table = dataset_url_table();
    if (const auto it = table.find(dataset_url); it != table.end()) {
        return it->second;
    }
    std::string upper = dataset_url;
    for (auto& ch : upper) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    if (const auto it = table.find(upper); it != table.end()) {
        return it->second;
    }
    return dataset_url;
}

inline std::string dataset_name_from_url(const std::string& dataset_url) {
    std::string trimmed = dataset_url;
    while (!trimmed.empty() && trimmed.back() == '/') {
        trimmed.pop_back();
    }
    const auto slash = trimmed.find_last_of('/');
    if (slash == std::string::npos || slash == 0) {
        throw std::runtime_error("Cannot parse dataset name from URL: " + dataset_url);
    }
    const auto prev = trimmed.find_last_of('/', slash - 1);
    if (prev == std::string::npos) {
        throw std::runtime_error("Cannot parse dataset name from URL: " + dataset_url);
    }
    return trimmed.substr(prev + 1, slash - prev - 1);
}

inline std::filesystem::path metadata_dir(const std::filesystem::path& data_dir) {
    return data_dir / "metadata";
}

inline std::filesystem::path observation_shard_path(
    const std::filesystem::path& data_dir,
    const std::string& observation_key,
    int shard_id) {
    char shard_name[16];
    std::snprintf(shard_name, sizeof(shard_name), "%05d.npy", shard_id);
    return data_dir / observation_key / shard_name;
}

inline std::filesystem::path resolve_data_dir(
    const std::filesystem::path& data_dir,
    const std::string& dataset_url) {
    const std::string normalized = normalize_dataset_url(dataset_url);
    const std::string dataset_name = dataset_name_from_url(normalized);
    if (data_dir.filename() == dataset_name) {
        return data_dir;
    }
    return data_dir / dataset_name;
}

}  // namespace cs179
