#pragma once

#include "cs179/mmap_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cs179 {

enum class NpyDtype {
    Float32,
    Int32,
    Int64,
};

struct NpyArray {
    std::shared_ptr<const MmapFile> file;
    const void* data = nullptr;
    NpyDtype dtype = NpyDtype::Float32;
    std::vector<std::size_t> shape;
    std::size_t element_size = 0;

    std::size_t size() const;
    std::size_t ndim() const { return shape.size(); }
};

NpyArray load_npy_mmap(const std::filesystem::path& path);

std::vector<std::int64_t> load_npy_int64_vector(const std::filesystem::path& path);
std::vector<std::int32_t> load_npy_int32_vector(const std::filesystem::path& path);
std::int64_t load_npy_scalar_int64(const std::filesystem::path& path);

}  // namespace cs179
