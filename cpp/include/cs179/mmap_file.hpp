#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace cs179 {

/// Read-only memory map of a file (POSIX ``mmap``).
class MmapFile {
public:
    MmapFile() = default;
    explicit MmapFile(const std::filesystem::path& path);

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    ~MmapFile();

    const std::byte* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    explicit operator bool() const { return data_ != nullptr; }

private:
    void release();

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
    int fd_ = -1;
};

}  // namespace cs179
