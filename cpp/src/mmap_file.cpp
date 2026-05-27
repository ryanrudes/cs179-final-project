#include "cs179/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>
#include <string>

namespace cs179 {

MmapFile::MmapFile(const std::filesystem::path& path) {
    const std::string native = path.string();
    fd_ = ::open(native.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open file for mmap: " + native);
    }

    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        release();
        throw std::runtime_error("Failed to stat file for mmap: " + native);
    }

    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ == 0) {
        return;
    }

    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        release();
        throw std::runtime_error("mmap failed for: " + native);
    }

    data_ = static_cast<const std::byte*>(mapped);
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : data_(other.data_), size_(other.size_), fd_(other.fd_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        size_ = other.size_;
        fd_ = other.fd_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

MmapFile::~MmapFile() { release(); }

void MmapFile::release() {
    if (data_ != nullptr && size_ > 0) {
        ::munmap(const_cast<std::byte*>(data_), size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
}

}  // namespace cs179
