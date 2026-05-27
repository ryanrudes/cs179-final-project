#include "cs179/npy.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cs179 {
namespace {

constexpr unsigned char kNpyMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

std::string read_file_string(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open npy file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void expect_magic(const std::byte* bytes, std::size_t size) {
    if (size < 10 || std::memcmp(bytes, kNpyMagic, sizeof(kNpyMagic)) != 0) {
        throw std::runtime_error("Invalid npy magic header");
    }
}

struct NpyHeaderInfo {
    NpyDtype dtype;
    std::vector<std::size_t> shape;
    std::size_t header_end = 0;
};

std::size_t parse_shape_component(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == ',')) {
        ++pos;
    }
    std::size_t end = pos;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) {
        ++end;
    }
    if (end == pos) {
        throw std::runtime_error("Malformed npy shape tuple");
    }
    std::size_t value = 0;
    const auto result = std::from_chars(text.data() + pos, text.data() + end, value);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("Failed to parse npy shape component");
    }
    pos = end;
    return value;
}

std::vector<std::size_t> parse_shape_tuple(const std::string& header, std::size_t shape_pos) {
    const std::size_t open = header.find('(', shape_pos);
    const std::size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("Malformed npy shape field");
    }

    const std::string tuple = header.substr(open, close - open + 1);
    std::vector<std::size_t> shape;
    std::size_t pos = 1;
    while (pos < tuple.size()) {
        while (pos < tuple.size() && (tuple[pos] == ' ' || tuple[pos] == ',')) {
            ++pos;
        }
        if (pos >= tuple.size() || tuple[pos] == ')') {
            break;
        }
        shape.push_back(parse_shape_component(tuple, pos));
    }
    return shape;
}

NpyDtype parse_descr(const std::string& header) {
    const std::size_t descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) {
        throw std::runtime_error("Missing npy descr field");
    }
    if (header.find("'<f4'", descr_pos) != std::string::npos ||
        header.find("\"<f4\"", descr_pos) != std::string::npos) {
        return NpyDtype::Float32;
    }
    if (header.find("'<i8'", descr_pos) != std::string::npos ||
        header.find("\"<i8\"", descr_pos) != std::string::npos) {
        return NpyDtype::Int64;
    }
    if (header.find("'<i4'", descr_pos) != std::string::npos ||
        header.find("\"<i4\"", descr_pos) != std::string::npos) {
        return NpyDtype::Int32;
    }
    throw std::runtime_error("Unsupported npy dtype (expected <f4, <i4, or <i8)");
}

void expect_c_order(const std::string& header) {
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": false") == std::string::npos) {
        throw std::runtime_error("Fortran-order npy arrays are not supported");
    }
}

NpyHeaderInfo parse_npy_header(const std::byte* bytes, std::size_t file_size) {
    expect_magic(bytes, file_size);

    const unsigned char major = static_cast<unsigned char>(bytes[6]);
    const unsigned char minor = static_cast<unsigned char>(bytes[7]);

    std::size_t header_len = 0;
    std::size_t header_start = 0;

    if (major == 1) {
        if (file_size < 10) {
            throw std::runtime_error("Truncated npy v1 header");
        }
        header_len = static_cast<unsigned char>(bytes[8]) |
                     (static_cast<unsigned char>(bytes[9]) << 8);
        header_start = 10;
    } else if (major == 2) {
        if (file_size < 12) {
            throw std::runtime_error("Truncated npy v2 header");
        }
        const auto* len_bytes = reinterpret_cast<const unsigned char*>(bytes + 8);
        header_len = static_cast<std::size_t>(len_bytes[0]) |
                     (static_cast<std::size_t>(len_bytes[1]) << 8) |
                     (static_cast<std::size_t>(len_bytes[2]) << 16) |
                     (static_cast<std::size_t>(len_bytes[3]) << 24);
        header_start = 12;
    } else {
        throw std::runtime_error("Unsupported npy version");
    }

    const std::size_t header_end = header_start + header_len;
    if (header_end > file_size) {
        throw std::runtime_error("Truncated npy header payload");
    }

    const std::string header(reinterpret_cast<const char*>(bytes + header_start), header_len);
    expect_c_order(header);

    const std::size_t shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Missing npy shape field");
    }

    NpyHeaderInfo info;
    info.dtype = parse_descr(header);
    info.shape = parse_shape_tuple(header, shape_pos);
    info.header_end = header_end;
    return info;
}

std::size_t dtype_size(NpyDtype dtype) {
    switch (dtype) {
        case NpyDtype::Float32:
            return sizeof(float);
        case NpyDtype::Int32:
            return sizeof(std::int32_t);
        case NpyDtype::Int64:
            return sizeof(std::int64_t);
    }
    return 0;
}

template <typename T>
std::vector<T> load_typed_vector(const std::filesystem::path& path, NpyDtype expected) {
    auto file = std::make_shared<MmapFile>(path);
    if (!file) {
        throw std::runtime_error("Failed to mmap npy file: " + path.string());
    }

    const auto* bytes = file->data();
    const auto info = parse_npy_header(bytes, file->size());
    if (info.dtype != expected) {
        throw std::runtime_error("Unexpected npy dtype in: " + path.string());
    }

    std::size_t count = 1;
    for (const auto dim : info.shape) {
        count *= dim;
    }

    const auto* data =
        reinterpret_cast<const T*>(bytes + info.header_end);
    return std::vector<T>(data, data + count);
}

}  // namespace

std::size_t NpyArray::size() const {
    std::size_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    return count;
}

NpyArray load_npy_mmap(const std::filesystem::path& path) {
    auto file = std::make_shared<MmapFile>(path);
    if (!file) {
        throw std::runtime_error("Failed to mmap npy file: " + path.string());
    }

    const auto* bytes = file->data();
    const auto info = parse_npy_header(bytes, file->size());
    if (info.dtype != NpyDtype::Float32) {
        throw std::runtime_error("Expected float32 observation shard: " + path.string());
    }

    std::size_t count = 1;
    for (const auto dim : info.shape) {
        count *= dim;
    }

    const std::size_t bytes_needed = info.header_end + count * sizeof(float);
    if (bytes_needed > file->size()) {
        throw std::runtime_error("Truncated float32 npy payload: " + path.string());
    }

    NpyArray array;
    array.file = std::move(file);
    array.dtype = NpyDtype::Float32;
    array.shape = info.shape;
    array.element_size = sizeof(float);
    array.data = bytes + info.header_end;
    return array;
}

std::vector<std::int64_t> load_npy_int64_vector(const std::filesystem::path& path) {
    return load_typed_vector<std::int64_t>(path, NpyDtype::Int64);
}

std::vector<std::int32_t> load_npy_int32_vector(const std::filesystem::path& path) {
    return load_typed_vector<std::int32_t>(path, NpyDtype::Int32);
}

std::int64_t load_npy_scalar_int64(const std::filesystem::path& path) {
    auto values = load_npy_int64_vector(path);
    if (values.size() != 1) {
        throw std::runtime_error("Expected scalar int64 npy: " + path.string());
    }
    return values[0];
}

}  // namespace cs179
