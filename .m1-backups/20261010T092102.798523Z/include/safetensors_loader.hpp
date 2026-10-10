#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

enum class DType { F16, F32, I32, I64, U8, UNKNOWN };

inline size_t dtype_bytes(DType dt) {
    switch (dt) {
    case DType::F16: return 2;
    case DType::F32: return 4;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::U8:  return 1;
    default: throw std::runtime_error("Unsupported dtype");
    }
}

inline DType parse_dtype(const std::string& s) {
    // safetensors は "F16", "F32", "I64" など（例）
    if (s == "F16") return DType::F16;
    if (s == "F32") return DType::F32;
    if (s == "I32") return DType::I32;
    if (s == "I64") return DType::I64;
    if (s == "U8")  return DType::U8;
    return DType::UNKNOWN;
}

struct TensorView {
    DType dtype{};
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr; // 生データ先頭
    size_t nbytes = 0;
};

struct SafeTensorsFile {
    std::vector<uint8_t> blob; // ファイル全体
    std::unordered_map<std::string, TensorView> tensors;
    nlohmann::json header;
};

inline uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    v = __builtin_bswap64(v);
#endif
    return v;
}
// inline uint64_t read_u64_le(const uint8_t* p) {
//     uint64_t v = 0;
//     for (int i = 7; i >= 0; --i) { v = (v << 8) | p[i]; }
//     return v;
// }

inline SafeTensorsFile load_safetensors(const std::string& path) {
    // read whole file
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open: " + path);
    ifs.seekg(0, std::ios::end);
    const size_t size = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    SafeTensorsFile st;
    st.blob.resize(size);
    if (!ifs.read((char*)st.blob.data(), (std::streamsize)size))
        throw std::runtime_error("Failed to read: " + path);

    if (size < 8) throw std::runtime_error("Invalid safetensors (too small)");

    const uint64_t header_len = read_u64_le(st.blob.data());
    std::cout << "safetensors header_len=" << header_len
          << " file_size=" << st.blob.size() << "\n";
    const size_t header_off = 8;
    const size_t data_off   = header_off + (size_t)header_len;

    if (data_off > st.blob.size())
        throw std::runtime_error("Invalid safetensors (header out of range)");

    // parse JSON header
    const char* hptr = (const char*)st.blob.data() + header_off;
    st.header = nlohmann::json::parse(std::string(hptr, hptr + header_len));

    // build tensor views
    for (auto it = st.header.begin(); it != st.header.end(); ++it) {
        const std::string name = it.key();
        if (name == "__metadata__") continue;

        const auto& obj = it.value();
        const std::string dtype_s = obj.at("dtype").get<std::string>();
        DType dt = parse_dtype(dtype_s);
        if (dt == DType::UNKNOWN)
            throw std::runtime_error("Unknown dtype for " + name + ": " + dtype_s);

        std::vector<int64_t> shape = obj.at("shape").get<std::vector<int64_t>>();
        auto offs = obj.at("data_offsets").get<std::vector<uint64_t>>();
        const uint64_t begin64 = (uint64_t)data_off + offs[0];
        const uint64_t end64   = (uint64_t)data_off + offs[1];
        
        // if (offs.size() != 2) throw std::runtime_error("data_offsets invalid: " + name);

        // const size_t begin = data_off + offs[0];
        // const size_t end   = data_off + offs[1];
        if (end64 > st.blob.size() || begin64 > end64)
            throw std::runtime_error("data range invalid: " + name);

        TensorView tv;
        tv.dtype = dt;
        tv.shape = std::move(shape);
        tv.data  = st.blob.data() + begin64;
        tv.nbytes = end64 - begin64;

        // sanity: size check
        size_t elems = 1;
        for (auto d : tv.shape) elems *= (size_t)d;
        const size_t expected = elems * dtype_bytes(tv.dtype);
        if (expected != tv.nbytes) {
            throw std::runtime_error("Size mismatch: " + name);
        }

        st.tensors.emplace(name, std::move(tv));
    }

    return st;
}