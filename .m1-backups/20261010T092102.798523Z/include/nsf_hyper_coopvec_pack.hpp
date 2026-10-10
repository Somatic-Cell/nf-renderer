#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#include "optix8.hpp"

#include <cuda_fp16.h>

// -------------------------
// small helpers
// -------------------------
static inline size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

static inline void optix_check(OptixResult r, const char* msg) {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(std::string(msg) + " (OptixResult=" + std::to_string((int)r) + ")");
}

// -------------------------
// you provide these hooks
// -------------------------
// safetensors から読み込んだテンソルをここで参照できる形にしてください。
// 例: key="transform.transforms.0.hyper.0.weight" のようなキーでFP16配列とshapeを取得。
struct Fp16TensorView {
    const __half* data = nullptr;           // contiguous
    int64_t dim0 = 0;                       // e.g., out_features
    int64_t dim1 = 0;                       // e.g., in_features
};

// -------------------------
// packed artifacts (host)
// -------------------------
struct CoopVecLayerDesc {
    uint32_t w_offset_bytes = 0; // base + offset -> packed matrix blob
    uint32_t b_offset_bytes = 0; // base + offset -> fp16 bias vector (raw)
    uint16_t in_dim = 0;         // K (after padding)
    uint16_t out_dim = 0;        // N
};

struct CoopVecPackedHost {
    std::vector<uint8_t> blob;          // [packed matrices][biases] in one buffer (aligned)
    std::vector<CoopVecLayerDesc> layers; // size = transforms * 3 (hyper MLP 3 layers assumed)
    int transforms = 0;
    int layers_per_transform = 3;
    int input_dim_padded = 0;           // e.g. 16 (if you pad 3->16)

    // 参照しやすいヘルパ
    const CoopVecLayerDesc& layer(int t, int l) const {
        return layers.at(t * layers_per_transform + l);
    }
};

// -------------------------
// packed artifacts (device)
// -------------------------
struct CoopVecPackedGpu {
    CUdeviceptr d_blob = 0;
    size_t bytes = 0;

    void free() {
        if (d_blob) {
            CUDA_CHECK(cudaFree((void*)d_blob), "cudaFree(d_blob)");
            d_blob = 0;
            bytes = 0;
        }
    }
    ~CoopVecPackedGpu() { free(); }

    // move-only
    CoopVecPackedGpu() = default;
    CoopVecPackedGpu(const CoopVecPackedGpu&) = delete;
    CoopVecPackedGpu& operator=(const CoopVecPackedGpu&) = delete;
    CoopVecPackedGpu(CoopVecPackedGpu&& o) noexcept { *this = std::move(o); }
    CoopVecPackedGpu& operator=(CoopVecPackedGpu&& o) noexcept {
        if (this != &o) {
            free();
            d_blob = o.d_blob; bytes = o.bytes;
            o.d_blob = 0; o.bytes = 0;
        }
        return *this;
    }

    void upload(const CoopVecPackedHost& host, cudaStream_t stream = 0) {
        free();
        bytes = host.blob.size();
        CUDA_CHECK(cudaMalloc((void**)&d_blob, bytes), "cudaMalloc(d_blob)");
        CUDA_CHECK(cudaMemcpyAsync((void*)d_blob, host.blob.data(), bytes, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(blob)");
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
};

struct CoopVecPackOptions {
    int input_pad = 16; // 3 -> 16 推奨
    // ここは OptiX ヘッダの enum 名に合わせてください
    OptixCoopVecMatrixLayout src_layout = OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
    OptixCoopVecMatrixLayout dst_layout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
    OptixCoopVecElemType element_type   = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16; // 例: FP16
};

static inline uint32_t roundUp64_u32(uint32_t x) { return (x + 63u) & ~63u; }
static inline size_t   roundUp64_sz(size_t x)    { return (x + 63ull) & ~63ull; }


static inline void setSingleMatrixDesc(
    OptixDeviceContext optixCtx,
    OptixNetworkDescription& net,
    OptixCoopVecMatrixDescription& layer,
    uint32_t N, uint32_t K,
    OptixCoopVecElemType elemType,
    OptixCoopVecMatrixLayout layout,
    size_t offsetInBytes
){
    layer.N = N; // out
    layer.K = K; //in
    layer.offsetInBytes = offsetInBytes;
    layer.elementType = elemType;
    layer.layout = layout;
    // ROW_MAJOR の場合 stride = sizeof(elem)*K（INFERENCING_OPTIMAL では無視される）
    const uint32_t elemBytes = (elemType == OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16) ? 2u : 4u;
    layer.rowColumnStrideInBytes = elemBytes * K;

    size_t sizeInBytes = 0;
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(
        optixCtx,
        N, K,
        elemType,
        layout,
        layer.rowColumnStrideInBytes,
        &sizeInBytes
    ));
    layer.sizeInBytes = roundUp64_sz(sizeInBytes);

    net.layers = &layer;
    net.numLayers = 1u;
}

static std::vector<__half> make_NxK_rowmajor_padded(
    const __half* W_outxk, int64_t N, int64_t K, int Kpad)
{
    std::vector<__half> M((size_t)N * (size_t)Kpad, __half{0});
    for (int64_t r = 0; r < N; ++r) {
        std::memcpy(&M[(size_t)r * (size_t)Kpad],
                    &W_outxk[(size_t)r * (size_t)K],
                    (size_t)K * sizeof(__half));
    }
    return M; // (N x Kpad) row-major
}


static inline uint32_t idxTL(uint32_t t, uint32_t l) { return t * 3u + l; } // l=0,1,2

// FP16行列の coopvec packed size（INFERENCING_OPTIMAL）を計算して 64B アラインした値を返す
static inline uint32_t coopSizeFP16(
    OptixDeviceContext ctx,
    uint32_t N, uint32_t K,
    OptixCoopVecMatrixLayout layout
){
    size_t sizeInBytes = 0;
    OPTIX_CHECK(optixCoopVecMatrixComputeSize(
        ctx,
        N, K,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
        layout,
        sizeof(uint16_t) * K, // ROW_MAJOR のときのstride（OPTIMALでは無視される想定）
        &sizeInBytes
    ));
    return roundUp64_u32((uint32_t)sizeInBytes);
}

// PyTorch weight は (N x K)。layer0 のみ Kpad にゼロ拡張して (N x Kpad) を作る。
static inline std::vector<uint16_t> makeNxKpad_fp16_bits(
    const uint16_t* W_bits_outxk, uint32_t N, uint32_t K, uint32_t Kpad
){
    std::vector<uint16_t> out((size_t)N * (size_t)Kpad, 0);
    for(uint32_t r=0; r<N; ++r){
        std::memcpy(
            out.data() + (size_t)r * (size_t)Kpad,
            W_bits_outxk + (size_t)r * (size_t)K,
            (size_t)K * sizeof(uint16_t)
        );
    }
    return out; // (N x Kpad) row-major
}
