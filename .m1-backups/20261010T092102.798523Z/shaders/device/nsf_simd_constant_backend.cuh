#pragma once

// Drop-in backend for the top MLP section of `neural_phase_function.cuh`.
// Usage inside `neural_phase_function.cuh`:
//   1. Keep the existing constexprs (`NSF_T`, `C`, `H`, `K`, `PHI`, `PHI_PAD`) above.
//   2. Replace the CoopVec block (`using VIn/VHid/VOut` ～ `phi_f`) with
//        #include "nsf_simd_constant_backend.cuh"
//   3. If you want a different generated constant-table header, define
//        #define NSF_SIMD_CONSTANT_DATA_HEADER "your_generated_constant_header.cuh"
//      before including this file.
//
// This backend intentionally keeps the legacy function name
// `evalHyperMLP_phi62_vec(...)` so the rest of the file can remain unchanged.

#ifndef NSF_T
#error "Include nsf_simd_constant_backend.cuh from neural_phase_function.cuh after NSF_T/C/H/K/PHI/PHI_PAD are defined."
#endif

#include <cuda_fp16.h>

#ifndef NSF_SIMD_CONSTANT_DATA_HEADER
#define NSF_SIMD_CONSTANT_DATA_HEADER "nsf_simd_constant_flow_flexible_data.cuh"
#endif
#include NSF_SIMD_CONSTANT_DATA_HEADER

namespace nsf_simd_constant_backend {

using namespace nsf_simd_constant_data;

static_assert(NSF_T   == kTransforms,   "NSF_T mismatch between neural_phase_function.cuh and generated constant data.");
static_assert(C       == kContext,      "C mismatch between neural_phase_function.cuh and generated constant data.");
static_assert(H       == kHidden,       "H mismatch between neural_phase_function.cuh and generated constant data.");
static_assert(K       == kBins,         "K mismatch between neural_phase_function.cuh and generated constant data.");
static_assert(PHI     == kPhi,          "PHI mismatch between neural_phase_function.cuh and generated constant data.");
static_assert(PHI_PAD == kPhiPad,       "PHI_PAD mismatch between neural_phase_function.cuh and generated constant data.");

struct alignas(16) PhiVec
{
    half lane[PHI_PAD];

    __device__ __forceinline__ half& operator[](int i) { return lane[i]; }
    __device__ __forceinline__ const half& operator[](int i) const { return lane[i]; }
};

static __forceinline__ __device__ half half_from_bits(uint16_t bits)
{
    return __ushort_as_half(static_cast<unsigned short>(bits));
}

template<int COUNT>
static __forceinline__ __device__ float dot_half2_row(const uint16_t* w_bits, const half* x)
{
    static_assert((COUNT % 2) == 0, "dot_half2_row requires an even COUNT.");
    const half* w_half = reinterpret_cast<const half*>(w_bits);
    const __half2* w2  = reinterpret_cast<const __half2*>(w_half);
    const __half2* x2  = reinterpret_cast<const __half2*>(x);

    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < COUNT / 2; ++i) {
        const float2 wf = __half22float2(w2[i]);
        const float2 xf = __half22float2(x2[i]);
        acc = fmaf(wf.x, xf.x, acc);
        acc = fmaf(wf.y, xf.y, acc);
    }
    return acc;
}

static __forceinline__ __device__ half relu_to_half(float x)
{
    return __float2half_rn(fmaxf(x, 0.0f));
}

template<int TIDX>
static __forceinline__ __device__ void eval_transform(float c0, float c1, float c2, PhiVec& y_out)
{
    alignas(8) half x0[kLayer0InputPad];
    x0[0] = __float2half_rn(c0);
    x0[1] = __float2half_rn(c1);
    x0[2] = __float2half_rn(c2);
    x0[3] = __float2half_rn(0.0f);

    alignas(16) half h0[H];
    #pragma unroll
    for (int r = 0; r < H; ++r) {
        float acc = __half2float(half_from_bits(kL0B[TIDX][r]));
        acc += dot_half2_row<kLayer0InputPad>(&kL0W[TIDX][r][0], x0);
        h0[r] = relu_to_half(acc);
    }

    alignas(16) half h1[H];
    #pragma unroll
    for (int r = 0; r < H; ++r) {
        float acc = __half2float(half_from_bits(kL1B[TIDX][r]));
        acc += dot_half2_row<H>(&kL1W[TIDX][r][0], h0);
        h1[r] = relu_to_half(acc);
    }

    #pragma unroll
    for (int r = 0; r < PHI_PAD; ++r) {
        float acc = __half2float(half_from_bits(kL2B[TIDX][r]));
        acc += dot_half2_row<H>(&kL2W[TIDX][r][0], h1);
        y_out.lane[r] = __float2half_rn(acc);
    }
}

} // namespace nsf_simd_constant_backend

using VOut = nsf_simd_constant_backend::PhiVec;

static __forceinline__ __device__
float phi_f(const VOut& y, int i)
{
    return __half2float(y.lane[i]);
}

static __noinline__ __device__
void evalHyperMLP_phi62_vec(const LaunchParams& lp, int t,
                            float c0, float c1, float c2,
                            VOut& y_out)
{
    (void)lp;
    switch (t) {
        case 0: nsf_simd_constant_backend::eval_transform<0>(c0, c1, c2, y_out); break;
        case 1: nsf_simd_constant_backend::eval_transform<1>(c0, c1, c2, y_out); break;
        default:
            #pragma unroll
            for (int i = 0; i < PHI_PAD; ++i) y_out.lane[i] = __float2half_rn(0.0f);
            break;
    }
}
