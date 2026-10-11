#pragma once

#include "phase_function.cuh"

#include <optix_device.h> 
#include <cuda_fp16.h>      // half を使うなら（OptixCoopVec<half, N> 等）

// 例：固定ネット（config 前提）
static constexpr int NSF_T  = 2;    // transforms
static constexpr int C      = 3;    // context
static constexpr int H      = 32;   // hidden
static constexpr int K      = 16;   // bins
static constexpr int PHI    = 2*K-1; // 62

// 実際の CoopVec サイズは padding 後（launchParams.nsf.inputPad や N/K を参照して決める）
// 典型：inputPad=16, phiPad=96 など
static constexpr int IN_PAD  = 16;
static constexpr int PHI_PAD = 32;

// ---- NSF constants (zuko 1.5.0 defaults unless overridden) ----
static constexpr int   NSF_BINS  = 16;
static constexpr int   NSF_PHI   = 2 * NSF_BINS - 1; // 62
static constexpr float NSF_BOUND = 2.5f;
static constexpr float NSF_SLOPE = 1e-3f;

static constexpr float NSF_SCALE = 3.0f;   // t = atanh(mu) / scale
static constexpr float NSF_EPS   = 1e-6f;  // clamp

#ifndef NSF_RQS_VARIANT
#define NSF_RQS_VARIANT 1
#endif

// // 型は状況に応じて（half/float）
// // まずは matmul は half のまま、後段の RQS は float に落とすのが無難です
// using VIn   = OptixCoopVec<half, IN_PAD>;
// using VHid  = OptixCoopVec<half, H>;
// using VOut  = OptixCoopVec<half, PHI_PAD>;

// static __forceinline__ __device__
// VHid relu(const VHid& x)
// {
//     // ReLU: max(x, 0)
//     return optixCoopVecMax(x, half(0.0f)); // あなたの OptiX ヘッダの名前に合わせてください
// }

// // static __forceinline__ __device__
// static __noinline__ __device__
// VHid evalHyperLayer0(const LaunchParams& lp, int t, const VIn& in)
// {
//     const CUdeviceptr base = lp.nsf.packedBase;
//     const uint32_t wOff = lp.nsf.wOffset[t][0];
//     const uint32_t bOff = lp.nsf.bOffset[t][0];

//     VHid out;
//     out = optixCoopVecMatMul<
//         VHid, VIn,
//         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
//         OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
//         false,
//         H, IN_PAD,
//         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
//     >(in, base, wOff, base, bOff);

//     return out;
// }

// // static __forceinline__ __device__
// static __noinline__ __device__
// VHid evalHyperLayer1(const LaunchParams& lp, int t, const VHid& in)
// {
//     const CUdeviceptr base = lp.nsf.packedBase;
//     const uint32_t wOff = lp.nsf.wOffset[t][1];
//     const uint32_t bOff = lp.nsf.bOffset[t][1];

//     VHid out;
//     out = optixCoopVecMatMul<
//         VHid, VHid,
//         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
//         OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
//         false,
//         H, H,
//         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
//     >(in, base, wOff, base, bOff);

//     return out;
// }

// // static __forceinline__ __device__
// static __noinline__ __device__
// VOut evalHyperLayer2(const LaunchParams& lp, int t, const VHid& in)
// {
//     const CUdeviceptr base = lp.nsf.packedBase;
//     const uint32_t wOff = lp.nsf.wOffset[t][2];
//     const uint32_t bOff = lp.nsf.bOffset[t][2];

//     VOut out;
//     out = optixCoopVecMatMul<
//         VOut, VHid,
//         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
//         OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
//         false,
//         PHI_PAD, H,
//         OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
//     >(in, base, wOff, base, bOff);

//     return out;
// }

// // static __forceinline__ __device__
// // void evalHyperMLP_phi95(const LaunchParams& lp, int t, const float c0, const float c1, const float c2,
// //                         float phi_out[PHI])
// // {
// //     // 入力パディング
// //     alignas(16) VIn x(half(0.0f));
// //     x[0] = __float2half(c0);
// //     x[1] = __float2half(c1);
// //     x[2] = __float2half(c2);

// //     // 3->64->64->phiPad
// //     alignas(16) VHid h0 = relu(evalHyperLayer0(lp, t, x));
// //     alignas(16) VHid h1 = relu(evalHyperLayer1(lp, t, h0));
// //     alignas(16) VOut y  = evalHyperLayer2(lp, t, h1);

// //     // 必要な 95 次元だけ float に取り出す
// //     #pragma unroll 1
// //     for (int i = 0; i < PHI; ++i)
// //         phi_out[i] = __half2float(y[i]);
// // }


// // static __forceinline__ __device__
// // void evalHyperMLP_phi95_vec(const LaunchParams& lp, int t,
// //                             float c0, float c1, float c2,
// //                             VOut& y_out)   // ← 参照で受ける（コピー回避）
// // {
// //     alignas(16) VIn x(half(0.0f));
// //     x[0] = __float2half(c0);
// //     x[1] = __float2half(c1);
// //     x[2] = __float2half(c2);

// //     alignas(16) VHid h0 = relu(evalHyperLayer0(lp, t, x));
// //     alignas(16) VHid h1 = relu(evalHyperLayer1(lp, t, h0));
// //     y_out = evalHyperLayer2(lp, t, h1); // VOut(half×96)
// // }

// static __noinline__ __device__
// // static __noinline__ __device__
// void evalHyperMLP_phi62_vec(const LaunchParams& lp, int t,
//                             float c0, float c1, float c2,
//                             VOut& y_out)   // ← 参照で受ける（コピー回避）
// {
//     alignas(16) VIn x(half(0.0f));
//     x[0] = __float2half(c0);
//     x[1] = __float2half(c1);
//     x[2] = __float2half(c2);

//     alignas(16) VHid h0 = relu(evalHyperLayer0(lp, t, x));
//     alignas(16) VHid h1 = relu(evalHyperLayer1(lp, t, h0));
//     y_out = evalHyperLayer2(lp, t, h1); // VOut(half×96)
// }

// static __forceinline__ __device__
// float phi_f(const VOut& y, int i) { return __half2float(y[i]); }

#include "nsf_simd_constant_backend.cuh"

// zuko: searchsorted(seq, x) = sum(seq < x)
static __forceinline__ __device__ int searchsorted_strict_lt(const float* seq, int n, float x) {
    int cnt = 0;
    #pragma unroll 1
    for (int i = 0; i < n; ++i) cnt += (seq[i] < x);
    return cnt;
}

static __forceinline__ __device__
void split_phi(const float phi[NSF_PHI], float w[NSF_BINS], float h[NSF_BINS], float d[NSF_BINS - 1])
{
    // #pragma unroll 1
    for (int i = 0; i < NSF_BINS; ++i) w[i] = phi[i];
    // #pragma unroll 1
    for (int i = 0; i < NSF_BINS; ++i) h[i] = phi[NSF_BINS + i];
    // #pragma unroll 1
    for (int i = 0; i < NSF_BINS - 1; ++i) d[i] = phi[2 * NSF_BINS + i];
}

static __forceinline__ __device__
void split_phi_fixedx(const float phi[NSF_PHI], float h[NSF_BINS], float d[NSF_BINS - 1])
{
    for (int i = 0; i < NSF_BINS; ++i) h[i] = phi[i];
    for (int i = 0; i < NSF_BINS - 1; ++i) d[i] = phi[NSF_BINS + i];
}


struct Rqs1D {
    float xk[NSF_BINS + 1]; // horizontal knots
    float yk[NSF_BINS + 1]; // vertical knots
    float dk[NSF_BINS + 1]; // derivatives at knots (positive)
};

static __forceinline__ __device__
float saturate_param(float v, float log_slope, float factor) {
    // zuko:
    // widths  = w / (1 + abs(2*w/log(slope)))
    // heights = h / (1 + abs(2*h/log(slope)))
    // deriv   = d / (1 + abs(d/log(slope)))
    const float denom = 1.0f + fabsf(factor * v / log_slope);
    return v / denom;
}

static __forceinline__ __device__
void width_softmax_stats_from_phi(
    const VOut& yphi,
    float log_slope,
    float& maxW,
    float& invSumW)
{
    maxW = -1e30f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        const float vW = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        maxW = fmaxf(maxW, vW);
    }

    float sumW = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        const float vW = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        sumW += __expf(vW - maxW);
    }

    invSumW = 1.0f / fmaxf(sumW, 1e-20f);
}

static __forceinline__ __device__
// static __noinline__ __device__
void softmax_32(const float in[NSF_BINS], float out[NSF_BINS])
{
    float m = in[0];
    #pragma unroll 4
    for (int i = 1; i < NSF_BINS; ++i) m = fmaxf(m, in[i]);

    float sum = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        out[i] = __expf(in[i] - m);
        sum += out[i];
    }
    const float inv = 1.0f / fmaxf(sum, 1e-20f);
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) out[i] *= inv;
}

static __device__
// // static __forceinline__ __device__
void softmax_exp32_from_phi(const VOut& y, int base, float log_slope, float factor,
                            float expv[NSF_BINS], float& inv_sum)
{
    // max
    float m = -1e30f;
    #pragma unroll 8
    for (int i = 0; i < NSF_BINS; ++i) {
        float v = saturate_param(phi_f(y, base + i), log_slope, factor);
        m = fmaxf(m, v);
    }
    // exp + sum
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < NSF_BINS; ++i) {
        float v = saturate_param(phi_f(y, base + i), log_slope, factor);
        float e = __expf(v - m);
        expv[i] = e;
        sum += e;
    }
    inv_sum = 1.0f / fmaxf(sum, 1e-20f);
}

// zuko 1.5.0 __init__ を反映して RQS のノット列/導関数を構築
static __forceinline__ __device__
void build_rqs_from_phi(const float phi[NSF_PHI], Rqs1D& rqs,
                        float bound = NSF_BOUND, float slope = NSF_SLOPE)
{
    float h_raw[NSF_BINS], d_raw[NSF_BINS - 1];
    // split_phi(phi, w_raw, h_raw, d_raw);
    split_phi_fixedx(phi, h_raw, d_raw);

    const float log_slope = __logf(slope); // < 0

    // saturation
    // #pragma unroll 1
    for (int i = 0; i < NSF_BINS; ++i) {
        // w_raw[i] = saturate_param(w_raw[i], log_slope, 2.0f);
        h_raw[i] = saturate_param(h_raw[i], log_slope, 2.0f);
    }
    // #pragma unroll 1
    for (int i = 0; i < NSF_BINS - 1; ++i) {
        d_raw[i] = saturate_param(d_raw[i], log_slope, 1.0f);
    }

    // widths/heights: softmax then pad left with 0, then cumsum, then scale to [-B,B]
    float h_sm[NSF_BINS];
    // softmax_32(w_raw, w_sm);
    softmax_32(h_raw, h_sm);

    // cumulative with pad(1,0) value=0:
    // float cw = 0.0f;
    float ch = 0.0f;

    // i=0 is padded 0 -> cumsum(0)=0
    rqs.xk[0] = -bound;
    rqs.yk[0] = -bound;

    // #pragma unroll 1
    for (int i = 1; i <= NSF_BINS; ++i) {
        // cw += w_sm[i - 1];
        ch += h_sm[i - 1];
        rqs.xk[i] = -bound + (2.0f * bound) * (float(i) / float(NSF_BINS));
        rqs.yk[i] = bound * (2.0f * ch - 1.0f);
    }
    // 理論上 rqs.xk[NSF_BINS]=+B, rqs.yk[NSF_BINS]=+B

    // derivatives: pad(1,1) with 0 then exp
    rqs.dk[0] = 1.0f; // exp(0)
    // #pragma unroll 1
    for (int i = 1; i < NSF_BINS; ++i) {
        rqs.dk[i] = __expf(d_raw[i - 1]);
    }
    rqs.dk[NSF_BINS] = 1.0f; // exp(0)
}

static __forceinline__ __device__
float dk_from_phi(const VOut& y, int knot, float log_slope)
{
    if (knot <= 0 || knot >= NSF_BINS) return 1.0f; // pad(1,1) exp(0)
    // d_raw index: 2*NSF_BINS + (knot-1)  (knot=1..31)
    // float d_raw = saturate_param(phi_f(y, 2 * NSF_BINS + (knot - 1)), log_slope, 1.0f);
    float d_raw = saturate_param(phi_f(y, NSF_BINS + (knot - 1)), log_slope, 1.0f);
    return __expf(d_raw);
}

static __noinline__ __device__
void rqs_forward_and_ladj(const Rqs1D& rqs, float x, float& y, float& ladj)
{
    // k = searchsorted(horizontal, x) - 1
    const int nKnots = NSF_BINS + 1;
    int k = searchsorted_strict_lt(rqs.xk, nKnots, x) - 1;

    // mask: 0 <= k < bins
    if (k < 0 || k >= NSF_BINS) {
        y = x;
        ladj = 0.0f;
        return;
    }

    const float x0 = rqs.xk[k];
    const float x1 = rqs.xk[k + 1];
    const float y0 = rqs.yk[k];
    const float y1 = rqs.yk[k + 1];
    const float d0 = rqs.dk[k];
    const float d1 = rqs.dk[k + 1];

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float z = (x - x0) * inv_dx; // in [0,1]
    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float numer = s * z * z + d0 * z * omz;

    y = y0 + (y1 - y0) * numer / fmaxf(denom, 1e-20f);

    // jacobian (zuko)
    const float term = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J = (s * s) * term / fmaxf(denom * denom, 1e-20f);

    ladj = logf(fmaxf(J, 1e-30f));
}

// static __device__ 
static __noinline__ __device__
void rqs_forward_and_ladj_from_phiV(const VOut& yphi, float x, float& y, float& ladj,
                                    float bound = NSF_BOUND, float slope = NSF_SLOPE)
{
#if NSF_RQS_VARIANT == 1

    // const float log_slope = __logf(slope);

    // // x を [0,1] に正規化（xk[i] = bound*(2*cw - 1) に対応）
    // float u = (x / bound + 1.0f) * 0.5f;
    // if (!(u > 0.0f && u < 1.0f)) { y = x; ladj = 0.0f; return; }

    // float maxW, invSumW;
    // width_softmax_stats_from_phi(yphi, log_slope, maxW, invSumW);

    // float cw = 0.0f;
    // int k = -1;
    // float x0u = 0.0f, x1u = 0.0f;

    // #pragma unroll 1
    // for (int i = 0; i < NSF_BINS; ++i) {
    //     const float vW = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
    //     const float eW = __expf(vW - maxW);
    //     const float w_sm = eW * invSumW;

    //     const float x0 = cw;
    //     cw += w_sm;
    //     const float x1 = cw;

    //     if (k < 0 && u <= x1) {
    //         k = i;
    //         x0u = x0;
    //         x1u = x1;
    //     }
    // }
    // if (k < 0 || k >= NSF_BINS) { y = x; ladj = 0.0f; return; }
    const float log_slope = __logf(slope);
    float u = (x / bound + 1.0f) * 0.5f;
    if (!(u > 0.0f && u < 1.0f)) { y = x; ladj = 0.0f; return; }


    int k = min(int(u * NSF_BINS), NSF_BINS - 1);
    const float x0u = float(k) / float(NSF_BINS);
    const float x1u = float(k + 1) / float(NSF_BINS);

    // ---- HLESS: y0u/y1u だけを高さsoftmaxから求める（h_exp配列なし） ----
    float maxH = -1e30f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        // const float vH = saturate_param(phi_f(yphi, NSF_BINS + i), log_slope, 2.0f);
        const float vH = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        maxH = fmaxf(maxH, vH);
    }

    float sumH = 0.0f;
    float y0E = 0.0f, y1E = 0.0f; // unnormalized prefix at k
    float pre = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        // const float vH = saturate_param(phi_f(yphi, NSF_BINS + i), log_slope, 2.0f);
        const float vH = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        const float e  = __expf(vH - maxH);
        sumH += e;
        if (i == k) {
            y0E = pre;
            pre += e;
            y1E = pre;
        } else {
            pre += e;
        }
    }
    const float invSumH = 1.0f / fmaxf(sumH, 1e-20f);
    const float y0u = y0E * invSumH;
    const float y1u = y1E * invSumH;

    // [-B,B] へ戻す
    const float x0 = bound * (2.0f * x0u - 1.0f);
    const float x1 = bound * (2.0f * x1u - 1.0f);
    const float y0 = bound * (2.0f * y0u - 1.0f);
    const float y1 = bound * (2.0f * y1u - 1.0f);

    const float d0 = dk_from_phi(yphi, k,   log_slope);
    const float d1 = dk_from_phi(yphi, k+1, log_slope);

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float z   = (x - x0) * inv_dx;
    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float numer = s * z * z + d0 * z * omz;

    y = y0 + (y1 - y0) * numer / fmaxf(denom, 1e-20f);

    const float term  = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J     = (s * s) * term / fmaxf(denom * denom, 1e-20f);
    ladj = logf(fmaxf(J, 1e-30f));
#else
    const float log_slope = __logf(slope);

    // x を [0,1] に正規化（xk[i] = bound*(2*cw - 1) に対応）
    float u = (x / bound + 1.0f) * 0.5f;
    if (!(u > 0.0f && u < 1.0f)) { y = x; ladj = 0.0f; return; }

    float w_exp[NSF_BINS], h_exp[NSF_BINS];
    float invW, invH;
    softmax_exp32_from_phi(yphi, 0,           log_slope, 2.0f, w_exp, invW);
    softmax_exp32_from_phi(yphi, NSF_BINS,    log_slope, 2.0f, h_exp, invH);

    // bin 探索（x方向は cumW）
    float cw = 0.0f, ch = 0.0f;
    int k = -1;
    float x0u=0, x1u=0, y0u=0, y1u=0;

    #pragma unroll 1
    for (int i = 0; i < NSF_BINS; ++i) {
        float w_sm = w_exp[i] * invW;
        float h_sm = h_exp[i] * invH;

        float x0 = cw; cw += w_sm; float x1 = cw;
        float y0 = ch; ch += h_sm; float y1 = ch;

        // searchsorted_strict_lt(...)-1 と同等にするため「<=」で左側 bin を取る
        if (k < 0 && u <= x1) {
            k = i;
            x0u = x0; x1u = x1;
            y0u = y0; y1u = y1;
        }
    }
    if (k < 0 || k >= NSF_BINS) { y = x; ladj = 0.0f; return; }

    // [-B,B] へ戻す
    const float x0 = bound * (2.0f * x0u - 1.0f);
    const float x1 = bound * (2.0f * x1u - 1.0f);
    const float y0 = bound * (2.0f * y0u - 1.0f);
    const float y1 = bound * (2.0f * y1u - 1.0f);

    const float d0 = dk_from_phi(yphi, k,   log_slope);
    const float d1 = dk_from_phi(yphi, k+1, log_slope);

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float z   = (x - x0) * inv_dx;
    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float numer = s * z * z + d0 * z * omz;

    y = y0 + (y1 - y0) * numer / fmaxf(denom, 1e-20f);

    const float term  = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J     = (s * s) * term / fmaxf(denom * denom, 1e-20f);
    ladj = logf(fmaxf(J, 1e-30f));
#endif
}

static __forceinline__ __device__
// static __noinline__ __device__
void rqs_inverse_and_ladj_fwd(const Rqs1D& rqs, float y, float& x, float& ladj_fwd)
{
    const int nKnots = NSF_BINS + 1;
    int k = searchsorted_strict_lt(rqs.yk, nKnots, y) - 1;

    if (k < 0 || k >= NSF_BINS) {
        x = y;
        ladj_fwd = 0.0f;
        return;
    }

    const float x0 = rqs.xk[k];
    const float x1 = rqs.xk[k + 1];
    const float y0 = rqs.yk[k];
    const float y1 = rqs.yk[k + 1];
    const float d0 = rqs.dk[k];
    const float d1 = rqs.dk[k + 1];

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float y_ = (y - y0); // mask * (y-y0)

    const float a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2.0f * s);
    const float b = (y1 - y0) * d0       - y_ * (d0 + d1 - 2.0f * s);
    const float c = -s * y_;

    const float disc = fmaxf(b * b - 4.0f * a * c, 0.0f);
    const float sqrt_disc = sqrtf(disc);

    // zuko: z = 2c / (-b - sqrt(...))
    const float denom = (-b - sqrt_disc);
    const float z = (fabsf(denom) > 1e-20f) ? (2.0f * c / denom) : 0.0f;

    x = x0 + z * (x1 - x0);

    // ladj_fwd at x: same as forward jacobian formula, using z
    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom2 = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float term = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J = (s * s) * term / fmaxf(denom2 * denom2, 1e-20f);

    ladj_fwd = logf(fmaxf(J, 1e-30f));
}

static __noinline__ __device__ 
// static __noinline__ __device__
void rqs_inverse_and_ladj_fwd_from_phiV(const VOut& yphi, float y, float& x, float& ladj_fwd,
                                        float bound = NSF_BOUND, float slope = NSF_SLOPE)
{
#if NSF_RQS_VARIANT == 1
    const float log_slope = __logf(slope);

    float v = (y / bound + 1.0f) * 0.5f;
    if (!(v > 0.0f && v < 1.0f)) { x = y; ladj_fwd = 0.0f; return; }

    // float maxW, invSumW;
    // width_softmax_stats_from_phi(yphi, log_slope, maxW, invSumW);

    // 既存どおり HLESS のための maxH / sumH を計算
    float maxH = -1e30f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        // const float vH = saturate_param(phi_f(yphi, NSF_BINS + i), log_slope, 2.0f);
        const float vH = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        maxH = fmaxf(maxH, vH);
    }

    float sumH = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < NSF_BINS; ++i) {
        // const float vH = saturate_param(phi_f(yphi, NSF_BINS + i), log_slope, 2.0f);
        const float vH = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        sumH += __expf(vH - maxH);
    }
    const float invSumH = 1.0f / fmaxf(sumH, 1e-20f);
    const float thrE = v * sumH;

    // scan
    // float cw = 0.0f, chE = 0.0f;
    float chE = 0.0f;
    int k = -1;
    // float x0u = 0.0f, x1u = 0.0f, y0u = 0.0f, y1u = 0.0f;
    float y0u = 0.0f, y1u = 0.0f;

    #pragma unroll 1
    for (int i = 0; i < NSF_BINS; ++i) {
        // const float vW = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        // const float eW = __expf(vW - maxW);
        // const float w_sm = eW * invSumW;

        // const float x0t = cw;
        // cw += w_sm;
        // const float x1t = cw;

        // const float vH = saturate_param(phi_f(yphi, NSF_BINS + i), log_slope, 2.0f);
        const float vH = saturate_param(phi_f(yphi, i), log_slope, 2.0f);
        const float eH = __expf(vH - maxH);
        const float y0E = chE;
        chE += eH;
        const float y1E = chE;

        if (k < 0 && thrE <= y1E) {
            k   = i;
            // x0u = x0t;
            // x1u = x1t;
            y0u = y0E * invSumH;
            y1u = y1E * invSumH;
        }
    }
    if (k < 0 || k >= NSF_BINS) { x = y; ladj_fwd = 0.0f; return; }

    const float x0u = float(k) / float(NSF_BINS);
    const float x1u = float(k + 1) / float(NSF_BINS);
    const float x0 = bound * (2.0f * x0u - 1.0f);
    const float x1 = bound * (2.0f * x1u - 1.0f);
    const float y0 = bound * (2.0f * y0u - 1.0f);
    const float y1 = bound * (2.0f * y1u - 1.0f);

    const float d0 = dk_from_phi(yphi, k,   log_slope);
    const float d1 = dk_from_phi(yphi, k+1, log_slope);

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float y_ = (y - y0);

    const float a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2.0f * s);
    const float b = (y1 - y0) * d0       - y_ * (d0 + d1 - 2.0f * s);
    const float c = -s * y_;

    const float disc = fmaxf(b * b - 4.0f * a * c, 0.0f);
    const float sqrt_disc = sqrtf(disc);

    const float denom = (-b - sqrt_disc);
    const float z = (fabsf(denom) > 1e-20f) ? (2.0f * c / denom) : 0.0f;

    x = x0 + z * (x1 - x0);

    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom2 = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float term = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J = (s * s) * term / fmaxf(denom2 * denom2, 1e-20f);

    ladj_fwd = logf(fmaxf(J, 1e-30f));
#else
    const float log_slope = __logf(slope);

    float v = (y / bound + 1.0f) * 0.5f;
    if (!(v > 0.0f && v < 1.0f)) { x = y; ladj_fwd = 0.0f; return; }

    float w_exp[NSF_BINS], h_exp[NSF_BINS];
    float invW, invH;
    softmax_exp32_from_phi(yphi, 0,           log_slope, 2.0f, w_exp, invW);
    softmax_exp32_from_phi(yphi, NSF_BINS,    log_slope, 2.0f, h_exp, invH);

    // bin 探索（y方向は cumH）。同時に x方向 cumW も回して x0/x1 を得る
    float cw = 0.0f, ch = 0.0f;
    int k = -1;
    float x0u=0, x1u=0, y0u=0, y1u=0;

    #pragma unroll 1
    for (int i = 0; i < NSF_BINS; ++i) {
        float w_sm = w_exp[i] * invW;
        float h_sm = h_exp[i] * invH;

        float x0t = cw; cw += w_sm; float x1t = cw;
        float y0t = ch; ch += h_sm; float y1t = ch;

        if (k < 0 && v <= y1t) {
            k = i;
            x0u = x0t; x1u = x1t;
            y0u = y0t; y1u = y1t;
        }
    }
    if (k < 0 || k >= NSF_BINS) { x = y; ladj_fwd = 0.0f; return; }

    const float x0 = bound * (2.0f * x0u - 1.0f);
    const float x1 = bound * (2.0f * x1u - 1.0f);
    const float y0 = bound * (2.0f * y0u - 1.0f);
    const float y1 = bound * (2.0f * y1u - 1.0f);

    const float d0 = dk_from_phi(yphi, k,   log_slope);
    const float d1 = dk_from_phi(yphi, k+1, log_slope);

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float y_ = (y - y0);

    const float a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2.0f * s);
    const float b = (y1 - y0) * d0       - y_ * (d0 + d1 - 2.0f * s);
    const float c = -s * y_;

    const float disc = fmaxf(b * b - 4.0f * a * c, 0.0f);
    const float sqrt_disc = sqrtf(disc);

    const float denom = (-b - sqrt_disc);
    const float z = (fabsf(denom) > 1e-20f) ? (2.0f * c / denom) : 0.0f;

    x = x0 + z * (x1 - x0);

    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom2 = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float term = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J = (s * s) * term / fmaxf(denom2 * denom2, 1e-20f);

    ladj_fwd = logf(fmaxf(J, 1e-30f));
#endif
}


static __forceinline__ __device__
float sampleMuHG(float u, float g)
{
    g = clamp(g, -0.999999f, 0.999999f);
    u = clamp(u, 0.0f, 0.99999994f);

    if (fabsf(g) < 1e-4f) {
        return 1.0f - 2.0f * u;
    } else {
        const float gg = g * g;
        const float t  = (1.0f - gg) / (1.0f - g + 2.0f * g * u);
        float mu = (1.0f + gg - t * t) / (2.0f * g);
        return clamp(mu, -1.0f, 1.0f);
    }
}

static __forceinline__ __device__
float logpdfMuHG(float mu, float g)
{
    g  = clamp(g, -0.999999f, 0.999999f);
    mu = clamp(mu, -1.0f, 1.0f);

    if (fabsf(g) < 1e-4f) {
        return logf(0.5f);
    } else {
        const float gg = g * g;
        const float denom = 1.0f + gg - 2.0f * g * mu;
        const float denom32 = denom * sqrtf(fmaxf(denom, 1e-20f));
        const float q = (1.0f - gg) / (2.0f * fmaxf(denom32, 1e-20f));
        return logf(fmaxf(q, 1e-30f));
    }
}

static __forceinline__ __device__
float logpdfBaseT(float t, float g)
{
    // mu = tanh(scale*t)
    const float x = NSF_SCALE * t;
    float mu = tanhf(x);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);

    const float logq = logpdfMuHG(mu, g);
    const float jac  = NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f); // dmu/dt
    return logq + logf(jac);
}

static __forceinline__ __device__
float sampleBaseT(float u, float g)
{
    float mu = sampleMuHG(u, g);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);
    return atanhf(mu) / NSF_SCALE;
}


// ----------------------------------------------------------------------------
// Phi cache (avoid repeated HyperMLP inference for the same (diameter, wavelength, g))
// ----------------------------------------------------------------------------
struct NSFPhiCache {
    alignas(16) VOut phi[NSF_T]; // per-transform spline parameters (half)
    float g;                     // base distribution parameter (HG g)
};

// static __forceinline__ __device__
static __noinline__ __device__
void buildNSFPhiCache(const LaunchParams& lp, float c0, float c1, float g, NSFPhiCache& cache)
{
    cache.g = g;
    #pragma unroll
    for (int i = 0; i < NSF_T; ++i) {
        evalHyperMLP_phi62_vec(lp, i, c0, c1, g, cache.phi[i]);
    }
}

static __forceinline__ __device__
float flow_logpdf_t_phiCached(const NSFPhiCache& cache, float t)
{
    float x = t;
    float sum_ladj_fwd = 0.0f;

    #pragma unroll
    for (int i = 0; i < NSF_T; ++i) {
        float y, ladj_fwd;
        rqs_forward_and_ladj_from_phiV(cache.phi[i], x, y, ladj_fwd);
        sum_ladj_fwd += ladj_fwd;
        x = y;
    }

    const float logp0 = logpdfBaseT(x, cache.g);
    return logp0 + sum_ladj_fwd;
}

static __forceinline__ __device__
float flow_sample_t_and_logpdf_phiCached(const NSFPhiCache& cache, float u_base, float& out_logp_t)
{
    float y = sampleBaseT(u_base, cache.g);
    float logp = logpdfBaseT(y, cache.g);

    #pragma unroll
    for (int i = NSF_T - 1; i >= 0; --i) {
        float x, ladj_fwd;
        rqs_inverse_and_ladj_fwd_from_phiV(cache.phi[i], y, x, ladj_fwd);
        logp += ladj_fwd;
        y = x;
    }

    out_logp_t = logp;
    return y; // t
}

static __noinline__ __device__
void profile_build_phi_cache(
    const LaunchParams& lp,
    float c0, float c1, float g,
    NSFPhiCache& cache)
{
    buildNSFPhiCache(lp, c0, c1, g, cache);
}

static __noinline__ __device__
float profile_flow_logpdf_cached(
    const NSFPhiCache& cache, float t)
{
    return flow_logpdf_t_phiCached(cache, t);
}

static __noinline__ __device__
float profile_flow_sample_cached(
    const NSFPhiCache& cache, float u_base, float& out_logp_t)
{
    return flow_sample_t_and_logpdf_phiCached(cache, u_base, out_logp_t);
}

static  __device__ __noinline__
float evalPhaseFunctionNF_phiCached(const NSFPhiCache& cache, float cosTheta)
{
    float mu = clamp(cosTheta, -1.0f + NSF_EPS, 1.0f - NSF_EPS);
    const float t = atanhf(mu) / NSF_SCALE;

    // const float logp_t = flow_logpdf_t_phiCached(cache, t);
    const float logp_t = profile_flow_logpdf_cached(cache, t);

    const float denom = (2.0f * m1::pi) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    return __expf(logp_t) / denom;
}

static __device__ __noinline__
PhaseSample samplePhaseFunctionNF_phiCached(const NSFPhiCache& cache,
                                            const float3& woWorld, float u1, float u2)
{
    PhaseSample ps;
    const float3 wo = normalize(woWorld);

    float logp_t;
    // const float t = flow_sample_t_and_logpdf_phiCached(cache, u1, logp_t);
    const float t = profile_flow_sample_cached(cache, u1, logp_t);

    float mu = tanhf(NSF_SCALE * t);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);

    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));

    const float phi = 2.0f * m1::pi * u2;
    const auto phiAngle = m1::bounded_sincos(phi);
    const float sinPhi = phiAngle.sine, cosPhi = phiAngle.cosine;

    const float3 local = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, mu);

    float3 tvec, bvec;
    makeONB(wo, tvec, bvec);
    ps.wi = normalize(local.x * tvec + local.y * bvec + local.z * wo);

    const float denom = (2.0f * m1::pi) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    ps.pdf = __expf(logp_t) / denom;

    return ps;
}

// ----------------------------------------------------------------------------
// RQS Cached
// ----------------------------------------------------------------------------

struct NSFFixedXRqsCache {
    alignas(16) float yk[NSF_T][NSF_BINS + 1];
    alignas(16) float dk[NSF_T][NSF_BINS + 1];
    float g;
};

static __noinline__ __device__
void buildNSFFixedXRqsCache(const LaunchParams& lp,
                            float c0, float c1, float g,
                            NSFFixedXRqsCache& cache)
{
    cache.g = g;
    const float log_slope = __logf(NSF_SLOPE);

    #pragma unroll
    for (int t = 0; t < NSF_T; ++t) {
        VOut phi;
        evalHyperMLP_phi62_vec(lp, t, c0, c1, g, phi);

        // ---- h の online softmax 統計を 1-pass で取る ----
        float m = -CUDART_INF_F;
        float s = 0.0f;

        #pragma unroll
        for (int i = 0; i < NSF_BINS; ++i) {
            const float a = saturate_param(phi_f(phi, i), log_slope, 2.0f);

            const float m_new = fmaxf(m, a);
            s = s * __expf(m - m_new) + __expf(a - m_new);
            m = m_new;
        }

        const float invS = 1.0f / fmaxf(s, 1e-20f);

        // ---- yk を prefix で直接構築 ----
        cache.yk[t][0] = -NSF_BOUND;
        float ch = 0.0f;

        #pragma unroll
        for (int i = 1; i <= NSF_BINS; ++i) {
            const float a = saturate_param(phi_f(phi, i - 1), log_slope, 2.0f);
            ch += __expf(a - m) * invS;
            cache.yk[t][i] = NSF_BOUND * (2.0f * ch - 1.0f);
        }

        // ---- dk はそのまま直書き ----
        cache.dk[t][0] = 1.0f;

        #pragma unroll
        for (int i = 1; i < NSF_BINS; ++i) {
            const float a = saturate_param(phi_f(phi, NSF_BINS + (i - 1)), log_slope, 1.0f);
            cache.dk[t][i] = __expf(a);
        }

        cache.dk[t][NSF_BINS] = 1.0f;
    }
}

static __forceinline__ __device__
void fixedx_rqs_forward_bin(float x, float x0, float x1,
                            float y0, float y1,
                            float d0, float d1,
                            float& y, float& ladj)
{
    const float w = x1 - x0;
    const float h = y1 - y0;
    const float s = h / fmaxf(w, 1e-20f);

    const float theta = (x - x0) / fmaxf(w, 1e-20f);
    const float omt   = 1.0f - theta;

    const float a = s * theta * theta + d0 * theta * omt;
    const float b = s + (d0 + d1 - 2.0f * s) * theta * omt;
    const float z = a / fmaxf(b, 1e-20f);

    y = y0 + h * z;

    const float num = s * s * (d1 * theta * theta + 2.0f * s * theta * omt + d0 * omt * omt);
    const float den = b * b;
    const float dydx = num / fmaxf(den, 1e-20f);
    ladj = __logf(fmaxf(dydx, 1e-20f));
}

static __forceinline__ __device__
void fixedx_rqs_forward_cached(const float yk[NSF_BINS + 1],
                               const float dk[NSF_BINS + 1],
                               float x, float& y, float& ladj)
{
    float u = (x / NSF_BOUND + 1.0f) * 0.5f;
    if (!(u > 0.0f && u < 1.0f)) {
        y = x;
        ladj = 0.0f;
        return;
    }

    const int k = min(int(u * NSF_BINS), NSF_BINS - 1);

    const float x0 = -NSF_BOUND + (2.0f * NSF_BOUND) * (float(k) / float(NSF_BINS));
    const float x1 = -NSF_BOUND + (2.0f * NSF_BOUND) * (float(k + 1) / float(NSF_BINS));

    fixedx_rqs_forward_bin(x, x0, x1,
                           yk[k], yk[k + 1],
                           dk[k], dk[k + 1],
                           y, ladj);
}


static __forceinline__ __device__
float flow_logpdf_t_rqsCached(const NSFFixedXRqsCache& cache, float t)
{
    float x = t;
    float sum_ladj_fwd = 0.0f;

    #pragma unroll
    for (int i = 0; i < NSF_T; ++i) {
        float y, ladj_fwd;
        fixedx_rqs_forward_cached(cache.yk[i], cache.dk[i], x, y, ladj_fwd);
        sum_ladj_fwd += ladj_fwd;
        x = y;
    }

    return logpdfBaseT(x, cache.g) + sum_ladj_fwd;
}

static __forceinline__ __device__
void fixedx_rqs_inverse_bin(float y, float x0, float x1,
                            float y0, float y1,
                            float d0, float d1,
                            float& x, float& ladj_fwd)
{
    const float w = x1 - x0;
    const float h = y1 - y0;
    const float s = h / fmaxf(w, 1e-20f);

    const float yy = y - y0;

    const float A = h * (s - d0) + yy * (d0 + d1 - 2.0f * s);
    const float B = h * d0 - yy * (d0 + d1 - 2.0f * s);
    const float C = -s * yy;

    const float disc = fmaxf(B * B - 4.0f * A * C, 0.0f);
    const float sqrt_disc = sqrtf(disc);

    float theta;
    if (fabsf(A) < 1e-20f) {
        theta = -C / fmaxf(B, 1e-20f);
    } else {
        theta = (2.0f * C) / fmaxf(-B - sqrt_disc, 1e-20f);
    }
    theta = fminf(fmaxf(theta, 0.0f), 1.0f);

    x = x0 + w * theta;

    const float omt = 1.0f - theta;
    const float den = s + (d0 + d1 - 2.0f * s) * theta * omt;
    const float num = s * s * (d1 * theta * theta + 2.0f * s * theta * omt + d0 * omt * omt);
    const float dydx = num / fmaxf(den * den, 1e-20f);
    ladj_fwd = __logf(fmaxf(dydx, 1e-20f));
}

static __forceinline__ __device__
void fixedx_rqs_inverse_cached(const float yk[NSF_BINS + 1],
                               const float dk[NSF_BINS + 1],
                               float y, float& x, float& ladj_fwd)
{
    float v = (y / NSF_BOUND + 1.0f) * 0.5f;
    if (!(v > 0.0f && v < 1.0f)) {
        x = y;
        ladj_fwd = 0.0f;
        return;
    }

    int k = -1;
    #pragma unroll
    for (int i = 0; i < NSF_BINS; ++i) {
        if (k < 0 && y <= yk[i + 1]) {
            k = i;
        }
    }
    if (k < 0) {
        x = y;
        ladj_fwd = 0.0f;
        return;
    }

    const float x0 = -NSF_BOUND + (2.0f * NSF_BOUND) * (float(k) / float(NSF_BINS));
    const float x1 = -NSF_BOUND + (2.0f * NSF_BOUND) * (float(k + 1) / float(NSF_BINS));

    fixedx_rqs_inverse_bin(y, x0, x1,
                           yk[k], yk[k + 1],
                           dk[k], dk[k + 1],
                           x, ladj_fwd);
}

static __forceinline__ __device__
float flow_sample_t_and_logpdf_rqsCached(const NSFFixedXRqsCache& cache,
                                         float u_base, float& out_logp_t)
{
    float y = sampleBaseT(u_base, cache.g);
    float logp = logpdfBaseT(y, cache.g);

    #pragma unroll
    for (int i = NSF_T - 1; i >= 0; --i) {
        float x, ladj_fwd;
        fixedx_rqs_inverse_cached(cache.yk[i], cache.dk[i], y, x, ladj_fwd);
        logp += ladj_fwd;
        y = x;
    }

    out_logp_t = logp;
    return y;
}

static __forceinline__ __device__
float evalPhaseFunctionNF_rqsCached(const NSFFixedXRqsCache& cache, float cosTheta)
{
    float mu = fminf(fmaxf(cosTheta, -1.0f + NSF_EPS), 1.0f - NSF_EPS);
    float t = atanhf(mu) / NSF_SCALE;
    float logp_t = flow_logpdf_t_rqsCached(cache, t);

    float denom = (2.0f * m1::pi) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    return __expf(logp_t) / denom;
}

static __forceinline__ __device__
PhaseSample samplePhaseFunctionNF_rqsCached(const NSFFixedXRqsCache& cache,
                                            const float3& woWorld,
                                            float u1, float u2)
{
    PhaseSample ps;
    const float3 wo = normalize(woWorld);

    float logp_t;
    float t = flow_sample_t_and_logpdf_rqsCached(cache, u1, logp_t);

    float mu = tanhf(NSF_SCALE * t);
    mu = fminf(fmaxf(mu, -1.0f + NSF_EPS), 1.0f - NSF_EPS);

    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));
    float phi = 2.0f * m1::pi * u2;

    const auto phiAngle = m1::bounded_sincos(phi);
    const float sinPhi = phiAngle.sine, cosPhi = phiAngle.cosine;

    float3 tvec, bvec;
    makeONB(wo, tvec, bvec);

    float3 local = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, mu);
    ps.wi = normalize(local.x * tvec + local.y * bvec + local.z * wo);

    float denom = (2.0f * m1::pi) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    ps.pdf = __expf(logp_t) / denom;
    return ps;
}


static __device__
float flow_logpdf_t(const LaunchParams& lp, float t, float c0, float c1, float c2)
{
    // forward 方向で base(z) に写しつつ、各段の ladj_fwd = log|dz/dx| を積算
    float x = t;
    float sum_ladj_fwd = 0.0f;

    #pragma unroll
    for (int i = 0; i < NSF_T; ++i) {

        VOut yphi;
        // evalHyperMLP_phi95_vec(lp, i, c0, c1, c2, yphi);
        evalHyperMLP_phi62_vec(lp, i, c0, c1, c2, yphi);

        float y, ladj_fwd;
        rqs_forward_and_ladj_from_phiV(yphi, x, y, ladj_fwd);

        sum_ladj_fwd += ladj_fwd;
        x = y;
    }

    // x は base の変数 z
    const float logp0 = logpdfBaseT(x, c2);
    return logp0 + sum_ladj_fwd;  // log p(t) = log p(z) + log|dz/dt|
}




static __device__
float flow_sample_t_and_logpdf(const LaunchParams& lp,
                               float u_base, float c0, float c1, float c2,
                               float& out_logp_t)
{
    // z ~ base
    float y = sampleBaseT(u_base, c2);
    float logp = logpdfBaseT(y, c2);

    // inverse 方向で data(t) に戻す。ただし logp は forward の ladj_fwd を足す。
    #pragma unroll
    for (int i = NSF_T - 1; i >= 0; --i) {

        alignas(16) VOut yphi;
        // evalHyperMLP_phi95_vec(lp, i, c0, c1, c2, yphi);
        evalHyperMLP_phi62_vec(lp, i, c0, c1, c2, yphi);

        float x, ladj_fwd;
        rqs_inverse_and_ladj_fwd_from_phiV(yphi, y, x, ladj_fwd);
        
        logp += ladj_fwd;                                // log p(x) = log p(y) + log|dy/dx|
        y = x;
    }

    out_logp_t = logp;
    return y; // t
}

static __device__
float evalPhaseFunctionNF(
    const LaunchParams& lp, 
    float cosTheta, 
    float c0,   // diameter
    float c1,   // wavelength
    float g
)
{
    // cosTheta は与えられるので t に戻す（atanh 安定化）
    float mu = clamp(cosTheta, -1.0f + NSF_EPS, 1.0f - NSF_EPS);
    const float t = atanhf(mu) / NSF_SCALE;

    const float logp_t = flow_logpdf_t(lp, t, c0, c1, g);

    // p_omega = p_t / (2π * scale * (1-mu^2))
    const float denom = (2.0f * m1::pi) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    const float pdf = __expf(logp_t) / denom;
    return pdf;
}

static __device__
PhaseSample samplePhaseFunctionNF(const LaunchParams& lp,
                                  const float3& woWorld, float u1, float u2,
                                  float c0, float c1, float g)
{
    PhaseSample ps;
    const float3 wo = normalize(woWorld);

    float logp_t;
    const float t = flow_sample_t_and_logpdf(lp, u1, c0, c1, g, logp_t);

    // t -> mu
    float mu = tanhf(NSF_SCALE * t);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);

    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));

    const float phi = 2.0f * m1::pi * u2;
    const auto phiAngle = m1::bounded_sincos(phi);
    const float sinPhi = phiAngle.sine, cosPhi = phiAngle.cosine;

    // wo を z 軸にしたローカル方向（z=mu）
    const float3 local = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, mu);

    float3 tvec, bvec;
    makeONB(wo, tvec, bvec);
    ps.wi = normalize(local.x * tvec + local.y * bvec + local.z * wo);

    // pdf over solid angle
    const float denom = (2.0f * m1::pi) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    ps.pdf = __expf(logp_t) / denom;

    return ps;
}

