#ifndef PHASE_FUNCTION_CUH_
#define PHASE_FUNCTION_CUH_


#include <optix.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/NanoVDB.h>

#include "../config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../utils/helper_math.h"
#include "../../utils/my_math.hpp"
#include "../params/per_ray_data.cuh"
#include "shader_common.cuh"
#include "m1/bounded_trig.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

#include <optix_device.h>  
#ifdef __CUDACC__
#include <cuda_fp16.h>      // half を使うなら（OptixCoopVec<half, N> 等）
#endif

static __forceinline__ __device__
float evalPhaseFunctionHG
(const float cosTheta, float g)
{
    g = fminf(fmaxf(g, -0.999999f), 0.999999f);
    float cos = fminf(fmaxf(cosTheta, -1.0f), 1.0f);

    const float gg = g * g;
    const float denom = 1.0f + gg - 2.0f * g * cos;
    const float denom32 = denom * sqrtf(fmaxf(denom, 1e-20f));

    return (1.0f - gg) / (4.0f * m1::pi * fmaxf(denom32, 1e-20f));
}


struct PhaseSample
{
    float3  wi;
    float   pdf;
};


static __forceinline__ __device__
PhaseSample samplePhaseFunctionHG
(const float3& woWorld, float u1, float u2, float g)
{
    PhaseSample ps;
    const float3 wo = normalize(woWorld);
    
    g = fminf(fmaxf(g, -0.999999f), 0.999999f);
    u1 = fminf(fmaxf(u1, 0.0f), 0.99999994f);
    
    float cosTheta;
    if (fabsf(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * u1; 
    } else {
        const float gg = g * g;
        const float t  = (1.0f - gg) / (1.0f - g + 2.0f * g * u1);
        cosTheta = (1.0f + gg - t * t) / (2.0f * g);
        cosTheta = fminf(fmaxf(cosTheta, -1.0f), 1.0f);
    }

    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    const float phi = 2.0f * m1::pi * u2;
    const auto phiAngle = m1::bounded_sincos(phi);
    const float sinPhi = phiAngle.sine, cosPhi = phiAngle.cosine;

    // wo を z 軸にしたローカル方向
    const float3 local =
        make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);

    // world へ
    float3 t, b;
    makeONB(wo, t, b);
    ps.wi = normalize(local.x * t + local.y * b + local.z * wo);

    const float pdf = evalPhaseFunctionHG(cosTheta, g);
    ps.pdf   = pdf;

    return ps;
}

static __forceinline__ __device__
int u01_to_index(float u, int N)
{
    u = fminf(fmaxf(u, 0.0f), 0.99999994f);
    int i = (int)floorf(u * (float)N);
    if (i < 0) i = 0;
    if (i > N - 1) i = N - 1;
    return i;
}

static __forceinline__ __device__
float tex3D_point_at_index(cudaTextureObject_t tex, int it, int il, int id,
                           int Ntheta, int Nlambda, int Nd)
{
    // point filter + normalizedCoords=1 前提：テクセル中心
    const float u = ((float)it + 0.5f) / (float)Ntheta;
    const float v = ((float)il + 0.5f) / (float)Nlambda;
    const float w = ((float)id + 0.5f) / (float)Nd;
    return tex3D<float>(tex, u, v, w);
}

struct alignas(16) MiePhaseTex
{
    cudaTextureObject_t pdfTex; // 3D: (theta, lambda, diam)
    cudaTextureObject_t cdfTex; // 3D: (theta, lambda, diam)
    int Ntheta;
    int Nlambda;
    int Nd;
    int _pad0;
};

static __forceinline__ __device__
int lowerBoundCDFTex(const MiePhaseTex T, int il, int id, float u)
{
    int low = 0, high = T.Ntheta - 1;
    while (low < high) {
        int mid = (low + high) >> 1;
        float c = tex3D_point_at_index(T.cdfTex, mid, il, id, T.Ntheta, T.Nlambda, T.Nd);
        if (u <= c) high = mid;
        else        low  = mid + 1;
    }
    return low;
}

static __forceinline__ __device__
float theta_from_index(float idx01, int Ntheta)
{
    // idx01 は [0, Ntheta-1] の実数
    float t01 = idx01 / (float)(Ntheta - 1);
    return t01 * m1::pi;
}

static __forceinline__ __device__
PhaseSample samplePhaseFunctionTabulated(const float3& woWorld,
                                        float u1, float u2,
                                        float u_d, float u_lambda,
                                        const MiePhaseTex T)
{
    PhaseSample ps{};
    const float3 wo = normalize(woWorld);

    // u クランプ（u==1 を避ける）
    u1 = fminf(fmaxf(u1, 0.0f), 0.99999994f);
    u2 = fminf(fmaxf(u2, 0.0f), 0.99999994f);

    // まずは nearest スライス
    const int id = u01_to_index(u_d,      T.Nd);
    const int il = u01_to_index(u_lambda, T.Nlambda);

    // CDF 逆変換：it1 は cdf[it1] >= u1 の最小 index
    const int it1 = lowerBoundCDFTex(T, il, id, u1);
    const int it0 = max(it1 - 1, 0);

    const float c0 = tex3D_point_at_index(T.cdfTex, it0, il, id, T.Ntheta, T.Nlambda, T.Nd);
    const float c1 = tex3D_point_at_index(T.cdfTex, it1, il, id, T.Ntheta, T.Nlambda, T.Nd);

    float t = 0.0f;
    const float denom = c1 - c0;
    if (denom > 0.0f) t = (u1 - c0) / denom;
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    // it = it0 + t（連続index）
    const float itf   = (float)it0 + t;
    const float theta = theta_from_index(itf, T.Ntheta);

    // CDF interpolation gives theta in [0, pi]. Preserve ordinary float
    // trig accuracy while excluding libdevice's huge-argument reduction.
    const auto thetaAngle = m1::bounded_sincos(theta);
    const float sinTheta = thetaAngle.sine, cosTheta = thetaAngle.cosine;

    // φ は一様
    const float phi = 2.0f * m1::pi * u2;
    const auto phiAngle = m1::bounded_sincos(phi);
    const float sinPhi = phiAngle.sine, cosPhi = phiAngle.cosine;

    // wo を z 軸にしたローカル方向（z=cosθ）
    const float3 local = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);

    // world へ
    float3 tvec, bvec;
    makeONB(wo, tvec, bvec);
    ps.wi = normalize(local.x * tvec + local.y * bvec + local.z * wo);

    // PDF 評価：pdfTex から it0/it1 を取って自前で線形補間（cdfから得た t を使う）
    const float p0 = tex3D_point_at_index(T.pdfTex, it0, il, id, T.Ntheta, T.Nlambda, T.Nd);
    const float p1 = tex3D_point_at_index(T.pdfTex, it1, il, id, T.Ntheta, T.Nlambda, T.Nd);
    ps.pdf = fmaxf(0.0f, (1.0f - t) * p0 + t * p1); // sr^-1

    return ps;
}

// 評価だけ（cosθ -> p(θ)）
static __forceinline__ __device__
float evalPhaseFunctionTabulated(float cosTheta,
                                float u_d, float u_lambda,
                                const MiePhaseTex T)
{
    cosTheta = fminf(fmaxf(cosTheta, -1.0f), 1.0f);
    const float theta = acosf(cosTheta); // 0..pi

    // theta -> 連続index [0, Ntheta-1]
    const float itf = (theta / m1::pi) * (float)(T.Ntheta - 1);
    int it0 = (int)floorf(itf);
    int it1 = min(it0 + 1, T.Ntheta - 1);
    float t = itf - (float)it0;

    const int id = u01_to_index(u_d,      T.Nd);
    const int il = u01_to_index(u_lambda, T.Nlambda);

    const float p0 = tex3D_point_at_index(T.pdfTex, it0, il, id, T.Ntheta, T.Nlambda, T.Nd);
    const float p1 = tex3D_point_at_index(T.pdfTex, it1, il, id, T.Ntheta, T.Nlambda, T.Nd);
    return fmaxf(0.0f, (1.0f - t) * p0 + t * p1);
}

#endif
