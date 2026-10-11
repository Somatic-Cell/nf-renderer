#ifndef SHADER_COMMON_CUH_
#define SHADER_COMMON_CUH_

#include "m1/bounded_trig.hpp"
#include <cuda_runtime.h>
#include "../../utils/helper_math.h"
#include "../../utils/my_math.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

struct LensRay
{
    float3 org;
    float3 dir;
    float weight    {1.0f};
};

struct IntersectedData_RGB
{
    float3 baseColor;
    float ior;
    float roughness;
    float metallic;
    float3 wiLocal;
    float3 normal;
};

struct IntersectedData_Spectral
{
    float baseColor;
    float ior;
    float roughness;
    float metallic;
    float3 wiLocal;
    float3 normal;
};

struct LightSample_RGB
{
    float3  position;
    float3  direction;
    float distance      {1e7f};
    float3  emission    {make_float3(0.0f)};
    float   pdf         {0.0f};
};

struct LightSample_Spectral
{
    float3  position;
    float3  direction;
    float distance      {1e7f};
    float3  emissionRGB  {make_float3(0.0f)};
    float   pdf         {0.0f};
};

inline __device__ __host__ void rotate(const float3 _u, const float r, float3& vx, float3& vy, float3& vz)
{
    float3 u = normalize(_u);
    float s = sinf(r);
    float c = cosf(r);
    const float3 rx = make_float3(u.x * u.x + (1.f - u.x * u.x) * c,   u.x * u.y * (1.f - c) - u.z * s,    u.x * u.z * (1.f - c) + u.y * s);
    const float3 ry = make_float3(u.x * u.y * (1.f - c) + u.z * s,     u.y * u.y + (1.f - u.y * u.y) * c,  u.y * u.z * (1.f - c) - u.x * s);
    const float3 rz = make_float3(u.x * u.z * (1.f - c) * u.y * s,     u.y * u.z * (1.f - c) + u.x * s,    u.z * u.z + (1.f - u.z * u.z) * c);

    // | rx.x  ry.x  rz.x |   | vx.x  vy.x  vz.x |
    // | rx.y  ry.y  rz.y | * | vx.y  vy.y  vz.y |
    // | rx.z  ry.z  rz.z |   | vx.z  vy.z  vz.z |
    //
    const float3 tmpx = make_float3(rx.x * vx.x + ry.x * vx.y + rz.x * vx.z, rx.x * vy.x + ry.x * vy.y + rz.x * vy.z, rx.x * vz.x + ry.x * vz.y + rz.x * vz.z);
    const float3 tmpy = make_float3(rx.y * vx.x + ry.y * vx.y + rz.y * vx.z, rx.y * vy.x + ry.y * vy.y + rz.y * vy.z, rx.y * vz.x + ry.y * vz.y + rz.y * vz.z);
    const float3 tmpz = make_float3(rx.z * vx.x + ry.z * vx.y + rz.z * vx.z, rx.z * vy.x + ry.z * vy.y + rz.z * vy.z, rx.z * vz.x + ry.z * vz.y + rz.z * vz.z);

    vx = tmpx;
    vy = tmpy;
    vz = tmpz;
}

inline __device__ __host__ float3 random_unit_hemisphere(float rnd1, float rnd2)
{
    float a = rnd1 * 2.f * m1::pi;
    float z = rnd2;
    float r = fmaxf(0.0f, sqrtf(1.f - z * z));
    const auto angle = m1::bounded_sincos(a); // rnd1 is in [0,1).
    return make_float3(r * angle.cosine, z, r * angle.sine);
}

inline __device__ __host__ float3 random_unit_sphere(float rnd1, float rnd2)
{
    float3 p;
    float costheta, phi;
    float sintheta;

    // 球上の一点をサンプリング（極座標表現）
    costheta = 2.f * rnd1 - 1.f; // 0 < theta < pi
    phi = 2.f * m1::pi * rnd2;

    sintheta = sqrtf(1.f - costheta * costheta);

    const auto angle = m1::bounded_sincos(phi); // rnd2 is in [0,1).
    p.x = sintheta * angle.cosine;
    p.y = sintheta * angle.sine;
    p.z = costheta;

    return p;
}

inline __device__ __host__ float2 random_unit_disk(float rnd1, float rnd2)
{
    float theta = 2.0f * m1::pi * rnd1;
    float r = sqrtf(rnd2);
    const auto angle = m1::bounded_sincos(theta);
    float2 p = r * make_float2(angle.cosine, angle.sine);
    return p;
}

inline __device__ __host__ float3 randomCosineHemisphere(float rnd1, float rnd2)
{
    const float r = sqrtf(rnd1);
    const float phi = rnd2 * 2.f * m1::pi;
    const auto angle = m1::bounded_sincos(phi); // rnd2 is in [0,1).
    const float x = r * angle.cosine;
    const float z = r * angle.sine;
    const float y = fmaxf(0.0f, sqrtf(1.f - x * x - z * z));
    return make_float3(x, y, z);
}

// 基底変換
inline __device__ __host__ float3 localToWorld(const float3 v, const float3 localX, const float3 localY, const float3 localZ) {
    return normalize(v.x * localX + v.y * localY + v.z * localZ);
}

inline __device__ __host__ float3 worldToLocal(const float3 v, const float3 localX, const float3 localY, const float3 localZ) {
    return normalize(make_float3(dot(v, localX), dot(v, localY), dot(v, localZ)));
}

// 球面座標
inline __device__ __host__ void orthogonalToUVCoord(const float3 dir, float* u, float* v) {
    // const float3 dir = normalize(_dir);
    float phi = atan2f(dir.z, dir.x);
    if(phi < 0.0f) phi += 2.0f * m1::pi;
    float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
    *u = phi / (2.f * m1::pi);
    *v = theta / m1::pi;
}


inline __device__ __host__ float3 sphericalToOrthogonalCoord(const float theta, const float phi) {
    // Environment-patch coordinates: theta in [0,pi], phi in [0,2*pi].
    const auto polar = m1::bounded_sincos(theta);
    const auto azimuth = m1::bounded_sincos(phi);
    return make_float3(azimuth.cosine * polar.sine, polar.cosine,
                      azimuth.sine * polar.sine);
}


// wi を返す
// inline __device__ float3 cosineSampling(const float u, const float v)
// {
//     const float theta = 0.5f * acosf(1.0f - 2.0f * u);
//     const float phi = 2.0f * M_PI * v;

//     const float sinTheta    = sinf(theta);
//     const float cosTheta    = cosf(theta);
//     const float sinPhi      = sinf(phi);
//     const float cosPhi      = cosf(phi);
//     return make_float3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
// }

// wm を返す
inline __device__ float3 visibleNormalSampling( const float alpha, 
                                                const float3 wo, 
                                                const float rnd1, 
                                                const float rnd2)
{
    float3 Vh = normalize(make_float3(alpha * wo.x, wo.y, alpha * wo.z));

    float3 normal = make_float3(0.f, 1.f, 0.f);
    if (Vh.y > 0.99f) {
        normal = make_float3(0.f, 0.f, -1.f);
    }

    float3 T1 = normalize(cross(Vh, normal));
    float3 T2 = cross(T1, Vh);

    float r = sqrtf(fmaxf(rnd1, 0.f));
    float phi = 2.f * m1::pi * rnd2;
    const auto angle = m1::bounded_sincos(phi); // rnd2 is in [0,1).
    float t1 = r * angle.cosine;
    float t2 = r * angle.sine;
    float s = 0.5f * (1.f + Vh.y);
    t2 = (1.f - s) * sqrtf(fmaxf(1.f - t1 * t1, 0.f)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(1e-7f, 1.f - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = normalize(make_float3(alpha * Nh.x, Nh.y, alpha * Nh.z));
    return Ne;
}

inline __device__ float3 evalLambertBRDF_RGB(const float3 baseColor, const float3 wo, const float3 wi)
{
    return baseColor / m1::pi;
}

inline __device__ float evalLambertBRDF_Spectral(const float baseColor, const float3 wo, const float3 wi)
{
    return baseColor / m1::pi;
}

inline __device__ float getLambertPdf(const float3 wo, const float3 wi){
    return wi.y / m1::pi;
}

inline __device__ float ggx_D(const float alpha, const float3 wm)
{
    const float t = wm.y * wm.y + (wm.x * wm.x + wm.z * wm.z) / (alpha * alpha);
    return 1.f / (m1::pi * alpha * alpha * t * t);
}

inline __device__ float lambda(const float alpha, const float3 w)
{
    float vy = fmaxf(w.y, 1e-5f);
    float t = alpha * alpha * (w.x * w.x + w.z * w.z) / (vy * vy);
    return (-1.f + sqrtf(1.f + t)) / 2.f;
}

inline __device__ float smith_G1(const float alpha, const float3 w)
{
    return 1.0f / (1.0f + lambda(alpha, w));
}

inline __device__ float smith_G2(const float alpha, const float3 wo, const float3 wi)
{
    return 1.0f / (1.0f + lambda(alpha, wo) + lambda(alpha, wi));
}

inline __device__ float3 schlick(const float3 wo, const float3 n, const float3 F0){
    return F0 + (make_float3(1.0f) - F0) * powf(1.f - dot(wo, n), 5);
}

inline __device__ float schlick(const float3 wo, const float3 n, const float F0){
    return F0 + (1.0f - F0) * powf(1.f - dot(wo, n), 5);
}

inline __device__ float3 evalSpecularBRDF_RGB( const float alpha, 
                                        const float3 F0, 
                                        const float3& wo, 
                                        const float3& wi)
{
    float3 wm = normalize(wo + wi);

    float D = ggx_D(alpha, wm);
    float G = smith_G2(alpha, wo, wi);
    float3 F = schlick(wo, wm, F0);

    float in = fmaxf(wi.y, 1e-4f);
    float on = fmaxf(wo.y, 1e-4f);

    float3 brdf = F * G * D / (4.f * in * on + 1e-5f);

    return brdf;
}

inline __device__ float evalSpecularBRDF_Spectral( const float alpha, 
                                        const float F0, 
                                        const float3& wo, 
                                        const float3& wi)
{
    float3 wm = normalize(wo + wi);

    float D = ggx_D(alpha, wm);
    float G = smith_G2(alpha, wo, wi);
    float F = schlick(wo, wm, F0);

    float in = fmaxf(wi.y, 1e-4f);
    float on = fmaxf(wo.y, 1e-4f);

    float brdf = F * G * D / (4.f * in * on + 1e-5f);

    return brdf;
}

inline __device__ float getGGXPdf(const float alpha, const float3 wo, const float3 wi){
    float3 wm = normalize(wo + wi);
    return ggx_D(alpha, wm) * smith_G1(alpha, wo) * fmaxf(dot(wo, wm), 1e-7f) / (fmaxf(wo.y, 1e-7f) * 4.0f * fmaxf(dot(wi, wm), 1e-7f));
}

__forceinline__ __device__ float balanceHeuristicWeight(const int num_a, const float pdf_a, const int num_b, const float pdf_b){
    float pa = (pdf_a > 0.f) ? pdf_a : 0.0f;
    float pb = (pdf_b > 0.f) ? pdf_b : 0.0f;

    const float np_a = (float)num_a * pa;
    const float np_b = (float)num_b * pb;
    const float s = np_a + np_b;
    return (s > 0.f) ? np_a / s : 0.0f;
}

static __forceinline__ __device__ int lowerBoundCDF(const float* __restrict__ cdf, int n, float u)
{
    int low = 0, high = n - 1;
    while (low < high) {
        int mid = (low + high) >> 1;
        if (u <= cdf[mid])  high = mid;
        else                low = mid + 1;
    }
    return low;
}

static __forceinline__ __device__ void thetaPhiFromPatch(
    const int row, 
    const int col,
    const int wp,
    const int hp,
    float* theta0,
    float* theta1,
    float* phi0,
    float* phi1
){
    float u0 = float(col)       / float(wp);
    float u1 = float(col + 1)   / float(wp);
    float v0 = float(row)       / float(hp);
    float v1 = float(row + 1)   / float(hp);
    
    *phi0 = 2.0f * m1::pi * u0;
    *phi1 = 2.0f * m1::pi * u1;
    *theta0 = m1::pi * v0;
    *theta1 = m1::pi * v1;
}

static __forceinline__ __device__ float2 envUVFromSpherical(
    const float theta,
    const float phi
)
{
    float u = phi / (2.0f * m1::pi);
    float v = theta / m1::pi;
    u = u - floorf(u);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return make_float2(u, v);
}

static __forceinline__ __device__ float upSamplingFromRGB(const float3 rgb, const PRDSpectral prd)
{
    // 0-1
    const float functionFetchValue = prd.waveLengthNormalized;
    
    // Sampled upsampling func
    const float upSamplingFuncR = tex2D<float>(optixLaunchParams.spectral.upSampleFunc[0], functionFetchValue, 0.5f);
    const float upSamplingFuncG = tex2D<float>(optixLaunchParams.spectral.upSampleFunc[1], functionFetchValue, 0.5f);
    const float upSamplingFuncB = tex2D<float>(optixLaunchParams.spectral.upSampleFunc[2], functionFetchValue, 0.5f);

    // upsampling
    return rgb.x * upSamplingFuncR + rgb.y * upSamplingFuncG + rgb.z * upSamplingFuncB;
}

static __forceinline__ __device__
bool intersectAABB(
    const float3& origin,
    const float3& direction,
    const float3& bMin, const float3& bMax,
    float tMin, float tMax,
    float& tEnter,  // aabb との交差位置（手前）
    float& tExit    // aabb との交差位置（奥）
)
{
    float t0 = tMin;
    float t1 = tMax;
    auto slab = [&](float oA, float dA, float mn, float mx) -> bool 
    { 
        if (fabsf(dA) < 1e-20f) { // 平行：原点がスラブ外なら不交差 
            return (oA >= mn && oA <= mx); 
        } const float invD = 1.0f / dA; 
        float tNear = (mn - oA) * invD; 
        float tFar  = (mx - oA) * invD;

        if (tNear > tFar) { 
            float tmp = tNear; 
            tNear = tFar; 
            tFar = tmp; 
        } 
        t0 = fmaxf(t0, tNear); 
        t1 = fminf(t1, tFar); 
        return (t0 < t1); 
    };

    if(!slab(origin.x, direction.x, bMin.x, bMax.x)) return false;
    if(!slab(origin.y, direction.y, bMin.y, bMax.y)) return false;
    if(!slab(origin.z, direction.z, bMin.z, bMax.z)) return false;

    float tE = t0;
    float tX = t1;

    if(tX <= tE) return false;

    // レイ区間でクランプ
    tE = fmaxf(tE, tMin);
    tX = fminf(tX, tMax);

    tEnter = tE;
    tExit  = tX;

    return (tEnter < tExit);
}

// Hero Wavelength Spectral Rendering


template<int C>
__device__ __forceinline__ float uLambda(int k, float u0) {
    return fracf(u0 + (float)k / (float)C);   // u_k = frac(u0 + k/C)
}

__device__ __forceinline__ float lambdaFromU(float u, float lambdaMin, float lambdaMax) {
    return lambdaMin + (lambdaMax - lambdaMin) * u;
}

__device__ __forceinline__ float logsumexp4(float4 v) {
    float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    return m + logf(expf(v.x - m) + expf(v.y - m) + expf(v.z - m) + expf(v.w - m));
}

__device__ __forceinline__ float hwssSpectralWeight(float4 logPprefix) {
    float logDen = logsumexp4(logPprefix); // log( (1/C) Σ p_prefix_k )
    if (!isfinite(logDen)) return 0.0f;
    return expf(logPprefix.x - logDen);
}

__device__ __forceinline__ float wrap01(float u)
{
    // [0,1) に折り返す（mod 1）
    u = u - floorf(u);
    // 念のため 1.0 ちょうどを避ける（テクスチャの端サンプル事故対策）
    return fminf(u, nextafterf(1.0f, 0.0f));
}

struct SampledWavelength
{
    float lambda;  // sampled wavelength
    float pdf;     // continuous pdf p(lambda)
    int   bin;     // sampled bin index
};

static __forceinline__ __device__
float clampUnitFloat(float u)
{
    // largest float < 1.0f
    const float oneMinusEps = 0x1.fffffep-1f;
    return fminf(fmaxf(u, 0.0f), oneMinusEps);
}


static __forceinline__ __device__
float evalWavelengthPdf(
    const SpectralParams& spectral,
    float lambdaNormalized)
{
    if (lambdaNormalized < 0.0f || lambdaNormalized >= 1.0f) {
        return 0.0f;
    }

    const float x = lambdaNormalized * float(spectral.wavelengthBinCount);
    int bin = (int)x;
    bin = max(0, min(bin, spectral.wavelengthBinCount - 1));

    const float pmass = fmaxf(spectral.wavelengthPdf[bin], 0.0f);
    return pmass * spectral.wavelengthBinCount;
}

static __forceinline__ __device__
void makeONB(const float3& w, float3& t, float3& b)
{
    if(fabsf(w.z) < 0.999f){
        const float a = 1.0f / sqrtf(fmaxf(1e-20f, 1.0f - w.z * w.z));
        t = make_float3(-w.y * a, w.x * a, 0.0f);
    } else {
        const float a = 1.0f / sqrtf(fmaxf(1e-20f, 1.0f - w.x * w.x));
        t = make_float3(0.0f, -w.z * a, w.y * a);
    }
    b = cross(w, t);
}

static __forceinline__ __device__
float fetchCdf(cudaTextureObject_t tex, int i)
{
    return tex1Dfetch<float>(tex, i);
}

static __forceinline__ __device__
int findCdfBin(cudaTextureObject_t cdfTex, int numBins, float u)
{
    int lo = 0;
    int hi = numBins - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (u >= fetchCdf(cdfTex, mid + 1)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static __forceinline__ __device__
SampledWavelength sampleWavelengthFromCdf(const SpectralParams& spectral, float u)
{
    SampledWavelength result{};
    u = clampUnitFloat(u);

    const int   N       = spectral.wavelengthBinCount;
    const float lambda0 = spectral.wavelengthMin;
    const float dLambda = spectral.wavelengthBinWidth;

    const int   bin   = findCdfBin(spectral.wavelengthCdfTex, N, u);
    const float c0    = fetchCdf(spectral.wavelengthCdfTex, bin);
    const float c1    = fetchCdf(spectral.wavelengthCdfTex, bin + 1);
    const float pmass = fmaxf(c1 - c0, 0.0f);

    float xi = 0.5f;
    if (pmass > 0.0f) xi = (u - c0) / pmass;
    xi = clampUnitFloat(xi);

    const float lamMinBin = lambda0 + float(bin) * dLambda;
    result.lambda = lamMinBin + xi * dLambda;
    result.pdf    = (pmass > 0.0f) ? (pmass / dLambda) : 0.0f;
    result.bin    = bin;
    return result;
}

// static __forceinline__ __device__
// float evalWavelengthPdf(const SpectralParams& spectral, float lambdaNormalized)
// {
//     if (lambdaNormalized < 0.0f || lambdaNormalized >= 1.0f) return 0.0f;

//     const float x = lambdaNormalized * float(spectral.wavelengthBinCount);
//     int bin = (int)x;
//     bin = max(0, min(bin, spectral.wavelengthBinCount - 1));

//     const float c0 = tex1Dfetch<float>(spectral.wavelengthCdfTex, bin);
//     const float c1 = tex1Dfetch<float>(spectral.wavelengthCdfTex, bin + 1);
//     const float pmass = fmaxf(c1 - c0, 0.0f);

//     return pmass * spectral.wavelengthBinCount; // 現行 semantics を維持
// }
#endif // SHADER_COMMON_CUH_
