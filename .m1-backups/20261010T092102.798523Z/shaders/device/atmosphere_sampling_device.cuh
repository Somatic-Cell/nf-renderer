#pragma once
#include "../../utils/helper_math.h"
#include "../../utils/my_math.hpp"
#include "../config.cuh"
#include "shader_common.cuh"


#include <cuda_runtime.h>
#include <math_constants.h>
#include <optix.h>


namespace atmo {

constexpr float kPi      = CUDART_PI_F;
constexpr float kInv4Pi  = 1.0f / (4.0f * CUDART_PI_F);

struct SkySamplingConfig
{
    float sunAngularRadiusRad = 0.004675f; // 約 0.267 deg
    float vmfKappa            = 128.0f;

    float weightSunDisk       = 0.0f;
    float weightSunVMF        = 0.85f;
    float weightUniformSphere = 0.15f;

    float groundAlbedo        = 0.03f;
    int   renderLowerGround   = 1; // miss で下半球地面も描く
};

struct SkyLightSample
{
    float3 dir        = make_float3(0.0f, 1.0f, 0.0f);
    float  pdf        = 0.0f; // mixture pdf
    int    component  = -1;   // 0=sun disk, 1=vmf, 2=uniform sphere
};

__device__ __forceinline__ float clamp01(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __forceinline__ float clampSigned(float x)
{
    return fminf(fmaxf(x, -1.0f), 1.0f);
}

__device__ __forceinline__ float safeRcp(float x)
{
    return (fabsf(x) > 1.0e-20f) ? (1.0f / x) : 0.0f;
}

__device__ __forceinline__ const cudaTextureObject_t* texHandlePtr(CUdeviceptr p)
{
    return reinterpret_cast<const cudaTextureObject_t*>(p);
}

__device__ __forceinline__ float linearTexCoord01(float x01, uint32_t n)
{
    x01 = clamp01(x01);
    if(n <= 1u) return 0.5f;
    return (((float)(n - 1u) * x01) + 0.5f) / (float)n;
}

__device__ __forceinline__ float clampNuValid(float mu, float muS, float nu)
{
    mu  = clampSigned(mu);
    muS = clampSigned(muS);
    nu  = clampSigned(nu);

    const float sinTheta  = sqrtf(fmaxf(0.0f, 1.0f - mu  * mu));
    const float sinThetaS = sqrtf(fmaxf(0.0f, 1.0f - muS * muS));

    const float lo = mu * muS - sinTheta * sinThetaS;
    const float hi = mu * muS + sinTheta * sinThetaS;
    return fminf(fmaxf(nu, lo), hi);
}

struct LambdaInterval01
{
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    float    t  = 0.0f;
};

__device__ __forceinline__
LambdaInterval01 findLambdaInterval01(float lambda01, uint32_t lambdaCount)
{
    LambdaInterval01 out{};

    if(lambdaCount <= 1u) {
        out.i0 = 0;
        out.i1 = 0;
        out.t  = 0.0f;
        return out;
    }

    const float x = fminf(fmaxf(lambda01, 0.0f), 1.0f) * float(lambdaCount - 1u);
    const float xf = floorf(x);

    out.i0 = min((uint32_t)xf, lambdaCount - 1u);
    out.i1 = min(out.i0 + 1u, lambdaCount - 1u);
    out.t  = x - xf;

    if(out.i0 == out.i1) {
        out.t = 0.0f;
    }
    return out;
}

__device__ __forceinline__ float skyMu01(float mu)
{
    return 0.5f * (clampSigned(mu) + 1.0f);
}

__device__ __forceinline__ float skyMuS01(float muS, float muSMin)
{
    return clamp01((muS - muSMin) / fmaxf(1.0f - muSMin, 1.0e-6f));
}

__device__ __forceinline__ float skyNu01(float nu)
{
    return 0.5f * (clampSigned(nu) + 1.0f);
}

// GPU 側 sky texture の軸は
// x = nu, y = mu, z = muS
// という前提
// host upload 側がこの軸順に直っていることが前提
__device__ __forceinline__ float sampleSkyFamilyNoPhase(
    CUdeviceptr handleBuffer,
    const AtmosphereDeviceData& atmo,
    float mu,
    float muS,
    float nu,
    float lambdaNormalized)
{
    if(handleBuffer == 0 || atmo.lambdaCount == 0) return 0.0f;

    const auto li = findLambdaInterval01(lambdaNormalized, atmo.lambdaCount);
    const cudaTextureObject_t* handles = texHandlePtr(handleBuffer);

    const float nuValid = clampNuValid(mu, muS, nu);
    const float x = linearTexCoord01(skyNu01(nuValid),  atmo.skyNu);
    const float y = linearTexCoord01(skyMu01(mu),  atmo.skyMu);
    const float z = linearTexCoord01(skyMuS01(muS, atmo.muSMin), atmo.skyMuS);

    const float v0 = tex3D<float>(handles[li.i0], x, y, z);
    const float v1 = tex3D<float>(handles[li.i1], x, y, z);
    return lerp(v0, v1, li.t);
}

__device__ __forceinline__ float phaseRayleigh(float nu)
{
    return (3.0f / (16.0f * kPi)) * (1.0f + nu * nu);
}

__device__ __forceinline__ float phaseCornetteShanks(float nu, float g)
{
    const float g2 = g * g;
    const float denomBase = 1.0f + g2 - 2.0f * g * nu;
    const float denom = powf(fmaxf(1.0e-6f, denomBase), 1.5f);
    const float coeff = (3.0f / (8.0f * kPi)) * ((1.0f - g2) / (2.0f + g2));
    return coeff * (1.0f + nu * nu) / denom;
}

// directIrradianceTex は
// x = muS, y = r, z = lambda
// を想定
__device__ __forceinline__ float sampleDirectIrradiance(
    const AtmosphereDeviceData& atmo,
    float r_m,
    float muS,
    float lambda)
{
    if(atmo.directIrradianceTex == 0 || atmo.lambdaCount == 0) {
        return 0.0f;
    }

    const float bottom = atmo.bottomRadius_m;
    const float top    = atmo.topRadius_m;

    const float H   = sqrtf(fmaxf(0.0f, top * top - bottom * bottom));
    const float rho = sqrtf(fmaxf(0.0f, r_m * r_m - bottom * bottom));
    const float w = fminf(fmaxf(lambda, 0.0f), 1.0f);

    const float u = skyMuS01(muS, atmo.muSMin);
    const float v = (H > 0.0f) ? clamp01(rho / H) : 0.0f;

    const float x = linearTexCoord01(u, atmo.irradianceMuS);
    const float y = linearTexCoord01(v, atmo.irradianceR);
    const float z = linearTexCoord01(w, atmo.lambdaCount);

    return tex3D<float>(atmo.directIrradianceTex, x, y, z);
}

__device__ __forceinline__ float sampleSunTransmittance(
    const AtmosphereDeviceData& atmo,
    float muS,
    float lambda)
{
    if(atmo.sunTransmittanceTex == 0 || atmo.lambdaCount == 0) {
        return 1.0f;
    }

    const float w = fminf(fmaxf(lambda, 0.0f), 1.0f);

    const float x = linearTexCoord01(skyMuS01(muS, atmo.muSMin), atmo.skyMuS);
    const float y = linearTexCoord01(w, atmo.lambdaCount);

    return tex2D<float>(atmo.sunTransmittanceTex, x, y);
}

__device__ __forceinline__ bool intersectGroundSphereFixedObserver(
    const AtmosphereDeviceData& atmo,
    const float3& viewDir,
    float3& outGroundPos,
    float3& outGroundNormal)
{
    const float rObserver = atmo.bottomRadius_m + atmo.observerAltitude_m;
    const float R = atmo.bottomRadius_m;

    const float3 origin = make_float3(0.0f, rObserver, 0.0f);

    const float b = dot(origin, viewDir);
    const float c = dot(origin, origin) - R * R;
    const float disc = b * b - c;
    if(disc < 0.0f) {
        return false;
    }

    const float s = sqrtf(fmaxf(0.0f, disc));
    const float t0 = -b - s;
    const float t1 = -b + s;
    const float t = (t0 > 0.0f) ? t0 : t1;
    if(t <= 0.0f) {
        return false;
    }

    outGroundPos = origin + viewDir * t;
    outGroundNormal = normalize(outGroundPos);
    return true;
}

__device__ __forceinline__ float evalGroundRadianceLambertSpectral(
    const AtmosphereDeviceData& atmo,
    const SkySamplingConfig& cfg,
    const float3& groundNormal,
    const float3& sunDir,
    float lambdaNm)
{
    const float muSLocal = fmaxf(0.0f, dot(groundNormal, sunDir));
    if(muSLocal <= 0.0f) {
        return 0.0f;
    }

    const float E_direct = sampleDirectIrradiance(
        atmo,
        atmo.bottomRadius_m,
        muSLocal,
        lambdaNm);

    return E_direct * muSLocal * (cfg.groundAlbedo / kPi);
}

__device__ __forceinline__ float sunSolidAngle(float angularRadiusRad)
{
    return 2.0f * kPi * (1.0f - cosf(angularRadiusRad));
}

__device__ __forceinline__ float 
evalSunDiskRadianceSpectral(
    const AtmosphereDeviceData& atmo,
    cudaTextureObject_t am0Tex,
    float waveLengthNormalized,
    const float3& viewDir,
    const float3& sunDir,
    float sunAngularRadiusRad)
{
    const float cosTheta = dot(viewDir, sunDir); // nu?
    const float cosMax = cosf(sunAngularRadiusRad); 
    if(cosTheta < cosMax) {
        return 0.0f;
    }

    const float AM0 = tex2D<float>(am0Tex, waveLengthNormalized, 0.5f);
    const float muS = sunDir.y; // fixed observer model
    const float T = sampleSunTransmittance(atmo, muS, waveLengthNormalized);
    const float omegaSun = fmaxf(sunSolidAngle(sunAngularRadiusRad), 1.0e-8f);

    // AM0 を spectral irradiance とみなし，円盤 radiance に変換
    return AM0 * T / omegaSun;
}

__device__ __forceinline__ float evalSkyRadianceSpectralFixedObserver(
    const AtmosphereDeviceData& atmo,
    const float3& viewDir,
    const float3& sunDir,
    float lambdaNmNormalized)
{
    const float mu  = viewDir.y;
    const float muS = sunDir.y;
    const float nu  = dot(viewDir, sunDir);
    const float theta  = acosf(nu) / kPi;

    const float phaseR = phaseRayleigh(nu);
    // const float phaseM = phaseCornetteShanks(nu, 0.85f);
    const float phaseM = tex3D<float>(optixLaunchParams.mieTexture.pdf, theta, lambdaNmNormalized, 0.1f);

    float L = 0.0f;
    L += phaseR * sampleSkyFamilyNoPhase(atmo.skyRayleighTexHandles, atmo, mu, muS, nu, lambdaNmNormalized);
    L += phaseM * sampleSkyFamilyNoPhase(atmo.skyMieTexHandles,      atmo, mu, muS, nu, lambdaNmNormalized);
    L +=          sampleSkyFamilyNoPhase(atmo.skyMultipleTexHandles, atmo, mu, muS, nu, lambdaNmNormalized);
    return fmaxf(L, 0.0f);
}

__device__ __forceinline__ float evalSkyMissSpectralFixedObserver(
    const AtmosphereDeviceData& atmo,
    const SkySamplingConfig& cfg,
    cudaTextureObject_t am0Tex,
    float wavelengthNormalized,
    const float3& viewDir,
    const float3& sunDir)
{
    float L = evalSkyRadianceSpectralFixedObserver(
        atmo, viewDir, sunDir, wavelengthNormalized);

    // 下半球地面
    if(cfg.renderLowerGround && viewDir.y < 0.0f) {
        float3 gp, gn;
        if(intersectGroundSphereFixedObserver(atmo, viewDir, gp, gn)) {
            L += evalGroundRadianceLambertSpectral(atmo, cfg, gn, sunDir, wavelengthNormalized);
        }
    }

    // 太陽円盤
    // L += evalSunDiskRadianceSpectral(
    //     atmo, am0Tex, wavelengthNormalized,
    //     viewDir, sunDir, cfg.sunAngularRadiusRad);

    return fmaxf(L, 0.0f);
}

__device__ __forceinline__ float pdfUniformSphere()
{
    return kInv4Pi;
}

__device__ __forceinline__ float pdfSunDiskUniform(
    const float3& dir,
    const float3& sunDir,
    float sunAngularRadiusRad)
{
    const float cosMax = cosf(sunAngularRadiusRad);
    const float cosTheta = dot(dir, sunDir);
    if(cosTheta < cosMax) {
        return 0.0f;
    }

    const float omega = fmaxf(sunSolidAngle(sunAngularRadiusRad), 1.0e-8f);
    return 1.0f / omega;
}

__device__ __forceinline__ float pdfVMF(
    const float3& dir,
    const float3& muDir,
    float kappa)
{
    if(kappa <= 1.0e-5f) {
        return kInv4Pi;
    }

    const float mu = clampSigned(dot(dir, muDir));
    const float denom = 4.0f * kPi * sinhf(kappa);
    const float C = (denom > 0.0f) ? (kappa / denom) : kInv4Pi;
    return C * expf(kappa * mu);
}

__device__ __forceinline__ float normalizedProposalWeightSum(const SkySamplingConfig& cfg)
{
    return fmaxf(cfg.weightSunDisk + cfg.weightSunVMF + cfg.weightUniformSphere, 1.0e-8f);
}

__device__ __forceinline__ float evalSkyEmitterMixturePdf(
    const SkySamplingConfig& cfg,
    const float3& dir,
    const float3& sunDir)
{
    const float wsum = normalizedProposalWeightSum(cfg);
    const float wSun = cfg.weightSunDisk       / wsum;
    const float wVmf = cfg.weightSunVMF        / wsum;
    const float wUni = cfg.weightUniformSphere / wsum;

    return
        wSun * pdfSunDiskUniform(dir, sunDir, cfg.sunAngularRadiusRad) +
        wVmf * pdfVMF(dir, sunDir, cfg.vmfKappa) +
        wUni * pdfUniformSphere();
}

__device__ __forceinline__ float3 sampleUniformSphere(float u1, float u2)
{
    const float z   = 1.0f - 2.0f * u1;
    const float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * kPi * u2;
    return make_float3(r * cosf(phi), z, r * sinf(phi));
}

__device__ __forceinline__ float3 sampleSunDiskUniform(
    const float3& sunDir,
    float sunAngularRadiusRad,
    float u1,
    float u2)
{
    const float cosThetaMax = cosf(sunAngularRadiusRad);
    const float cosTheta = lerp(1.0f, cosThetaMax, u1);
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * kPi * u2;

    const float x = sinTheta * cosf(phi);
    const float y = sinTheta * sinf(phi);
    const float z = cosTheta;
    float3 t, b;
    makeONB(sunDir, t, b);
    return localToWorld(make_float3(x, y, z), t, b, sunDir);
}

__device__ __forceinline__ float3 sampleVMF(
    const float3& muDir,
    float kappa,
    float u1,
    float u2)
{
    if(kappa <= 1.0e-5f) {
        return sampleUniformSphere(u1, u2);
    }

    // 安定な逆変換
    const float e = expf(-2.0f * kappa);
    const float cosTheta = 1.0f + logf(u1 + (1.0f - u1) * e) / kappa;
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * kPi * u2;

    const float x = sinTheta * cosf(phi);
    const float y = sinTheta * sinf(phi);
    const float z = cosTheta;
    float3 t, b;
    makeONB(muDir, t, b);
    return localToWorld(make_float3(x, y, z), t, b, muDir);
}

__device__ __forceinline__ SkyLightSample sampleSkyEmitterMixture(
    const SkySamplingConfig& cfg,
    const float3& sunDir,
    float uChoose,
    float u1,
    float u2)
{
    SkyLightSample s{};

    const float wsum = normalizedProposalWeightSum(cfg);
    const float wSun = cfg.weightSunDisk       / wsum;
    const float wVmf = cfg.weightSunVMF        / wsum;
    // const float wUni = cfg.weightUniformSphere / wsum;

    if(uChoose < wSun) {
        s.component = 0;
        s.dir = sampleSunDiskUniform(sunDir, cfg.sunAngularRadiusRad, u1, u2);
    }
    else if(uChoose < wSun + wVmf) {
        s.component = 1;
        s.dir = sampleVMF(sunDir, cfg.vmfKappa, u1, u2);
    }
    else {
        s.component = 2;
        s.dir = sampleUniformSphere(u1, u2);
    }

    s.pdf = evalSkyEmitterMixturePdf(cfg, s.dir, sunDir);
    return s;
}

__device__ __forceinline__
float3 sunDirFromAngles(float zenithRad, float azimuthRad)
{
    const float sinTheta = sinf(zenithRad);
    const float cosTheta = cosf(zenithRad);

    const float x = sinTheta * sinf(azimuthRad);
    const float y = cosTheta;
    const float z = sinTheta * cosf(azimuthRad);
    return make_float3(x, y, z);
}

} // namespace atmo