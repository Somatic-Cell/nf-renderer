#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/atmosphere_sampling_device.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include "../../include/atmosphere_lut.h"
#include <stdio.h>

// radiance ray が交差しない = 環境マップにあたる
extern "C" __global__ void __miss__radiance_rgb()
{
    PRDRGB &prd = *getPRD<PRDRGB>(); 
    
    float u, v;
    orthogonalToUVCoord(prd.wi, &u, &v);
    float3 emission =  make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    
    
    if(prd.bounce == 0){
        prd.contribution += prd.albedo * emission;
        prd.continueTrace = false;
        return;
    }

    const int wp = optixLaunchParams.envMapInfo.patchSize.x;
    const int hp = optixLaunchParams.envMapInfo.patchSize.y;

    int col = min(max(int(u * (float)wp), 0), wp -1);
    int row = min(max(int(v * (float)hp), 0), hp -1);

    const int patchIndex = row * wp + col;

    float th0, th1, ph0, ph1;
    thetaPhiFromPatch(row, col, wp, hp, &th0, &th1, &ph0, &ph1);

    const float deltaPhi = ph1 - ph0;
    const float deltaCosT = cosf(th0) - cosf(th1);
    const float dOmega = deltaPhi *deltaCosT;

    const float totalWeight = optixLaunchParams.envMapInfo.totalWeight;
    float pdfEnvDir = 0.0f;

    if(totalWeight > 0.0f && dOmega > 0.0f){
        const float pPatch = optixLaunchParams.envMapInfo.patchWeight[patchIndex] / totalWeight;
        pdfEnvDir = pPatch * (1.0f / dOmega);
    }

    const float pSelected = 1.0f / float(optixLaunchParams.light.numLights);
    const float pdfLight = pdfEnvDir * pSelected;

    const float weight = balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
    emission *= weight; 
    prd.contribution += emission * prd.albedo;
    prd.continueTrace = false;
}


extern "C" __global__ void __miss__radiance_spectral()
{
    PRDSpectral &prd = *getPRD<PRDSpectral>(); 
    
    float u, v;
    orthogonalToUVCoord(prd.wi, &u, &v);
    float3 emissionRGB =  make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    float emission = upSamplingFromRGB(emissionRGB, prd);
    const float D65 = tex2D<float>(optixLaunchParams.spectral.D65, prd.waveLengthNormalized, 0.5f);

    
    if(prd.bounce == 0){
        prd.contribution += prd.albedo * emission * D65;
        prd.continueTrace = 0;
        return;
    }

    const int wp = optixLaunchParams.envMapInfo.patchSize.x;
    const int hp = optixLaunchParams.envMapInfo.patchSize.y;

    int col = min(max(int(u * (float)wp), 0), wp -1);
    int row = min(max(int(v * (float)hp), 0), hp -1);

    const int patchIndex = row * wp + col;

    float th0, th1, ph0, ph1;
    thetaPhiFromPatch(row, col, wp, hp, &th0, &th1, &ph0, &ph1);

    const float deltaPhi = ph1 - ph0;
    const float deltaCosT = cosf(th0) - cosf(th1);
    const float dOmega = deltaPhi *deltaCosT;

    const float totalWeight = optixLaunchParams.envMapInfo.totalWeight;
    float pdfEnvDir = 0.0f;

    if(totalWeight > 0.0f && dOmega > 0.0f){
        const float pPatch = optixLaunchParams.envMapInfo.patchWeight[patchIndex] / totalWeight;
        pdfEnvDir = pPatch * (1.0f / dOmega);
    }

    const float pSelected = 1.0f / float(optixLaunchParams.light.numLights);
    const float pdfLight = pdfEnvDir * pSelected;

    const float weight = balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
    emission *= weight; 
    prd.contribution += emission * prd.albedo * D65;
    prd.continueTrace = 0;
}

extern "C" __global__ void __miss__radiance_noEnvMap_rgb()
{
    PRDRGB &prd = *getPRD<PRDRGB>();
    prd.continueTrace = false;
    // if(prd.bounce == 0){
    //     prd.primaryAlbedo = make_float3(0.0f);
    //     prd.primaryNormal = - prd.wi; 
    // }
}

extern "C" __global__ void __miss__radiance_noEnvMap_spectral()
{
    PRDSpectral &prd = *getPRD<PRDSpectral>();
    prd.continueTrace = 0;
    // if(prd.bounce == 0){
    //     prd.primaryAlbedo = make_float3(0.0f);
    //     prd.primaryNormal = - prd.wi; 
    // }
}


extern "C" __global__ void __miss__radiance_sky_spectral()
{
    PRDSpectral &prd = *getPRD<PRDSpectral>();
    const AtmosphereDeviceData& atmo = optixLaunchParams.atmo;

    const float3 viewDir = prd.wi;
    const float3 sunDir = atmo::sunDirFromAngles(
        optixLaunchParams.sunParams.sunZenithRad,
        optixLaunchParams.sunParams.sunAzimuthRad
    );

    // const float misWeight = hwssSpectralWeight(prd.logPOrefix);
    const atmo::SkySamplingConfig config;
    const float uNormalized = prd.waveLengthNormalized;
    // constexpr int C = 4;
    // const float invC = 1.0f / (float)C;
    float weight = 1.0f;
    
    const float L = atmo::evalSkyMissSpectralFixedObserver(
        atmo,
        config,
        optixLaunchParams.spectral.D65,
        uNormalized,
        viewDir,
        sunDir
    );

    if(prd.bounce > 0){
        const float pdfLight = atmo::evalSkyEmitterMixturePdf(
            config,
            viewDir,
            sunDir
        );

        weight = balanceHeuristicWeight(
            1, fmaxf(prd.pdf.bxdf,  1.0e-7f),
            1, fmaxf(pdfLight,      1.0e-7f));
    }

    prd.contribution += L * weight / prd.pdf.spectral; // bounce == 0 では prd.bxdf = 1.0f 
    prd.continueTrace = 0;
    return;
}