#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../device/atmosphere_sampling_device.cuh"
#include "../../include/launch_params.h"

extern "C" __device__ LightSample_RGB __direct_callable__light_env_sphere_constant_rgb(LightDefinition light, PRDRGB* prd)
{
    LightSample_RGB lightSample;
    lightSample.pdf = 1.0f /  (4.0f * (float)M_PI) / float(optixLaunchParams.light.numLights);
    lightSample.distance = 1e7f;
    
    const float3 direction = random_unit_sphere(prd->random(), prd->random()); 
    lightSample.direction = direction; 
    float u, v;
    orthogonalToUVCoord(direction, &u, &v);
    lightSample.emission = make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    return lightSample;
}


extern "C" __device__ LightSample_RGB __direct_callable__light_env_sphere_is_rgb(LightDefinition light, PRDRGB* prd)
{
    LightSample_RGB lightSample;
    lightSample.distance = 1e7f;

    const int wp = optixLaunchParams.envMapInfo.patchSize.x;
    const int hp = optixLaunchParams.envMapInfo.patchSize.y;
    const float* __restrict__ cdfRow = optixLaunchParams.envMapInfo.coarseMarginal;
    const float* __restrict__ cdfCol = optixLaunchParams.envMapInfo.coarseConditional;
    const float* __restrict__ wPatch = optixLaunchParams.envMapInfo.patchWeight;
    const float totalWeight = optixLaunchParams.envMapInfo.totalWeight;

    if(totalWeight <= 0.0f){
        // 普通の一様サンプリング
        lightSample.pdf = 1.0f /  (4.0f * (float)M_PI) / float(optixLaunchParams.light.numLights);
        const float3 direction = random_unit_sphere(prd->random(), prd->random()); 
        lightSample.direction = direction; 
        float u, v;
        orthogonalToUVCoord(direction, &u, &v);
        lightSample.emission = make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
        return lightSample;
    }

    // 乱数生成 (バッチの選択 x2, バッチ内 x2)
    const float xi1 = prd->random();
    const float xi2 = prd->random();
    const float xi3 = prd->random();
    const float xi4 = prd->random();

    // どの行にするかを引く (y方向のサンプリング)
    const int row = lowerBoundCDF(cdfRow, hp, xi1);
    const int col = lowerBoundCDF(&cdfCol[row * wp], wp, xi2);

    const int patchIndex = row * wp + col;

    // パッチの角度の範囲
    float th0, th1, ph0, ph1;
    thetaPhiFromPatch(row, col, wp, hp, &th0, &th1, &ph0, &ph1);

    const float deltaPhi = ph1 - ph0;
    const float deltaCosT = cosf(th0) - cosf(th1);
    const float dOmega = deltaPhi *deltaCosT;
    
    const float phi = ph0 + xi3 * deltaPhi;
    const float cosTheta = cosf(th0) + xi4 * deltaCosT;
    const float theta = acosf(fminf(fmaxf(cosTheta, -1.0f), 1.0f));

    const float3 wi = sphericalToOrthogonalCoord(theta, phi);
    lightSample.direction = wi;

    // サンプリング
    float2 uv = envUVFromSpherical(theta, phi);
    lightSample.emission = make_float3(tex2D<float4>(optixLaunchParams.envMap, uv.x, uv.y));

    // PDF の計算
    const float pPatch = wPatch[patchIndex] / totalWeight;
    const float pIn = 1.0f / dOmega;
    float pdfDir = pPatch * pIn;

    lightSample.pdf = pdfDir / float(optixLaunchParams.light.numLights);

    return lightSample;
}


extern "C" __device__ LightSample_RGB __direct_callable__light_triangle_rgb(LightDefinition light, PRDRGB* prd)
{
    LightSample_RGB lightSample; // emission = (0, 0, 0) , pdf = 0 に初期化される 

    TriangleLightData& triangleLightData = optixLaunchParams.light.triangleLightData[light.lightIndexInType];
    
    // 三角形上の一様サンプリング
    mymath::matrix3x4 matrixO2WPoint = optixLaunchParams.frame.objectMatrixBuffer[prd->instanceID];
    const float3 V0 = triangleLightData.v0;
    const float3 V1 = triangleLightData.v1;
    const float3 V2 = triangleLightData.v2;

    const float3 v0 = mymath::mul3x4(matrixO2WPoint, make_float4(V0, 1.0f));
    const float3 v1 = mymath::mul3x4(matrixO2WPoint, make_float4(V1, 1.0f));
    const float3 v2 = mymath::mul3x4(matrixO2WPoint, make_float4(V2, 1.0f));
    

    const float3 a = v1 - v0;
    const float3 b = v2 - v0;

    const float     areaInWorld = 0.5f * length(cross(a, b));
    const float3    normalInWorld = normalize(cross(a, b));

    const float ta = clamp(1.0f - sqrtf(prd->random()), 0.0f, 1.0f);
    const float tb = clamp((1.0f - ta) * prd->random(), 0.0f, 1.0f);
    const float3 sampledPosition = v0 + ta * a + tb * b;

    float3 emission = triangleLightData.constantEmission;

    float distance = length(sampledPosition - prd->position);
    float3 lightDirection = normalize(sampledPosition - prd->position);

    const float cosTheta = dot(-lightDirection, normalInWorld);

    if(cosTheta > 1e-5f){
        // テクスチャの参照位置
        if(triangleLightData.emissiveTexture.texture > 0){
            const float2 u0 = triangleLightData.uv0;
            const float2 u1 = triangleLightData.uv1;
            const float2 u2 = triangleLightData.uv2;

            const float2 sampledUVCoodinate = u0 + ta * (u1 - u0) + tb * (u2 - u0);
            emission *= make_float3(tex2D<float4>(triangleLightData.emissiveTexture.texture, sampledUVCoodinate.x, 1.0 - sampledUVCoodinate.y));
        }

        float pdfInTriangle = 1.0f / areaInWorld; // 面積
        float geometricTerm = cosTheta / fmaxf(distance * distance, 1e-7f);
        lightSample.pdf         = pdfInTriangle /  float(optixLaunchParams.light.numLights) / geometricTerm;
        lightSample.distance    = distance;
        lightSample.position    = sampledPosition;
        lightSample.direction   = lightDirection;
        lightSample.emission    = emission * optixLaunchParams.light.lightIntensityFactor; // MEMO: pdfChoseLight とするのが分かりやすいい？
    }

    return lightSample;
}

// Spectral rendering 
extern "C" __device__ LightSample_Spectral __direct_callable__light_env_sphere_constant_spectral(LightDefinition light, PRDSpectral* prd)
{
    LightSample_Spectral lightSample;
    lightSample.pdf = 1.0f /  (4.0f * (float)M_PI) / float(optixLaunchParams.light.numLights);
    lightSample.distance = 1e7f;
    
    const float3 direction = random_unit_sphere(prd->random(), prd->random()); 
    lightSample.direction = direction; 
    float u, v;
    orthogonalToUVCoord(direction, &u, &v);
    lightSample.emissionRGB = make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    return lightSample;
}


extern "C" __device__ LightSample_Spectral __direct_callable__light_env_sphere_is_spectral(LightDefinition light, PRDSpectral* prd)
{
    LightSample_Spectral lightSample;
    lightSample.distance = 1e7f;

    const int wp = optixLaunchParams.envMapInfo.patchSize.x;
    const int hp = optixLaunchParams.envMapInfo.patchSize.y;
    const float* __restrict__ cdfRow = optixLaunchParams.envMapInfo.coarseMarginal;
    const float* __restrict__ cdfCol = optixLaunchParams.envMapInfo.coarseConditional;
    const float* __restrict__ wPatch = optixLaunchParams.envMapInfo.patchWeight;
    const float totalWeight = optixLaunchParams.envMapInfo.totalWeight;

    if(totalWeight <= 0.0f){
        // 普通の一様サンプリング
        lightSample.pdf = 1.0f /  (4.0f * (float)M_PI) / float(optixLaunchParams.light.numLights);
        const float3 direction = random_unit_sphere(prd->random(), prd->random()); 
        lightSample.direction = direction; 
        float u, v;
        orthogonalToUVCoord(direction, &u, &v);
        lightSample.emissionRGB = make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
        return lightSample;
    }

    // 乱数生成 (バッチの選択 x2, バッチ内 x2)
    const float xi1 = prd->random();
    const float xi2 = prd->random();
    const float xi3 = prd->random();
    const float xi4 = prd->random();

    // どの行にするかを引く (y方向のサンプリング)
    const int row = lowerBoundCDF(cdfRow, hp, xi1);
    const int col = lowerBoundCDF(&cdfCol[row * wp], wp, xi2);

    const int patchIndex = row * wp + col;

    // パッチの角度の範囲
    float th0, th1, ph0, ph1;
    thetaPhiFromPatch(row, col, wp, hp, &th0, &th1, &ph0, &ph1);

    const float deltaPhi = ph1 - ph0;
    const float deltaCosT = cosf(th0) - cosf(th1);
    const float dOmega = deltaPhi *deltaCosT;
    
    const float phi = ph0 + xi3 * deltaPhi;
    const float cosTheta = cosf(th0) + xi4 * deltaCosT;
    const float theta = acosf(fminf(fmaxf(cosTheta, -1.0f), 1.0f));

    const float3 wi = sphericalToOrthogonalCoord(theta, phi);
    lightSample.direction = wi;

    // サンプリング
    float2 uv = envUVFromSpherical(theta, phi);
    lightSample.emissionRGB = make_float3(tex2D<float4>(optixLaunchParams.envMap, uv.x, uv.y));
    
    // PDF の計算
    const float pPatch = wPatch[patchIndex] / totalWeight;
    const float pIn = 1.0f / dOmega;
    float pdfDir = pPatch * pIn;

    lightSample.pdf = pdfDir / float(optixLaunchParams.light.numLights);

    return lightSample;
}


extern "C" __device__ LightSample_Spectral __direct_callable__light_triangle_spectral(LightDefinition light, PRDSpectral* prd)
{
    LightSample_Spectral lightSample; // emission = (0, 0, 0) , pdf = 0 に初期化される 

    TriangleLightData& triangleLightData = optixLaunchParams.light.triangleLightData[light.lightIndexInType];
    
    // 三角形上の一様サンプリング
    mymath::matrix3x4 matrixO2WPoint = optixLaunchParams.frame.objectMatrixBuffer[prd->instanceID];
    const float3 V0 = triangleLightData.v0;
    const float3 V1 = triangleLightData.v1;
    const float3 V2 = triangleLightData.v2;

    const float3 v0 = mymath::mul3x4(matrixO2WPoint, make_float4(V0, 1.0f));
    const float3 v1 = mymath::mul3x4(matrixO2WPoint, make_float4(V1, 1.0f));
    const float3 v2 = mymath::mul3x4(matrixO2WPoint, make_float4(V2, 1.0f));
    

    const float3 a = v1 - v0;
    const float3 b = v2 - v0;

    const float     areaInWorld = 0.5f * length(cross(a, b));
    const float3    normalInWorld = normalize(cross(a, b));

    const float ta = clamp(1.0f - sqrtf(prd->random()), 0.0f, 1.0f);
    const float tb = clamp((1.0f - ta) * prd->random(), 0.0f, 1.0f);
    const float3 sampledPosition = v0 + ta * a + tb * b;

    float3 emissionRGB = triangleLightData.constantEmission;

    float distance = length(sampledPosition - prd->position);
    float3 lightDirection = normalize(sampledPosition - prd->position);

    const float cosTheta = dot(-lightDirection, normalInWorld);

    if(cosTheta > 1e-5f){
        // テクスチャの参照位置
        // if(triangleLightData.emissiveTexture.texture > 0){
        //     const float2 u0 = triangleLightData.uv0;
        //     const float2 u1 = triangleLightData.uv1;
        //     const float2 u2 = triangleLightData.uv2;

        //     const float2 sampledUVCoodinate = u0 + ta * (u1 - u0) + tb * (u2 - u0);
        //     emissionRGB *= make_float3(tex2D<float4>(triangleLightData.emissiveTexture.texture, sampledUVCoodinate.x, 1.0 - sampledUVCoodinate.y));
        // }
        // float emission = upSamplingFromRGB(emissionRGB, *prd);
        float pdfInTriangle = 1.0f / areaInWorld; // 面積
        float geometricTerm = cosTheta / fmaxf(distance * distance, 1e-7f);
        lightSample.pdf         = pdfInTriangle /  float(optixLaunchParams.light.numLights) / geometricTerm;
        lightSample.distance    = distance;
        lightSample.position    = sampledPosition;
        lightSample.direction   = lightDirection;
        lightSample.emissionRGB = emissionRGB; // MEMO: pdfChoseLight とするのが分かりやすい？
    }

    return lightSample;
}

extern "C" __device__
LightSample_Spectral __direct_callable__light_sky_spectral(LightDefinition light, PRDSpectral* prd)
{
    LightSample_Spectral ls{};

    const float lambda01 = prd->waveLengthNormalized;
    const float3 sunDir  = atmo::sunDirFromAngles(
        optixLaunchParams.sunParams.sunZenithRad,
        optixLaunchParams.sunParams.sunAzimuthRad);

    atmo::SkySamplingConfig config;
    const atmo::SkyLightSample s = atmo::sampleSkyEmitterMixture(
        config,
        sunDir,
        prd->random(),
        prd->random(),
        prd->random());

    ls.direction = s.dir;
    ls.distance  = 1.0e7f;

    // 既存流儀に合わせて full pdf にする
    const float pSelect = 1.0f / float(optixLaunchParams.light.numLights);
    ls.pdf = s.pdf * pSelect;

    ls.emissionRGB.y = atmo::evalSkyMissSpectralFixedObserver(
        optixLaunchParams.atmo,
        config,
        optixLaunchParams.spectral.D65,
        lambda01,
        s.dir,
        sunDir);

    return ls;
}