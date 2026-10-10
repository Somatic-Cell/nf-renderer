#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __raygen__renderFrame_rgb()
{
    // // ピクセル位置の取得
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int accumID = optixLaunchParams.frame.accumID;
    const int frameID = optixLaunchParams.frame.frameID;
    const auto &camera = optixLaunchParams.camera;

    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;


    // PRD: per-ray-data
    PRDRGB prd;
    // 乱数をピクセル位置とフレーム数をシード値として初期化
    prd.random.init(fbIndex, optixLaunchParams.frame.frameID);
    
    uint32_t u0, u1;
    packPointer( &prd, u0, u1);

    float3 pixelColor = make_float3(0.f); 
    // float3 pixelNormal = make_float3(0.f);
    // float3 pixelAlbedo = make_float3(0.f);

    for(int sampleID = 0; sampleID < optixLaunchParams.frame.numPixelSamples; sampleID ++){

        // prd.emission = make_float3(0.f);
        prd.contribution = make_float3(0.f);
        prd.albedo   = make_float3(1.f);
        prd.continueTrace = true;
        prd.bounce = 0;
        prd.pdf.bxdf = 1.0f;
        // prd.pdf.light = 1.0f;

        // スクリーン空間上のサンプル点をサブピクセル精度でサンプリング
        const float2 screen = make_float2(
            ((float)(optixLaunchParams.frame.size.x - ix - 1) + prd.random()) / (float)optixLaunchParams.frame.size.x, 
            ((float)(optixLaunchParams.frame.size.y - iy - 1) + prd.random()) / (float)optixLaunchParams.frame.size.y 
        );


        // Attention! : レンズ系の direct callable 関数は，プログラム全体でオフセット：0
        const LensRay ray = 
            optixDirectCall<LensRay, const float2, const float, const float>(optixLaunchParams.camera.cameraMode, screen, prd.random(), prd.random());

        prd.position = ray.org;
        prd.wi = ray.dir;

        

        while(prd.bounce < optixLaunchParams.frame.maxBounce)
        {
            optixTrace( optixLaunchParams.traversable,
                prd.position,
                prd.wi,
                1e-4f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( MASK_ALL ),
                OPTIX_RAY_FLAG_NONE,
                RADIANCE_RAY_TYPE,
                RAY_TYPE_COUNT,
                RADIANCE_RAY_TYPE,
                u0, u1);

            if(!prd.continueTrace){
                break;
            }
            prd.bounce ++;
        }

        pixelColor = pixelColor + (prd.contribution * ray.weight - pixelColor) / (sampleID + 1.0f);
        // pixelNormal = pixelNormal + (prd.primaryNormal - pixelNormal) / (sampleID + 1.0f);
        // pixelAlbedo = pixelAlbedo + (prd.primaryAlbedo - pixelAlbedo) / (sampleID + 1.0f);

    }
    pixelColor = clamp(pixelColor, 0.0f, 1000.0f);
    if(!isfinite(pixelColor.x) || !isfinite(pixelColor.y) || !isfinite(pixelColor.z)){
        pixelColor = make_float3(0.0f);
    }

    float4 rgba = make_float4(pixelColor, 1.0f);

    
    if (optixLaunchParams.frame.frameID > 0){
        float4 rgba_now = optixLaunchParams.frame.colorBuffer[fbIndex];
        rgba = rgba_now + (rgba - rgba_now) / (optixLaunchParams.frame.frameID + 1.0f); 
    }
    
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    // optixLaunchParams.frame.albedoBuffer[fbIndex] = make_float4(pixelAlbedo, 1.0f);
    // optixLaunchParams.frame.normalBuffer[fbIndex] = make_float4(pixelNormal, 1.0f);
}


//Spectral rendering
extern "C" __global__ void __raygen__renderFrame_spectral()
{
    // // ピクセル位置の取得
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // const int accumID = optixLaunchParams.frame.accumID;
    // const int frameID = optixLaunchParams.frame.frameID;
    // const auto &camera = optixLaunchParams.camera;

    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;


    // PRD: per-ray-data
    PRDSpectral prd;
    // 乱数をピクセル位置とフレーム数をシード値として初期化
    prd.random.init(fbIndex, optixLaunchParams.frame.frameID);
    
    uint32_t u0, u1;
    packPointer( &prd, u0, u1);

    float3 pixelColor = make_float3(0.f); 
    // float3 pixelNormal = make_float3(0.f);
    // float3 pixelAlbedo = make_float3(0.f);

    for(int sampleID = 0; sampleID < optixLaunchParams.frame.numPixelSamples; sampleID ++){

        prd.contribution = 0.0f;
        prd.albedo   = 1.f;
        prd.continueTrace = 1;
        prd.bounce = 0;
        prd.pdf.bxdf = 1.0f;

        // スペクトルの決定
        const SampledWavelength sampledWavelength = sampleWavelengthFromCdf(optixLaunchParams.spectral, prd.random());
        // const float wavelengthTexSamplePoint = prd.random();
        // prd.waveLengthNormalized = wavelengthTexSamplePoint; // wavelength (0-1)
        const float wavelengthMin = optixLaunchParams.spectral.wavelengthMin;
        const float wavelengthMax = optixLaunchParams.spectral.wavelengthMax;
        // prd.waveLength = wavelengthMin + (wavelengthMax - wavelengthMin) * wavelengthTexSamplePoint;
        prd.waveLength = sampledWavelength.lambda;
        const float wavelengthNormalized  = (sampledWavelength.lambda - wavelengthMin) 
            / (wavelengthMax - wavelengthMin); // wavelength (0-1)
        prd.waveLengthNormalized = wavelengthNormalized;
        prd.pdf.spectral = fmaxf(sampledWavelength.pdf, 1.0e-20f);

        // For HWSS
        // prd.beta = make_float4(1.0f / fmaxf(sampledWavelength.pdf, 1.0e-20f));

        // constexpr int C = 4;
        // const float invC = 1.0f / (float)C;
        // for(int k = 0; k < C; ++k)
        // {
        //     const float u_k = wrap01(prd.waveLengthNormalized + float(k) * invC);
        //     const float p_k = evalWavelengthPdf(optixLaunchParams.spectral, u_k);
        //     (&prd.logPOrefix.x)[k] = logf(fmaxf(p_k, 1.0e-20f));
        // }

        // スクリーン空間上のサンプル点をサブピクセル精度でサンプリング
        const float2 screen = make_float2(
            ((float)(optixLaunchParams.frame.size.x - ix - 1) + prd.random()) / (float)optixLaunchParams.frame.size.x, 
            ((float)(optixLaunchParams.frame.size.y - iy - 1) + prd.random()) / (float)optixLaunchParams.frame.size.y 
        );


        // Attention! : レンズ系の direct callable 関数は，プログラム全体でオフセット：0
        const LensRay ray = 
            optixDirectCall<LensRay, const float2, const float, const float>(optixLaunchParams.camera.cameraMode, screen, prd.random(), prd.random());

        prd.position = ray.org;
        prd.wi = ray.dir;
        

        while(prd.bounce < optixLaunchParams.frame.maxBounce)
        {
            optixTrace( optixLaunchParams.traversable,
                prd.position,
                prd.wi,
                1e-4f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( MASK_ALL ),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RADIANCE_RAY_TYPE,
                RAY_TYPE_COUNT,
                RADIANCE_RAY_TYPE,
                u0, u1);

            if(!(prd.continueTrace == 1)){
                break;
            }
            prd.bounce ++;
        }

        // Spectral -> XYZ

        // const float weightLambda = (wavelengthMax - wavelengthMin) / float(C);
        
        // float x = 0.0f, y = 0.0f, z = 0.0f;

        // for(int k = 0; k < C; ++k){
        //     const float u = wrap01(prd.waveLengthNormalized + float(k) / float(C));

        //     const float xFunc = tex2D<float>(optixLaunchParams.spectral.xyzFunc[0], u, 0.5f);
        //     const float yFunc = tex2D<float>(optixLaunchParams.spectral.xyzFunc[1], u, 0.5f);
        //     const float zFunc = tex2D<float>(optixLaunchParams.spectral.xyzFunc[2], u, 0.5f);

        //     const float Lk = (&prd.contribution.x)[k] * 0.001f;
        //     x += xFunc * Lk * weightLambda;
        //     y += yFunc * Lk * weightLambda;
        //     z += zFunc * Lk * weightLambda;
        // }

        const float x = prd.contribution * 0.1f * tex2D<float>(optixLaunchParams.spectral.xyzFunc[0], wavelengthNormalized, 0.5f);
        const float y = prd.contribution * 0.1f * tex2D<float>(optixLaunchParams.spectral.xyzFunc[1], wavelengthNormalized, 0.5f);
        const float z = prd.contribution * 0.1f * tex2D<float>(optixLaunchParams.spectral.xyzFunc[2], wavelengthNormalized, 0.5f);
        // XYZ -> sRGB (D65)
        // MEMO: この変換行列は簡易版であり，本当は厳密に計算して求める必要がある
        const float contribR =  3.240542f * x -1.5371835f * y -0.4985314f * z;
        const float contribG = -0.969260f * x +1.8760108f * y +0.0415560f * z;
        const float contribB =  0.055643f * x -0.2040259f * y +1.0572552f * z;

        const float3 contributionRGB = make_float3(contribR, contribG, contribB);

        pixelColor = pixelColor + (contributionRGB * ray.weight - pixelColor) / (sampleID + 1.0f);
        // pixelNormal = pixelNormal + (prd.primaryNormal - pixelNormal) / (sampleID + 1.0f);
        // pixelAlbedo = pixelAlbedo + (prd.primaryAlbedo - pixelAlbedo) / (sampleID + 1.0f);

    }
    pixelColor = clamp(pixelColor, -1000.0f, 1e7f);
    if(!isfinite(pixelColor.x) || !isfinite(pixelColor.y) || !isfinite(pixelColor.z)){
        pixelColor = make_float3(0.0f);
    }

    float4 rgba = make_float4(pixelColor, 1.0f);

    
    if (optixLaunchParams.frame.frameID > 0){
        float4 rgba_now = optixLaunchParams.frame.colorBuffer[fbIndex];
        rgba = rgba_now + (rgba - rgba_now) / (optixLaunchParams.frame.frameID + 1.0f); 
    }
    
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    // optixLaunchParams.frame.albedoBuffer[fbIndex] = make_float4(pixelAlbedo, 1.0f);
    // optixLaunchParams.frame.normalBuffer[fbIndex] = make_float4(pixelNormal, 1.0f);
}
