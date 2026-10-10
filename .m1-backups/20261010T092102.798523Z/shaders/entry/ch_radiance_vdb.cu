#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include "../device/trace_volume.cuh"
#include "../device/phase_function.cuh"
#include "../device/neural_phase_function.cuh"
#include "../device/atmosphere_sampling_device.cuh"
#include <stdio.h>


static __forceinline__ __device__
float evalPhaseFunction(){
    return 1.f / (4.0f * M_PI);
}
extern "C" __global__ void __closesthit__vdb_radiance_rgb()
{
    PRDRGB & prd = *getPRD<PRDRGB>();
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t vdbIndex         = sbtData.vdb.vdbIndex;
    // const uint32_t materialIndex    = sbtData.vdb.materialIndex;

    const float tEnter  = optixGetRayTmax();
    const float tExit   = __uint_as_float(optixGetAttribute_0());
    if(!(tExit > tEnter)){
        prd.continueTrace = false;
        return;
    }
    // ray の情報
    const float3 rayDirectionWorld  = normalize(optixGetWorldRayDirection());
    const float3 rayOriginWorld     = optixGetWorldRayOrigin();

    const float3 rayDirectionObject = optixTransformVectorFromWorldToObjectSpace(rayDirectionWorld);
    const float3 rayOriginObject    = optixTransformPointFromWorldToObjectSpace(rayOriginWorld);

    // prd.instanceID = optixGetInstanceId();

    // AABB
    const int primitiveIndex = optixGetPrimitiveIndex();
    const OptixAabb aabb = optixLaunchParams.vdbAABBs[primitiveIndex];
    const float3 bMin = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    const float3 bMax = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

    // --------------------------
    // 媒質のパラメータ
    // --------------------------
    const float sigmaTScale = optixLaunchParams.vdbs[primitiveIndex].densityScale;

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        optixLaunchParams.vdbs[vdbIndex].nanoGrid
    );
    const VDBGeomData& vdbGeom = optixLaunchParams.vdbs[vdbIndex];
    const float densityScale = optixLaunchParams.vdbs[vdbIndex].densityScale;

    auto acc = grid->getAccessor();
    auto sampler = makeSampler(acc);

    // デルタトラッキング
    float tScatter;
    const bool isScatter = deltaTrack_localMajorant(
        prd, vdbGeom, densityScale, grid, acc, sampler,
        rayOriginObject, rayDirectionObject,
        tEnter, tExit,
        sigmaTScale,
        tScatter
    );

    if(!isScatter)
    {
        // 散乱しなかった場合，radiance ray をもう一度飛ばしてメッシュと交差させる
        const float3 newOrigin = rayOriginWorld + tExit * rayDirectionWorld;

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        optixTrace(
            optixLaunchParams.traversable,
            newOrigin,
            rayDirectionWorld,
            1e-4f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( MASK_ALL ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RADIANCE_RAY_TYPE,
            RAY_TYPE_COUNT,
            RADIANCE_RAY_TYPE,
            u0, u1
        );

        return;
    }

    // 散乱する位置
    const float3 scatteredPointWorld = rayOriginWorld + tScatter * rayDirectionWorld;
    const float3 scatteredPointObject = rayOriginObject + tScatter * rayDirectionObject;

    prd.position = scatteredPointWorld;

    // 本当は，ここで散乱あるべとを反映させる
    // prd.albedo *= mediumAlbedo;

    // --------------------------
    // NEE
    // --------------------------
    const int numLights = optixLaunchParams.light.numLights;
    if(numLights > 0){
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample_RGB lightSample = optixDirectCall<LightSample_RGB, LightDefinition, PRDRGB*>(callLightType, light, &prd);

        if(lightSample.pdf > 0.f)
        {
            const float3 wiWorld = normalize(lightSample.direction);
            const float distance = lightSample.distance;

            // ratio tracking
            const float3 wiObject   = optixTransformVectorFromWorldToObjectSpace(wiWorld);
            const float transmittance = ratioTrack_localMajorant(
                prd, vdbGeom, densityScale, grid, acc, sampler,
                rayOriginObject, rayDirectionObject,
                tEnter, tExit,
                sigmaTScale
            );

            const float3 xExitW = rayOriginWorld + tExit * rayDirectionWorld;
            const float3 newOrigin = xExitW + rayDirectionWorld * 1e-3f;

            ShadowPRD shadowPrd;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);
            
            // 光源へ接続して可視性を判断
            optixTrace( 
                optixLaunchParams.traversable,
                newOrigin, // 出射位置
                wiWorld,
                1e-3f,
                distance - 1e-3f,
                0.0f,
                OptixVisibilityMask( MASK_SURFACE ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,    // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW_RAY_TYPE,    // miss SBT Index
                u0, u1
            );

            if(shadowPrd.visible && transmittance > 0.0f){
                const float phasePdf    = evalPhaseFunction();
                const float phaseValue  = phasePdf;

                float weight = balanceHeuristicWeight(1, fmaxf(lightSample.pdf, 1e-7f), 1, fmaxf(phasePdf, 1e-7f));
                prd.contribution += prd.albedo * (transmittance * lightSample.emission) * phaseValue * weight / fmaxf(lightSample.pdf, 1e-7f);
            }
        }
        prd.continueTrace = true;
    }

    // --------------------------
    // 次の方向の決定（位相関数のサンプリング）
    // --------------------------
    const float3 newDirection = random_unit_sphere(prd.random(), prd.random());

    prd.wi = newDirection;
    prd.position += 1e-3f * newDirection;
    prd.pdf.bxdf = evalPhaseFunction();
    prd.continueTrace = true;

    // if(prd.bounce == 0){
    //     prd.primaryNormal = make_float3(0.0f); // 媒質なので 0 埋め等
    //     prd.primaryAlbedo = make_float3(0.8f);
    // }
    
}


extern "C" __global__ void __closesthit__vdb_radiance_spectral()
{
    

    PRDSpectral & prd = *getPRD<PRDSpectral>();
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    // tmax (AABB との交差点, tEnter）から，媒質にあたるまで，あるいは AABB を抜ける (tExit) まで，レイを延長していく
    const float tEnter  = optixGetRayTmax();
    const float tExit   = __uint_as_float(optixGetAttribute_0());
    if(!(tExit > tEnter)){
        prd.continueTrace = 0;
        return;
    }
    
    // ray の情報
    const float3 rayDirectionWorld  = normalize(optixGetWorldRayDirection());
    const float3 rayOriginWorld     = optixGetWorldRayOrigin();

    const float3 rayDirectionObject = optixTransformVectorFromWorldToObjectSpace(rayDirectionWorld);
    const float3 rayOriginObject    = optixTransformPointFromWorldToObjectSpace(rayOriginWorld);

    prd.instanceID = optixGetInstanceId();

    // AABB
    const int primitiveIndex = optixGetPrimitiveIndex();
    const OptixAabb aabb = optixLaunchParams.vdbAABBs[primitiveIndex];
    const float3 bMin = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    const float3 bMax = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

    const uint32_t vdbIndex         = primitiveIndex;
    

    // --------------------------
    // 媒質のパラメータ
    // --------------------------
    const float sigmaTScale = 1.0f;

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        optixLaunchParams.vdbs[vdbIndex].nanoGrid
    );
    const VDBGeomData& vdbGeom = optixLaunchParams.vdbs[vdbIndex];
    const float densityScale = optixLaunchParams.vdbs[vdbIndex].densityScale;

    auto acc = grid->getAccessor();
    auto sampler = makeSampler(acc);

    const float uDiameter = (7.5f - 5.0f) / (20.0f - 5.0f);
    const float uWavelength = prd.waveLengthNormalized;


#if !defined(PHASE_FUNCTION_TABULATED) && !defined(PHASE_FUNCTION_HG)
    const float gHero = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, uWavelength, uDiameter);
    const float uDiameterNormalized = uDiameter * 2.0f - 1.0f;
    const float uHeroNormalized     = uWavelength * 2.0f - 1.0f;
    NSFPhiCache nfCache;
    // NSFFixedXRqsCache nfCache;
    bool nfCacheBuilt = false;
#endif
// #if defined(PHASE_FUNCTION_TABULATED)
    MiePhaseTex miePhaseTex = {};
    miePhaseTex.pdfTex = optixLaunchParams.mieTexture.pdf;
    miePhaseTex.cdfTex = optixLaunchParams.mieTexture.cdf;
    miePhaseTex.Ntheta = optixLaunchParams.mieTexture.numTheta;
    miePhaseTex.Nlambda = optixLaunchParams.mieTexture.numLambda;
    miePhaseTex.Nd = optixLaunchParams.mieTexture.numDiameter;
// #endif


    // デルタトラッキング
    float tScatter;
    const bool isScatter = deltaTrack_localMajorant(
        prd, vdbGeom, densityScale, grid, acc, sampler,
        rayOriginObject, rayDirectionObject,
        tEnter, tExit,
        sigmaTScale, tScatter
    );

    if(!isScatter)
    {
        // // 散乱しなかった場合，radiance ray をもう一度飛ばしてメッシュと交差させる
        // const float3 newOrigin = rayOriginWorld + tExit * rayDirectionWorld + 1e-4f * rayDirectionWorld;

        // uint32_t u0, u1;
        // packPointer(&prd, u0, u1);

        // optixTrace(
        //     optixLaunchParams.traversable,
        //     newOrigin,
        //     rayDirectionWorld,
        //     1e-4f,    // tmin
        //     1e20f,  // tmax
        //     0.0f,   // rayTime
        //     OptixVisibilityMask( 255 ),
        //     OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        //     RADIANCE_RAY_TYPE,
        //     RAY_TYPE_COUNT,
        //     RADIANCE_RAY_TYPE,
        //     u0, u1
        // );
        const float3 scatteredPointObject = rayOriginObject + (tExit + 1e-4f) * rayDirectionObject;
        const float3 scatteredPointWorld  = optixTransformPointFromObjectToWorldSpace(scatteredPointObject);
        prd.position = scatteredPointWorld;

        return;
    }

    #if !defined(PHASE_FUNCTION_TABULATED) && !defined(PHASE_FUNCTION_HG)
        // gHero = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, uHero, uDiameter);
        // uDiameterNormalized = uDiameter * 2.0f - 1.0f;
        // uHeroNormalized     = uHero * 2.0f - 1.0f;
        profile_build_phi_cache(
            optixLaunchParams,
            uDiameterNormalized,
            uHeroNormalized,
            gHero,
            nfCache
        );
        nfCacheBuilt = true;
        // buildNSFFixedXRqsCache(optixLaunchParams,
        //                    /* c0 */ uDiameterNormalized,
        //                    /* c1 */ uHeroNormalized,
        //                    gHero,
        //                    nfCache);
        // nfCacheBuilt = true;
    #endif

    // 散乱する位置
    const float3 scatteredPointObject = rayOriginObject + tScatter * rayDirectionObject;
    const float3 scatteredPointWorld  = optixTransformPointFromObjectToWorldSpace(scatteredPointObject);
    prd.position = scatteredPointWorld;

    // 本当は，ここで散乱あるべとを反映させる
    // prd.albedo *= mediumAlbedo;

    // --------------------------
    // NEE
    // --------------------------
    const int numLights = optixLaunchParams.light.numLights;
    if(numLights > 0){
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample_Spectral lightSample = optixDirectCall<LightSample_Spectral, LightDefinition, PRDSpectral*>(callLightType, light, &prd);
        const float pdfLight = fmaxf(lightSample.pdf, 1e-7f);

        
        const float3 wiWorld = normalize(lightSample.direction);
        const float distance = lightSample.distance;

        // ratio tracking
        const float3 wiObject   = optixTransformVectorFromWorldToObjectSpace(wiWorld);
        float tA0, tA1;
        // float tMediaExit = 0.0f;
        float transmittance = 1.0f;
        if(intersectAABB(scatteredPointObject, wiObject, bMin, bMax, 0.0f, distance, tA0, tA1)){
            const float t0 = fmaxf(0.0f, tA0);
            const float t1 = fminf(distance, tA1);
            if(t1 > t0){
                transmittance = ratioTrack_localMajorant(
                    prd, vdbGeom, densityScale, grid, acc, sampler,
                    scatteredPointObject, wiObject,
                    t0, t1, sigmaTScale
                );
            }
            // tMediaExit = fmaxf(0.0f, t1);
        }
        // shadow ray は体積の外から計算
        // const float3 xExitW = scatteredPointWorld + tMediaExit * wiWorld;
        // const float3 newOrigin = xExitW + wiWorld * 1e-3f;
        // const float  shadowTMax = distance - tMediaExit - 1e-3f;

        // ShadowPRD shadowPrd;
        // shadowPrd.visible = true;
        // uint32_t u0, u1;
        // packPointer(&shadowPrd, u0, u1);
        
        // // 光源へ接続して可視性を判断
        // if(shadowTMax > 0.0f){
        //     optixTrace( 
        //         optixLaunchParams.traversable,
        //         newOrigin, // 出射位置
        //         wiWorld,
        //         0.0f,
        //         shadowTMax,
        //         0.0f,
        //         OptixVisibilityMask( MASK_SURFACE ),
        //         OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        //             | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        //         SHADOW_RAY_TYPE,    // SBT offset
        //         RAY_TYPE_COUNT,     // SBT stride
        //         SHADOW_RAY_TYPE,    // miss SBT Index
        //         u0, u1
        //     );
        // }
        
        // if(shadowPrd.visible && transmittance > 0.0f){
        if(transmittance > 0.0f){
            // HWSS
            // const float spectralMISWeight = hwssSpectralWeight(prd.logPOrefix);
            const float cosTheta = dot(rayDirectionWorld, wiWorld); // 散乱角

            // constexpr int C = 4; 
            // const float invC = 1.0f / float(C);
            const float uNormalized = prd.waveLengthNormalized;
            // float phasePdfHeroForMIS = 0.0f;
            float emission = lightSample.emissionRGB.y * optixLaunchParams.light.lightIntensityFactor;
            float pdfPhase = 0.0f;
            
#if defined(PHASE_FUNCTION_TABULATED)
            pdfPhase = fmaxf(evalPhaseFunctionTabulated(cosTheta, uDiameter, uNormalized, miePhaseTex), 1e-20f);
#elif defined(PHASE_FUNCTION_HG)
            const float g = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, uNormalized, uDiameter);
            pdfPhase = fmaxf(evalPhaseFunctionHG(cosTheta, g), 1e-20f);
#else
            // const float g = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, uNormalized, uDiameter);
            // const float uRenormalized = 2.0f * uNormalized - 1.0f;
            // const float uDiameterNormalized = uDiameter * 2.0f - 1.0f;
            // pdfPhase = fmaxf(evalPhaseFunctionNF(optixLaunchParams, cosTheta, uDiameterNormalized, uRenormalized, g), 1e-20f);
            // if(!nfCacheBuilt){
            //     // buildNSFPhiCache(optixLaunchParams, uDiameterNormalized, uHeroNormalized, gHero, nfCache);
            //     profile_build_phi_cache(optixLaunchParams, uDiameterNormalized, uHeroNormalized, gHero, nfCache);
            //     nfCacheBuilt = true;
            // }
            // pdfPhase = fmaxf(evalPhaseFunctionNF_phiCached(nfCache, cosTheta), 1e-20f);
            pdfPhase = fmaxf(evalPhaseFunctionNF_rqsCached(nfCache, cosTheta), 1e-20f);
#endif

            // const float phaseValue  = phasePdf;

            const float misWeight = balanceHeuristicWeight(1, pdfLight, 1, pdfPhase);
            
            // float emission = 0.0f;
            // if(light.lightType == LIGHT_TYPE_SKY){
            //     const float3 sunDir = atmo::sunDirFromAngles(
            //         optixLaunchParams.sunParams.sunZenithRad,
            //         optixLaunchParams.sunParams.sunAzimuthRad
            //     );
            //     const atmo::SkySamplingConfig config;
            //     emission = atmo::evalSkyMissSpectralFixedObserver(
            //         optixLaunchParams.atmo,
            //         config,
            //         optixLaunchParams.spectral.D65,
            //         uNormalized,
            //         wiWorld,
            //         sunDir
            //     );

            //     // const float D65 = tex2D<float>(optixLaunchParams.spectral.D65, u, 0.5f);
            //     // float emission = baseEmission * D65;

            //     // throuput
            //     // float beta_k = (&prd.beta.x)[k];
            //     float add = emission * transmittance * pdfPhase * misWeight / pdfLight;
            //     (&prd.contribution.x)[k] += add;
            // }
            prd.contribution += prd.albedo * emission * transmittance * pdfPhase * misWeight / pdfLight / prd.pdf.spectral;

        }
        prd.continueTrace = 1;
    }

    // --------------------------
    // 次の方向の決定（位相関数のサンプリング）
    // --------------------------
    const float uHero = prd.waveLengthNormalized;
#if defined(PHASE_FUNCTION_TABULATED)
    PhaseSample ps = samplePhaseFunctionTabulated(rayDirectionWorld, prd.random(), prd.random(), uDiameter, uHero, miePhaseTex);
#elif defined(PHASE_FUNCTION_HG)
    const float gHero = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, uHero, uDiameter);
    PhaseSample ps = samplePhaseFunctionHG(rayDirectionWorld, prd.random(), prd.random(), gHero);
#else
    // const float gHero = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, uHero, uDiameter);
    // const float uDiameterNormalized = uDiameter * 2.0f - 1.0f;
    // const float uHeroNormalized = uHero * 2.0f - 1.0f;
    // PhaseSample ps = samplePhaseFunctionNF(optixLaunchParams, rayDirectionWorld, prd.random(), prd.random(), uDiameterNormalized, uHeroNormalized, gHero);
    // if(!nfCacheBuilt){
    //     buildNSFPhiCache(optixLaunchParams, uDiameterNormalized, uHeroNormalized, gHero, nfCache);
    //     nfCacheBuilt = true;
    // }
    PhaseSample ps = samplePhaseFunctionNF_phiCached(nfCache, rayDirectionWorld, prd.random(), prd.random());
    // PhaseSample ps = samplePhaseFunctionNF_rqsCached(nfCache, rayDirectionWorld, prd.random(), prd.random());
#endif

    prd.wi = ps.wi;
    prd.position += 1e-3f * ps.wi;

    // hero の PDF
    const float phasePDFHero = fmaxf(ps.pdf, 1e-20f);
    prd.pdf.bxdf = phasePDFHero;
    // prd.logPOrefix.x += logf(phasePDFHero);

    const float cosTheta = dot(rayDirectionWorld, ps.wi); // 散乱角
    
    // HWSS
    // constexpr int C = 4;
    // const float invC = 1.0f / (float)C;
//     const float u_k = wrap01(uHero + float(k) * invC);

// #if defined(PHASE_FUNCTION_TABULATED)
//     const float phasePdf_k = fmaxf(evalPhaseFunctionTabulated(cosTheta, uDiameter, u_k, miePhaseTex), 1e-20f);
// #elif defined(PHASE_FUNCTION_HG)
//     const float g_k = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, u_k, uDiameter);
//     const float phasePdf_k = fmaxf(evalPhaseFunctionHG(cosTheta, g_k), 1e-7f);
// #else
//     const float g_k = tex2D<float>(optixLaunchParams.mieTexture.phaseParameterG, u_k, uDiameter);
//     const float uNormalized = 2.0f * u_k - 1.0f;
//     const float uDiameterNormalized = uDiameter * 2.0f - 1.0f;
//     const float phasePdf_k = fmaxf(evalPhaseFunctionNF(optixLaunchParams, cosTheta, uDiameterNormalized, uNormalized, g_k), 1e-7f);
// #endif
    
    prd.continueTrace = 1;
    
    // if(prd.bounce == 0){
    //     prd.primaryNormal = make_float3(0.0f); // 媒質なので 0 埋め等
    //     prd.primaryAlbedo = make_float3(0.8f);
    // }
}