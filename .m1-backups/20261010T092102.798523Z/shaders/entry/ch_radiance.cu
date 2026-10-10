#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __closesthit__radiance_rgb()
{
    PRDRGB &prd = *getPRD<PRDRGB>();

    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t meshIndex = sbtData.tri.meshIndex;
    const uint32_t materialIndex = sbtData.tri.materialIndex;
    const TriangleMeshGeomData &mesh = optixLaunchParams.meshes[meshIndex];
    const MaterialData & material = optixLaunchParams.materials[materialIndex];
    
    // 基本的な交差点の情報を取得
    const int primID = optixGetPrimitiveIndex();
    const uint3 &vertexIndex = mesh.index[primID]; 

    const float2 uv = optixGetTriangleBarycentrics();
    const float u = uv.x;
    const float v = uv.y;

    // レイの進行方向を取得
    const float3 rayDirection = optixGetWorldRayDirection();
    // レイとプリミティブの交差点を取得
    const float rayLength = optixGetRayTmax();
    const float3 intersectedPoint = optixGetWorldRayOrigin() + rayLength * rayDirection;

    // 形状処理用. 面法線を取得
    // prd.instanceID = optixGetInstanceId();
    const float3 &V1 = optixTransformPointFromObjectToWorldSpace(mesh.vertex[vertexIndex.x]);
    const float3 &V2 = optixTransformPointFromObjectToWorldSpace(mesh.vertex[vertexIndex.y]);
    const float3 &V3 = optixTransformPointFromObjectToWorldSpace(mesh.vertex[vertexIndex.z]);
     
    float3 Ng = normalize(cross(V2-V1, V3-V1));
    const float triangleArea = 0.5f * fmaxf(length(cross(V2 - V1, V3 - V1)), 1e-7f);

    
    // Diffuse テクスチャ座標を取得
    const float2 &UVDiffuse1 = mesh.texcoord[vertexIndex.x];
    const float2 &UVDiffuse2 = mesh.texcoord[vertexIndex.y];
    const float2 &UVDiffuse3 = mesh.texcoord[vertexIndex.z];

    const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    // Normal テクスチャ座標を取得
    const float2 &UVNormal1 = mesh.texcoord[vertexIndex.x];
    const float2 &UVNormal2 = mesh.texcoord[vertexIndex.y];
    const float2 &UVNormal3 = mesh.texcoord[vertexIndex.z];
    
    const float2 normalTextureCoordinate = (1.0f - u - v) * UVNormal1
        + u * UVNormal2
        + v * UVNormal3;

    // Emissive テクスチャ座標を取得
    const float2 &UVEmissive1 = mesh.texcoord[vertexIndex.x];
    const float2 &UVEmissive2 = mesh.texcoord[vertexIndex.y];
    const float2 &UVEmissive3 = mesh.texcoord[vertexIndex.z];
    const float2 emissiveTextureCoordinate = (1.0f - u - v) * UVEmissive1
        + u * UVEmissive2
        + v * UVEmissive3;

    // シェーディング用．頂点法線があれば頂点法線を，なければ面法線を使用
    
    float3 Ns = Ng;
    if(mesh.hasNormal){
        const float3 &N1 = mesh.normal[vertexIndex.x];
        const float3 &N2 = mesh.normal[vertexIndex.y];
        const float3 &N3 = mesh.normal[vertexIndex.z];
        Ns =    (1.0f - u - v) * N1
                +            u * N2
                +            v * N3;
        Ns = optixTransformNormalFromObjectToWorldSpace(Ns);
    }
    Ns = normalize(Ns);
    
    float3 nml = make_float3(0.0f, 0.0f, 1.0f);
    if(material.normalTexture.texture > 0) {
        float4 fromTexture = tex2D<float4>(material.normalTexture.texture, normalTextureCoordinate.x, 1.0f - normalTextureCoordinate.y);
        nml = (make_float3(fromTexture) - make_float3(0.5f)) * 2.0f ;
        nml.y = -nml.y;
        nml.z = sqrtf(fmaxf(1.0f - nml.y * nml.y - nml.x * nml.x, 0.0001f));
    }

    // 接ベクトル空間の計算
    float4 tan = (fabsf(Ns.x) > 0.999f) ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    if(mesh.hasTangent){
        const float4 &T1 = mesh.tangent[vertexIndex.x];
        const float4 &T2 = mesh.tangent[vertexIndex.y];
        const float4 &T3 = mesh.tangent[vertexIndex.z];
        tan =  (1.0f - u - v) * T1
            +            u * T2
            +            v * T3;

        tan = make_float4(optixTransformNormalFromObjectToWorldSpace(make_float3(tan)), tan.w);
    }

    
    const float3 T = normalize(make_float3(tan));
    const float3 B = normalize(cross(Ns, T) * tan.w);
    
    Ns = normalize(T * nml.x + B * nml.y + Ns * nml.z);
    float3 biNormal = normalize(cross(T, Ns)); 
    float3 tangent = normalize(cross(Ns, biNormal));
    
    Ng = normalize(Ng);

    // Albedo の計算
    float3 albedo = material.color;
    
    if(material.diffuseTexture.texture > 0){
        float4 fromTexture = tex2D<float4>(
            material.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
        albedo = make_float3(fromTexture);
    }

    float4 arm = make_float4(0.0f, 0.1f, 0.0f, 0.0f);
    
    if(material.rmTexture.texture > 0){
        arm = tex2D<float4>(material.rmTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
    }

    IntersectedData_RGB matData;
    matData.ior = 1.45f; // glass
    matData.metallic = arm.z;
    matData.roughness = arm.y;
    matData.baseColor = albedo;

    if(dot(rayDirection, Ng) > 0.f && material.materialType != MATERIAL_TYPE_LIGHT) {
        Ng = -Ng;
        Ns = -Ns;
        tangent = -tangent;
        biNormal = normalize(cross(Ns, tangent));
        matData.ior = 1.0f / matData.ior; // glass
    }

    prd.position = intersectedPoint; // 後で置き場を考える
    // メッシュの光源と交差した場合の処理
    if(material.materialType == MATERIAL_TYPE_LIGHT)
    {
        const float cosTheta = fabsf(dot(-1.0f * rayDirection, Ng));
        if(cosTheta > 1e-7f)
        {
            float3 emission = material.emissive;
            if(material.emissiveTexture.texture > 0){
                float4 fromTexture = tex2D<float4>(
                    material.emissiveTexture.texture, emissiveTextureCoordinate.x, 1.0f - emissiveTextureCoordinate.y);
                emission = make_float3(fromTexture);
            }
            emission *= optixLaunchParams.light.lightIntensityFactor;

            if(prd.bounce != 0)
            {
                const float r = rayLength;
                const float pSelect = 1.0f / float(optixLaunchParams.light.numLights);
                float pArea = 1.0f / (triangleArea);
                // const float geometricTerm = cosTheta / (rayLength * rayLength);
                const float pdfLight =  pSelect * pArea * (r * r) / cosTheta;
                
                const float weight = balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
                prd.contribution += emission * prd.albedo * weight;
            } else {
                prd.contribution += emission * prd.albedo;
            }
        }
            

        // if(prd.bounce == 0){
        //     prd.primaryNormal = Ns;
        //     prd.primaryAlbedo = albedo;
        // }
        prd.continueTrace = false;
        return;
    }


    const float3 woLocal = worldToLocal(-1.0f * rayDirection, tangent, Ns, biNormal);
    // ガラスのメッシュと交差した場合の処理（レイトレーシングを続行）
    if(material.materialType == MATERIAL_TYPE_GLASS)
    {
        int callableFunctionOffset = NUM_LENS_TYPE + MATERIAL_TYPE_GLASS * 2; // 登録した順番が lens の後で，glass * 2 が 対応するマテリアルの sample 関数
        prd.position = intersectedPoint;
        float3 wiLocal = optixDirectCall<float3, const float3, const IntersectedData_RGB& , PRDRGB*>(callableFunctionOffset, woLocal, matData, &prd);
        prd.wi = localToWorld(wiLocal, tangent, Ns, biNormal);
        prd.continueTrace = true;
        prd.position += (wiLocal.y > 0.0f) ? Ng * 1e-3f : -1.0f * Ng * 1e-3f;

        // if(prd.bounce == 0){
        //     prd.primaryNormal = Ns;
        //     prd.primaryAlbedo = albedo;
        // }
        return;
    }


    const int numLights = optixLaunchParams.light.numLights; // 後で置き場を考える
    // MIS によるシェーディング
    // ライトサンプリング
    if (numLights > 0){

        //MIS
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample_RGB lightSample = optixDirectCall<LightSample_RGB, LightDefinition, PRDRGB*>(callLightType, light, &prd);
        const float3 wiLocal = worldToLocal(lightSample.direction, tangent, Ns, biNormal);

        if(lightSample.pdf > 0.f && wiLocal.y > 0.0f)
        {
            ShadowPRD shadowPrd;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);
            

            // BSDF の計算
            int callableFunctionOffset = NUM_LENS_TYPE + material.materialType * 2 + 1; // 登録した順番が lens の後で，materialType * 2 + 1 が 対応するマテリアルの eval 関数 
            float3 bxdfValue = optixDirectCall<float3, const float3, const float3, const IntersectedData_RGB& , PRDRGB*>(callableFunctionOffset, wiLocal, woLocal, matData, &prd);
            float bxdfPdf = prd.pdf.bxdf;
            const float3 wi = localToWorld(wiLocal, tangent, Ns, biNormal); 


            // // 光源へ接続して可視性を判断
            optixTrace( 
                optixLaunchParams.traversable,
                intersectedPoint, // 出射位置
                wi,
                1e-3f,
                lightSample.distance - 1e-3f,
                0.0f,
                OptixVisibilityMask( MASK_SURFACE ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,    // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW_RAY_TYPE,    // miss SBT Index
                u0, u1
            );

            if(shadowPrd.visible){
                float weight = balanceHeuristicWeight(1, lightSample.pdf, 1, bxdfPdf);
                prd.contribution += prd.albedo * lightSample.emission * bxdfValue * wiLocal.y * weight / fmaxf(lightSample.pdf, 1e-7f);
            }
        }
        prd.continueTrace = true;
    }

    
    // 次の方向の決定
    // BSDF のサンプリング とアルベドの変更
    int callableFunctionOffset = NUM_LENS_TYPE + material.materialType * 2;
    const float3 sampledWiLocal = optixDirectCall<float3, const float3, const IntersectedData_RGB&, PRDRGB*>(callableFunctionOffset, woLocal, matData, &prd);
    callableFunctionOffset = NUM_LENS_TYPE + material.materialType * 2 + 1;
    const float3 bxdfValue = optixDirectCall<float3, const float3, const float3, const IntersectedData_RGB& , PRDRGB*>(callableFunctionOffset, sampledWiLocal, woLocal, matData, &prd);
    const float3 sampledWi = localToWorld(sampledWiLocal, tangent, Ns, biNormal);

    // if(sampledWiLocal.y < 1e-7f){
    //     prd.continueTrace = false;
    //     return;
    // }
    
    prd.position += 1e-3f * Ng;
    prd.albedo *= bxdfValue * sampledWiLocal.y / fmaxf(prd.pdf.bxdf, 1e-7f);
    prd.wi = sampledWi; 
    prd.continueTrace = true;
    
    // if(prd.bounce == 0){
    //     prd.primaryNormal = Ns;
    //     prd.primaryAlbedo = albedo;
    // }
}


extern "C" __global__ void __closesthit__radiance_spectral()
{
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();
    PRDSpectral &prd = *getPRD<PRDSpectral>();

    const uint32_t meshIndex = sbtData.tri.meshIndex;
    const uint32_t materialIndex = sbtData.tri.materialIndex;
    const TriangleMeshGeomData* __restrict__ mesh = &optixLaunchParams.meshes[meshIndex];
    const MaterialData* __restrict__ material = &optixLaunchParams.materials[materialIndex];
    
    // 基本的な交差点の情報を取得
    const int primID = optixGetPrimitiveIndex();

    const uint3 vertexIndex = mesh->index[primID];

    
    const float2 uv = optixGetTriangleBarycentrics();
    const float u = uv.x;
    const float v = uv.y;

    // レイの進行方向を取得
    const float3 rayDirection = normalize(optixGetWorldRayDirection());
    // レイとプリミティブの交差点を取得
    const float rayLength = optixGetRayTmax();
    const float3 intersectedPoint = optixGetWorldRayOrigin() + rayLength * rayDirection;

    // 形状処理用. 面法線を取得
    prd.instanceID = optixGetInstanceId();
    const float3* __restrict__ vertex = mesh->vertex;
    const float3 V1 = optixTransformPointFromObjectToWorldSpace(vertex[vertexIndex.x]);
    const float3 V2 = optixTransformPointFromObjectToWorldSpace(vertex[vertexIndex.y]);
    const float3 V3 = optixTransformPointFromObjectToWorldSpace(vertex[vertexIndex.z]);
    
    float3 Ng = normalize(cross(V2-V1, V3-V1));
    const float triangleArea = 0.5f * fmaxf(length(cross(V2 - V1, V3 - V1)), 1e-7f);

    // Diffuse テクスチャ座標を取得
    const float2* __restrict__ texcoord = mesh->texcoord;
    const float2 UVDiffuse1 = texcoord[vertexIndex.x];
    const float2 UVDiffuse2 = texcoord[vertexIndex.y];
    const float2 UVDiffuse3 = texcoord[vertexIndex.z];

    const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    // Normal テクスチャ座標を取得
    // const float2 &UVNormal1 = mesh.texcoord[vertexIndex.x];
    // const float2 &UVNormal2 = mesh.texcoord[vertexIndex.y];
    // const float2 &UVNormal3 = mesh.texcoord[vertexIndex.z];
    
    const float2 normalTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    // Emissive テクスチャ座標を取得
    // const float2 &UVEmissive1 = mesh.texcoord[vertexIndex.x];
    // const float2 &UVEmissive2 = mesh.texcoord[vertexIndex.y];
    // const float2 &UVEmissive3 = mesh.texcoord[vertexIndex.z];
    const float2 emissiveTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    // シェーディング用．頂点法線があれば頂点法線を，なければ面法線を使用
    
    float3 Ns = Ng;
    if(mesh->hasNormal){
        const float3* __restrict__ normal = mesh->normal;
        const float3 N1 = normal[vertexIndex.x];
        const float3 N2 = normal[vertexIndex.y];
        const float3 N3 = normal[vertexIndex.z];
        Ns =    (1.0f - u - v) * N1
                +            u * N2
                +            v * N3;
        Ns = optixTransformNormalFromObjectToWorldSpace(Ns);
    }
    Ns = normalize(Ns);
    
    float3 nml = make_float3(0.0f, 0.0f, 1.0f);
    if(material->normalTexture.texture > 0) {
        float4 fromTexture = tex2D<float4>(material->normalTexture.texture, normalTextureCoordinate.x, 1.0f - normalTextureCoordinate.y);
        nml = (make_float3(fromTexture) - make_float3(0.5f)) * 2.0f ;
        nml.y = -nml.y;
        nml.z = sqrtf(fmaxf(1.0f - nml.y * nml.y - nml.x * nml.x, 0.0001f));
    }

    // 接ベクトル空間の計算
    float4 tan = (fabsf(Ns.x) > 0.999f) ? make_float4(0.0f, 1.0f, 0.0f, 1.0f) : make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    if(mesh->hasTangent){
        const float4* __restrict__ tangent = mesh->tangent;
        const float4 T1 = tangent[vertexIndex.x];
        const float4 T2 = tangent[vertexIndex.y];
        const float4 T3 = tangent[vertexIndex.z];
        tan =  (1.0f - u - v) * T1
            +            u * T2
            +            v * T3;

        tan = make_float4(optixTransformNormalFromObjectToWorldSpace(make_float3(tan)), tan.w);
    }

    
    const float3 T = normalize(make_float3(tan));
    const float3 B = normalize(cross(Ns, T) * tan.w);
    
    Ns = normalize(T * nml.x + B * nml.y + Ns * nml.z);
    float3 biNormal = normalize(cross(T, Ns)); 
    float3 tangent = normalize(cross(Ns, biNormal));

    // Albedo の計算
    float3 albedoRGB = material->color;
    
    if(material->diffuseTexture.texture > 0){
        float4 fromTexture = tex2D<float4>(
            material->diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
        albedoRGB = make_float3(fromTexture);
    }

    const float albedo = upSamplingFromRGB(albedoRGB, prd);


    float4 arm = make_float4(0.0f, 0.1f, 0.0f, 0.0f);
    
    if(material->rmTexture.texture > 0){
        arm = tex2D<float4>(material->rmTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
    }

    IntersectedData_Spectral matData;
    matData.ior = 1.444829f + 3506.29f / (prd.waveLength * prd.waveLength); // glass
    matData.metallic = arm.z;
    matData.roughness = arm.y;
    matData.baseColor = albedo;

    if(dot(rayDirection, Ng) > 0.f && material->materialType != MATERIAL_TYPE_LIGHT) {
        Ng = -Ng;
        Ns = -Ns;
        tangent = -tangent;
        biNormal = normalize(cross(Ns, tangent));
        matData.ior = 1.0f / matData.ior; // glass
    }

    prd.position = intersectedPoint; // 後で置き場を考える

    // メッシュの光源と交差した場合の処理
    if(material->materialType == MATERIAL_TYPE_LIGHT)
    {
        const float cosTheta = fabsf(dot(-1.0f * rayDirection, Ng));
        if(cosTheta > 1e-7f)
        {
            float3 emissionRGB = material->emissive;
            // if(material->emissiveTexture.texture > 0){
            //     float4 fromTexture = tex2D<float4>(
            //         material->emissiveTexture.texture, emissiveTextureCoordinate.x, 1.0f - emissiveTextureCoordinate.y);
            //     emissionRGB = make_float3(fromTexture);
            // }
            

            // 方向の MIS
            float misWeight = 1.0f;
            if(prd.bounce != 0)
            {
                const float r = rayLength;
                const float pSelect = 1.0f / float(optixLaunchParams.light.numLights);
                float pArea = 1.0f / (triangleArea);
                float pdfLight =  pSelect * pArea * (r * r) / cosTheta;
                pdfLight = fmaxf(pdfLight, 1e-7f);

                const float pdfBXDF = fmaxf(prd.pdf.bxdf, 1e-7f);
                misWeight = balanceHeuristicWeight(1, pdfBXDF, 1, pdfLight);
            }
            
            // 波長方向の MIS
            // const float spectralMISWeight = hwssSpectralWeight(prd.logPOrefix);
            // constexpr int C = 4;
            // const float invC = 1.0f/ float(C);
            // const float uHero = prd.waveLengthNormalized;
            const float uWavelength = prd.waveLengthNormalized;

            float emission = emissionRGB.y;
            const float D65 = tex2D<float>(optixLaunchParams.spectral.D65, u, 0.5f);
            prd.contribution = emission * misWeight;

            // for(int k = 0; k < C; ++k){
            //     const float u = wrap01(uHero + float(k) * invC);
            //     const float D65 = tex2D<float>(optixLaunchParams.spectral.D65, u, 0.5f);
            //     emission *= (optixLaunchParams.light.lightIntensityFactor  * D65);

            //     // throughput
            //     float beta_k = (&prd.beta.x)[k];
            //     (&prd.contribution.x)[k] += emission * beta_k * directionalMISWeight * spectralMISWeight;
            // }
        }
            
        prd.continueTrace = false;
        return;
    }


    const float3 woLocal = worldToLocal(-1.0f * rayDirection, tangent, Ns, biNormal);
    // // ガラスのメッシュと交差した場合の処理（レイトレーシングを続行）
    // if(material->materialType == MATERIAL_TYPE_GLASS)
    // {
    //     int callableFunctionOffset = NUM_LENS_TYPE + MATERIAL_TYPE_GLASS * 2; // 登録した順番が lens の後で，glass * 2 が 対応するマテリアルの sample 関数
    //     prd.position = intersectedPoint;
    //     float3 wiLocal = optixDirectCall<float3, const float3, const IntersectedData_Spectral& , PRDSpectral*>(callableFunctionOffset, woLocal, matData, &prd);
    //     prd.wi = localToWorld(wiLocal, tangent, Ns, biNormal);
    //     prd.continueTrace = true;
        
    //     prd.position += (wiLocal.y > 0.0f) ? Ng * 1e-3f : -1.0f * Ng * 1e-3f;

    //     if(prd.bounce == 0){
    //         prd.primaryNormal = Ns;
    //         prd.primaryAlbedo = albedoRGB;
    //     }
    //     return;
    // }


    const int numLights = optixLaunchParams.light.numLights; // 後で置き場を考える
    // MIS によるシェーディング
    // ライトサンプリング
    if (numLights > 0){

        //MIS
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample_Spectral lightSample = optixDirectCall<LightSample_Spectral, LightDefinition, PRDSpectral*>(callLightType, light, &prd);
        const float3 wiLocal = worldToLocal(lightSample.direction, tangent, Ns, biNormal);

        if(lightSample.pdf > 0.f && wiLocal.y > 0.0f)
        {
            // Visibility の評価は視点に非依存なので 1 回だけで OK
            ShadowPRD shadowPrd;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);

            const float3 wi = localToWorld(wiLocal, tangent, Ns, biNormal);
            
            // 光源へ接続して可視性を判断
            optixTrace( 
                optixLaunchParams.traversable,
                intersectedPoint, // 出射位置
                wi,
                1e-3f,
                lightSample.distance - 1e-3f,
                0.0f,
                OptixVisibilityMask( MASK_SURFACE ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,    // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW_RAY_TYPE,    // miss SBT Index
                u0, u1
            );


            if(shadowPrd.visible){
                // 波長の MIS
                // const float spectralMISWeight = hwssSpectralWeight(prd.logPOrefix);

                // constexpr int C = 4;
                // const float invC = 1.0f / (float)C;
                // const float uHero = prd.waveLengthNormalized;
                const float uNormalized = prd.waveLengthNormalized;


                const float pdfLight = fmaxf(lightSample.pdf, 1e-7f);

                // BSDF の計算
                int callableFunctionOffset = NUM_LENS_TYPE + material->materialType * 2 + 1; // 登録した順番が lens の後で，materialType * 2 + 1 が 対応するマテリアルの eval 関数 
                
                float bxdfValue = optixDirectCall<float, const float3, const float3, const IntersectedData_Spectral& , PRDSpectral*>(callableFunctionOffset, wiLocal, woLocal, matData, &prd);
                const float bxdfPdf = fmaxf(prd.pdf.bxdf, 1e-7f);

                // emission の取得
                float emission = lightSample.emissionRGB.y;
                const float D65 = tex2D<float>(optixLaunchParams.spectral.D65, u, 0.5f);
                emission *= (optixLaunchParams.light.lightIntensityFactor  * D65);

                const float misWeight = balanceHeuristicWeight(1, pdfLight, 1, bxdfPdf);
                prd.contribution += emission * bxdfValue * wiLocal.y * misWeight / pdfLight / prd.pdf.spectral;
                
                
                // // 波長ごとに BSDF を評価
                // for(int k = 0; k < C; ++k){
                //     const float u = wrap01(uHero + float(k) * invC);

                //     // 波長を一時的に差し替え
                //     prd.waveLengthNormalized = u;
                //     float bxdfValue = optixDirectCall<float, const float3, const float3, const IntersectedData_Spectral& , PRDSpectral*>(callableFunctionOffset, wiLocal, woLocal, matData, &prd);
                //     const float bxdfPdf = fmaxf(prd.pdf.bxdf, 1e-7f);
                //     // 戻す
                //     prd.waveLengthNormalized = uHero;

                //     // emission の取得
                //     float emission = lightSample.emissionRGB.y;
                //     const float D65 = tex2D<float>(optixLaunchParams.spectral.D65, u, 0.5f);
                //     emission *= (optixLaunchParams.light.lightIntensityFactor  * D65);

                //     const float directionalMISWeight = balanceHeuristicWeight(1, pdfLight, 1, bxdfPdf);

                //     // throughput
                //     float beta_k = (&prd.beta.x)[k];
                //     (&prd.contribution.x)[k] += emission * beta_k * bxdfValue * wiLocal.y * directionalMISWeight * spectralMISWeight / pdfLight;

                // }
                
            }
        }
        prd.continueTrace = true;
    }

    
    // 次の方向の決定
    // BSDF のサンプリング とアルベドの変更
    // constexpr int C = 4;
    // const float invC = 1.0f / (float)C;
    // const float uHero = prd.waveLengthNormalized;
    const float uNormalized = prd.waveLengthNormalized;

    // BSDF のサンプル (Hero のみ 1 回)

    int callableFunctionOffset = NUM_LENS_TYPE + material->materialType * 2;
    const float3 sampledWiLocal = optixDirectCall<float3, const float3, const IntersectedData_Spectral&, PRDSpectral*>(callableFunctionOffset, woLocal, matData, &prd);
    const float3 sampledWi = localToWorld(sampledWiLocal, tangent, Ns, biNormal);
    
    // if(sampledWiLocal.y < 1e-7f){
    //     prd.continueTrace = false;
    //     return;
    // }

    const float pdfHero = fmaxf(prd.pdf.bxdf, 1e-7f);
    // BSDF の評価
    callableFunctionOffset = NUM_LENS_TYPE + material->materialType * 2 + 1;
    const float bxdfValueHero = optixDirectCall<float, const float3, const float3, const IntersectedData_Spectral& , PRDSpectral*>(callableFunctionOffset, sampledWiLocal, woLocal, matData, &prd);


    // HWSS
    // prd.beta.x *= bxdfValueHero * sampledWiLocal.y / pdfHero;
    // prd.logPOrefix.x += logf(pdfHero);
    
    // k = 1 -- C-1
    // const float oldU = prd.waveLengthNormalized;
    // const float oldPdf = prd.pdf.bxdf;

    // for(int k = 1; k < C; ++k){
    //     const float u = wrap01(uHero + float(k) * invC);
    //     prd.waveLengthNormalized = u;
    //     const float fk = optixDirectCall<float, const float3, const float3, const IntersectedData_Spectral& , PRDSpectral*>(callableFunctionOffset, sampledWiLocal, woLocal, matData, &prd);
    //     const float pdfk = fmaxf(prd.pdf.bxdf, 1e-7f);
    //     (&prd.beta.x)[k] += fk * sampledWiLocal.y / pdfHero;
    //     (&prd.logPOrefix.x)[k] += logf(pdfk);
    // }

    // prd.waveLengthNormalized = uHero;
    // prd.pdf.bxdf = oldPdf;

    prd.position += 1e-3f * Ng;
    // prd.albedo *= bxdfValue * sampledWiLocal.y / fmaxf(prd.pdf.bxdf, 1e-7f);
    prd.wi = sampledWi; 
    prd.continueTrace = 1;
    
    // if(prd.bounce == 0){
    //     prd.primaryNormal = Ns;
    //     prd.primaryAlbedo = albedoRGB;
    // }
}