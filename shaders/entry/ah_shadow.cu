#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include "../../include/m2/alpha.hpp"
#include "../device/m1_diagnostics.cuh"
#include <stdio.h>

extern "C" __global__ void __anyhit__shadow()
{
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t meshIndex = sbtData.tri.meshIndex;
    const uint32_t materialIndex = sbtData.tri.materialIndex;
    ShadowPRD &prd = *getPRD<ShadowPRD>();
    if(optixLaunchParams.m1.enabled &&
       (sbtData.geomType!=GeomType::Triangle || meshIndex>=unsigned(optixLaunchParams.numMeshes) ||
        materialIndex>=unsigned(optixLaunchParams.numMaterials))) {
        m1diagnostics::fail_shadow(prd,sbtData,m1::FailureReason::ShadowSbt,
            {{float(sbtData.geomType),float(optixLaunchParams.numMeshes),float(optixLaunchParams.numMaterials)}});
        optixTerminateRay();
        return;
    }
    const TriangleMeshGeomData* __restrict__ mesh = &optixLaunchParams.meshes[meshIndex];
    const MaterialData* __restrict__ material = &optixLaunchParams.materials[materialIndex];
   
    if(!optixLaunchParams.m1.enabled && material->materialType == MATERIAL_TYPE_GLASS){
        optixIgnoreIntersection();
        return;
    }

    if(optixLaunchParams.m1.enabled && material->diffuseTexture.texture && !mesh->texcoord) {
        m1diagnostics::fail_shadow(prd,sbtData,m1::FailureReason::ShadowUV,{{1}});
        optixTerminateRay();
        return;
    }

    float opacity = 1.0f;
    float2 diagnosticUV=make_float2(0);
    if(material->diffuseTexture.texture > 0){
        // 基本的な交差点の情報を取得
        const int primID = optixGetPrimitiveIndex();
        const uint3 index = mesh->index[primID];

        const float2 uv = optixGetTriangleBarycentrics();
        const float u = uv.x;
        const float v = uv.y;
        
        // Diffuse テクスチャ座標を取得
        const float2* __restrict__ texcoord = mesh->texcoord;
        const float2 &UVDiffuse1 = texcoord[index.x];
        const float2 &UVDiffuse2 = texcoord[index.y];
        const float2 &UVDiffuse3 = texcoord[index.z];

        const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
            + u * UVDiffuse2
            + v * UVDiffuse3;

        diagnosticUV=diffuseTextureCoordinate;
        opacity = tex2D<float4>(material->diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;
    }

    if(optixLaunchParams.m1.enabled) {
        float coverage=0;
        if(!m2::surface_coverage(opacity,material->opacity,material->alphaCutoff,coverage)) {
            m1diagnostics::fail_shadow(prd,sbtData,m1::FailureReason::ShadowOpacity,
                {{opacity,material->opacity,material->alphaCutoff,diagnosticUV.x,diagnosticUV.y}});
            optixTerminateRay();
            return;
        }
        opacity=coverage;
    }
    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }
}
