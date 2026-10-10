#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __anyhit__shadow()
{
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t meshIndex = sbtData.tri.meshIndex;
    const uint32_t materialIndex = sbtData.tri.materialIndex;
    const TriangleMeshGeomData* __restrict__ mesh = &optixLaunchParams.meshes[meshIndex];
    const MaterialData* __restrict__ material = &optixLaunchParams.materials[materialIndex];
   
    if(material->materialType == MATERIAL_TYPE_GLASS){
        optixIgnoreIntersection();
        return;
    }
    ShadowPRD &prd = *getPRD<ShadowPRD>();

    float opacity = 1.0f;     
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

        opacity = tex2D<float4>(material->diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;
    }

    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }
}