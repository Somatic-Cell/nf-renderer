#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __anyhit__radiance_rgb()
{
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t meshIndex = sbtData.tri.meshIndex;
    const uint32_t materialIndex = sbtData.tri.materialIndex;
    const TriangleMeshGeomData &mesh = optixLaunchParams.meshes[meshIndex];
    const MaterialData & material = optixLaunchParams.materials[materialIndex];
  
    PRDRGB &prd = *getPRD<PRDRGB>();
    
    float opacity = 1.0f;     
    if(material.diffuseTexture.texture > 0){
        // 基本的な交差点の情報を取得
        const int primID = optixGetPrimitiveIndex();
        const uint3 index = mesh.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Diffuse テクスチャ座標を取得
        const float2 &UVDiffuse1 = mesh.texcoord[index.x];
        const float2 &UVDiffuse2 = mesh.texcoord[index.y];
        const float2 &UVDiffuse3 = mesh.texcoord[index.z];

        const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
            + u * UVDiffuse2
            + v * UVDiffuse3;

        opacity = tex2D<float4>(material.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;
    }

    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }
}


extern "C" __global__ void __anyhit__radiance_spectral()
{
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t meshIndex = sbtData.tri.meshIndex;
    const uint32_t materialIndex = sbtData.tri.materialIndex;
    const TriangleMeshGeomData &mesh = optixLaunchParams.meshes[meshIndex];
    const MaterialData & material = optixLaunchParams.materials[materialIndex];

    PRDSpectral &prd = *getPRD<PRDSpectral>();
    
    float opacity = 1.0f;     
    if(material.diffuseTexture.texture > 0){
        // 基本的な交差点の情報を取得
        const int primID = optixGetPrimitiveIndex();
        const uint3 index = mesh.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Diffuse テクスチャ座標を取得
        const float2 &UVDiffuse1 = mesh.texcoord[index.x];
        const float2 &UVDiffuse2 = mesh.texcoord[index.y];
        const float2 &UVDiffuse3 = mesh.texcoord[index.z];

        const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
            + u * UVDiffuse2
            + v * UVDiffuse3;

        opacity = tex2D<float4>(material.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;
    }

    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }
}
