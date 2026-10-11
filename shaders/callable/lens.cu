#include "../config.cuh"
#include "m1/bounded_trig.hpp"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"

extern "C" __device__ LensRay __direct_callable__lens_pinhole(const float2 screen, const float sample1, const float sample2)
{
    LensRay ray;
    const auto &camera = optixLaunchParams.camera;
    const float sensorWidth     = 35.0f / 1000.0f; // m 単位に正規化
    const float aspect          = (float)optixLaunchParams.frame.size.x / (float)optixLaunchParams.frame.size.y;
    const float sensorHeight    = sensorWidth / aspect;

    const float fovY = camera.fov * float(M_PI) / 180.0f;
    const float focalLength = (sensorHeight / 2.0f) / m1::bounded_tan(fovY / 2.0f);

    const float3 sensorCenter = camera.position + normalize(camera.direction) * focalLength;

    const float3 sampledSensorGlobalPos = sensorCenter 
                - (screen.x - 0.5f) * normalize(camera.horizontal) * sensorWidth
                - (screen.y - 0.5f) * normalize(camera.vertical) * sensorHeight;
            ray.org = camera.position;
            ray.dir = normalize(sampledSensorGlobalPos - camera.position);
            ray.weight = 1.f;
    return ray;
}

extern "C" __device__ LensRay __direct_callable__lens_thinLens(const float2 screen,  const float sample1, const float sample2)
{
    LensRay ray;
    const auto &camera = optixLaunchParams.camera;
    const float sensorWidth     = 35.0f / 1000.0f; // m 単位に正規化
    const float sensorHeight    = sensorWidth * (float)optixLaunchParams.frame.size.y / (float)optixLaunchParams.frame.size.x; // mm 単位に正規化
    const float pint = optixLaunchParams.camera.pintDist;//m 単位に正規化
    const float focalLength = camera.focalLength / 1000.0f; // m 
    const float fValue = camera.fValue;
    const float sensorDist = (pint * focalLength) / (pint - focalLength);
    const float lensEffectiveRadius = focalLength / (camera.fValue *2.0f);

    const float3 lensCenterGlobalPos =  camera.position + sensorDist * camera.direction;
    const float2 sampledPointOnLens = random_unit_disk(sample1, sample2) * lensEffectiveRadius;
    const float3 sampledLensGlobalPos = lensCenterGlobalPos 
        + sampledPointOnLens.x * normalize(camera.horizontal) 
        + sampledPointOnLens.y * normalize(camera.vertical);

    const float3 lensSensorGlobalPos = camera.position;
    const float3 sampledSensorGlobalPos = lensSensorGlobalPos 
        + (screen.x - 0.5f) * normalize(camera.horizontal) * sensorWidth
        + (screen.y - 0.5f) * normalize(camera.vertical) * sensorHeight;

    const float3 dir = normalize(lensCenterGlobalPos - sampledSensorGlobalPos);
    const float3 focusPoint = lensCenterGlobalPos + (pint / dot(dir, camera.direction)) * dir;
            
    ray.org = sampledLensGlobalPos;
    ray.dir = normalize(focusPoint - sampledLensGlobalPos);
    ray.weight = powf(dot(ray.dir, camera.direction), 4.f) * lensEffectiveRadius / powf(dot(sampledLensGlobalPos - sampledSensorGlobalPos, camera.direction), 2.0f) * camera.sensitivity;
    return ray;
}
