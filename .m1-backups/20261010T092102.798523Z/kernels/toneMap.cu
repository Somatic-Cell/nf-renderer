#include <cuda_runtime.h>
#include "../utils/helper_math.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

__device__ float clamp01(float x) {
    return fminf(fmaxf(x, 0.f), 1.f);
}

inline __device__ float4 d_toneMapLocal(const float4 color, const float white){
    float luminance = 0.2f * color.x + 0.7f * color.y + 0.1f * color.z;
    return color *  (1.f + luminance / powf(white, 2)) / (1.f + luminance);
    // return color / (make_float4(1.0f) + color);
    // return  color / (make_float4(1.0f) + color) * (make_float4(1.0f) + color / pow(white, 2));
}

inline __device__ float4 d_gammacorrect(float4 color, const float gamma) {
    float c = 1.f / gamma;
    float tmp_x = powf(color.x, c);
    float tmp_y = powf(color.y, c);
    float tmp_z = powf(color.z, c);
    return make_float4(clamp(tmp_x, 0.f, 1.f), clamp(tmp_y, 0.f, 1.f), clamp(tmp_z, 0.f, 1.f), 1.0f);
}

extern "C" __global__ void computeFinalPixelColorsKernel(uint32_t *finalColorBuffer,
                                            float4  *denoisedBuffer,
                                            int2    size,
                                            float   white,
                                            float   exposure
                                        
                                        )
{
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= size.x) return;
    if (pixelY >= size.y) return;

    int pixelID = pixelX + size.x * pixelY;

    float4 f4 = denoisedBuffer[pixelID];
    f4 = clamp(d_gammacorrect(d_toneMapLocal(f4 * exposure, white), 2.2f), 0.0f, 1.0f);
    uint32_t rgba = 0;
    rgba |= (uint32_t)(f4.x * 255.9f) <<  0;
    rgba |= (uint32_t)(f4.y * 255.9f) <<  8;
    rgba |= (uint32_t)(f4.z * 255.9f) << 16;
    rgba |= (uint32_t)255             << 24;
    finalColorBuffer[pixelID] = rgba;
    return;
}