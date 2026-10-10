#include <cuda_runtime.h>
#include "../utils/helper_math.h"

extern "C" __global__ void copyBufferToSurfaceKernel(uint32_t   *finalColorBuffer,
                                            cudaSurfaceObject_t surface,
                                            int2     size)
{
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= size.x) return;
    if (pixelY >= size.y) return;

    int pixelID = pixelX + size.x * pixelY;

    uint32_t rgba = finalColorBuffer[pixelID];
    uchar4 color = make_uchar4(
        (rgba >>  0) & 0xFF,
        (rgba >>  8) & 0xFF,
        (rgba >> 16) & 0xFF,
        (rgba >> 24) & 0xFF
    );

    surf2Dwrite(color, surface, static_cast<size_t>(pixelX) *sizeof(uchar4), pixelY);
    return;
}