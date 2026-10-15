#pragma once
#include "asset.hpp"
#include <cuda.h>
namespace m4mie {
// Transfers the validated, read-only Mie table once to the renderer context.
// Does not own or modify the stream/context. Release before context teardown.
class GpuAsset {
public:
    GpuAsset()=default;
    ~GpuAsset(){release();}
    GpuAsset(const GpuAsset&)=delete;
    GpuAsset& operator=(const GpuAsset&)=delete;
    void upload(CUstream stream,const Asset& asset);
    void release() noexcept;
    View view{};
private:
    CUdeviceptr buffer_=0;
};
}
