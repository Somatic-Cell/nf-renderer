#pragma once
#include "asset.hpp"
#include "coop_types.hpp"
#include <cuda.h>
#include <optix.h>
#if !defined(OPTIX_VERSION) || OPTIX_VERSION < 90000
#error "The M4 CoopVec backend requires OptiX SDK 9.0 or newer."
#endif
namespace m4nf {
// Owns only new NF allocations; never modifies the renderer's context/stream.
// Release before destroying the CUDA context. Copying is deliberately disabled.
class GpuAsset {
public:
    GpuAsset()=default;
    ~GpuAsset();
    GpuAsset(const GpuAsset&)=delete;
    GpuAsset& operator=(const GpuAsset&)=delete;
    void upload(OptixDeviceContext optix,CUstream stream,const Asset& asset);
    void release() noexcept;
    ModelView model{};
    CoopView coop{};
private:
    CUdeviceptr fields_=0,packed_=0;
};
}
