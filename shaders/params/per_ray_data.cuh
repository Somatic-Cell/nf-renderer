#ifndef PER_RAY_DATA_CUH_
#define PER_RAY_DATA_CUH_

#include "../config.cuh"
#include <optix_device.h>

struct PRDRGB {
    Random random;

    float3 position;
    float3 wo;          // outgoing direction (カメラ方向，ワールド空間)
    float3 wi;          // incoming direction (光源方向，ワールド空間)

    float3 albedo;
    float3 primaryNormal;
    float3 primaryAlbedo;
    float3 contribution;
    
    struct{
        float light;
        float bxdf;
    } pdf;
    
    int bounce;
    int lastHitMaterialType;
    bool continueTrace;

    uint instanceID;
};

struct PRDSpectral {
    Random random;

    float3 position;
    // float3 wo;          // outgoing direction (カメラ方向，ワールド空間)
    float3 wi;          // incoming direction (光源方向，ワールド空間)

    float  albedo;
    // float3 primaryNormal;
    // float3 primaryAlbedo;
    float contribution;

    float  waveLength; // one wavelength per path, in nm on the source table grid
    float  waveLengthNormalized; // 0-1, テクスチャフェッチ用
    
    struct{
        float bxdf;
        float spectral;
    } pdf;
    
    int bounce;
    // int lastHitMaterialType;
    uint32_t continueTrace;

    uint instanceID;

    // M1 uses albedo as scalar path throughput and contribution as accumulated
    // spectral radiance. Wavelength PDF compensation is applied once in raygen.
    float m1EtaScale=1.0f;
    float3 m1PreviousPosition{};
    float3 m1PrimaryNormal{};
    float3 m1PrimaryAlbedo{};
    bool m1PreviousDelta=true;
    bool m1Invalid=false;

    // for HWSS
    // float4 beta;
    // float4 logPOrefix;
};

struct ShadowPRD {
    Random random;
    bool visible    {false};
    bool m1Invalid  {false};
};


static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr; 
}

static __forceinline__ __device__
void packPointer( void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer(u0, u1));
}

#endif // PER_RAY_DATA_CUH_
