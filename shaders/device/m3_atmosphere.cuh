#pragma once

#include "../params/per_ray_data.cuh"
#include "shader_common.cuh"
#include "m1_diagnostics.cuh"
#include "../../include/m1/math.hpp"
#include "../../include/m3/tracking.hpp"
#include "../../include/m3/transport.hpp"

namespace m3device {
namespace medium = m3::tracking;

// The midpoint of each 23-bit cell is exactly representable in FP32 and inside
// (0,1). Adding half a 24-bit cell to a float can instead round the maximum to 1.
struct OpenRandom {
    Random& random;
    __forceinline__ __device__ float operator()() {
        return (float(random.next_u32() >> 9u) + 0.5f) * (1.0f / 8388608.0f);
    }
};

static __forceinline__ __device__ medium::RaySegment ray(
    m1::Vec3 origin,m1::Vec3 direction,float t_max) {
    return {origin,direction,t_max};
}

static __forceinline__ __device__ void geometry_failure(
    PRDSpectral& p,const medium::RaySegment& r,medium::Status status) {
    m1diagnostics::fail(p,m1::FailureReason::AtmosphereGeometry,
        {{float(status),r.t_max,r.origin.x,r.origin.y,r.origin.z,
          r.direction.x,r.direction.y,r.direction.z}});
}

static __forceinline__ __device__ bool coefficient(PRDSpectral& p,float& beta) {
    if(medium::rayleigh_coefficient(optixLaunchParams.rayleigh,p.waveLength,beta))return true;
    m1diagnostics::fail(p,m1::FailureReason::AtmosphereCoefficient,
        {{p.waveLength,optixLaunchParams.rayleigh.beta550}});
    return false;
}

// t_max is the parameter of the raw float ray, not an assumed metric length.
// This is used ONLY by explicit light connections. Sampled path continuation
// already includes free-flight survival and must never multiply this T again.
static __forceinline__ __device__ float connection_transmittance(PRDSpectral& p,
    m1::Vec3 origin,m1::Vec3 direction,float t_max,bool include_attenuation=true) {
    const auto& a=optixLaunchParams.rayleigh;
    if(!a.enabled)return 1.0f;
    const auto r=ray(origin,direction,t_max);
    if(!include_attenuation) {
        // The surface-to-area approximation still respects the opaque planet.
        const auto segment=medium::clip(a,r);
        if(segment.status!=medium::Status::Ok){geometry_failure(p,r,segment.status);return 0.0f;}
        return segment.ground_occludes?0.0f:1.0f;
    }
    float beta=0.0f;
    if(!coefficient(p,beta))return 0;
    OpenRandom rng{p.random};
    const auto result=medium::ratio_transmittance(a,r,beta,rng);
    if(result.status!=medium::Status::Ok){
        m1diagnostics::fail(p,m1::FailureReason::AtmosphereRatio,
            {{float(result.status),result.clip.begin_t,result.clip.end_t,result.transmittance,
              float(result.candidates),float(result.roulette_tests),p.waveLength}});
        return 0;
    }
    return result.transmittance;
}

} // namespace m3device
