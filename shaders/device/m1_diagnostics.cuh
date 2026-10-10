#pragma once

#include "../params/per_ray_data.cuh"

namespace m1diagnostics {

static __forceinline__ __device__ void set_hit_context(
    PRDSpectral& p,m1::FailureStage stage,const HitgroupSBTData& sbt) {
    p.m1FailureStage=stage;
    p.m1FailureContext={sbt.tri.meshIndex,sbt.tri.materialIndex,
                       optixGetInstanceId(),optixGetPrimitiveIndex()};
}

static __forceinline__ __device__ void fail_with_ray(PRDSpectral& p,
    m1::FailureReason reason,m1::FailureValues values,float3 position,float3 direction) {
    if(p.m1Invalid)return; // Preserve the first failure, including shadow errors.
    p.m1Invalid=true;
    p.continueTrace=0;
    auto* buffer=optixLaunchParams.m1.failure;
    if(!buffer || atomicCAS(&buffer->claimed,0u,1u)!=0u)return;
    m1::FailureRecord record;
    record.reason=reason;
    record.stage=p.m1FailureStage;
    record.context=p.m1FailureContext;
    record.pixel_x=optixGetLaunchIndex().x;
    record.pixel_y=optixGetLaunchIndex().y;
    record.sample=p.m1SampleIndex;
    record.frame=optixLaunchParams.frame.frameID;
    record.seed=optixLaunchParams.m1.seed;
    record.bounce=p.bounce;
    record.wavelength_nm=p.waveLength;
    record.throughput=p.albedo;
    record.radiance=p.contribution;
    record.bsdf_pdf=p.pdf.bxdf;
    record.eta_scale=p.m1EtaScale;
    record.current_position[0]=position.x;
    record.current_position[1]=position.y;
    record.current_position[2]=position.z;
    record.current_direction[0]=direction.x;
    record.current_direction[1]=direction.y;
    record.current_direction[2]=direction.z;
    record.values=values;
    buffer->first=record;
}

static __forceinline__ __device__ void fail(PRDSpectral& p,
    m1::FailureReason reason,m1::FailureValues values={}) {
    fail_with_ray(p,reason,values,p.position,p.wi);
}

static __forceinline__ __device__ void fail_shadow(ShadowPRD& shadow,
    const HitgroupSBTData& sbt,m1::FailureReason reason,m1::FailureValues values={}) {
    shadow.m1Invalid=true;
    if(shadow.m1Path) {
        set_hit_context(*shadow.m1Path,m1::FailureStage::ShadowAnyHit,sbt);
        fail_with_ray(*shadow.m1Path,reason,values,
                      optixGetWorldRayOrigin(),optixGetWorldRayDirection());
    }
}

} // namespace m1diagnostics
