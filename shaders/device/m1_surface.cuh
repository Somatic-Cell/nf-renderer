#pragma once

#include "m1/bounded_trig.hpp"

// M1/M2 surfaces and the optional M3 Rayleigh atmosphere share the existing
// launch symbol, pointer payload, mesh/instance SBT and RGB framebuffer.
#include "../config.cuh"
#include "../params/per_ray_data.cuh"
#include "shader_common.cuh"
#include "m1_diagnostics.cuh"
#include "m3_atmosphere.cuh"
#include "../../include/m1/bsdf.hpp"
#include "../../include/m1/light_pdf.hpp"
#include "../../include/m1/rgb2spec_device.hpp"
#include "../../include/m2/alpha.hpp"
#include "../../include/m2/tangent_frame.hpp"
#include <math_constants.h>

namespace m1surface {
using m1::Vec3;
using m1::FailureReason;
using m1::FailureStage;
using m1diagnostics::fail;

static __forceinline__ __device__ Vec3 vector(float3 v){return {v.x,v.y,v.z};}
static __forceinline__ __device__ float3 vector(Vec3 v){return make_float3(v.x,v.y,v.z);}
static __forceinline__ __device__ bool finite3(float3 v){return m1::finite(vector(v));}

static __forceinline__ __device__ float reflectance(float3 rgb,float wavelength,PRDSpectral& p) {
    const auto& m=optixLaunchParams.m1;
    const m1::Rgb2SpecTableView table{m.rgb2specScale,m.rgb2specData,m.rgb2specResolution};
    float value=0;
    if(!m1::rgb2spec_reflectance_device(table,rgb.x,rgb.y,rgb.z,wavelength,value))
        fail(p,FailureReason::ReflectanceSpectrum,{{rgb.x,rgb.y,rgb.z,wavelength,value}});
    return value;
}

static __forceinline__ __device__ float d65(float wavelength) {
    const auto* table=optixLaunchParams.m1.d65;
    const unsigned count=optixLaunchParams.m1.d65Count;
    if(count<2||wavelength<table[0].wavelength_nm||wavelength>table[count-1].wavelength_nm)return 0;
    unsigned lo=0,hi=count-1;
    while(hi-lo>1){const unsigned mid=(hi+lo)/2;if(wavelength<table[mid].wavelength_nm)hi=mid;else lo=mid;}
    const float t=(wavelength-table[lo].wavelength_nm)/(table[hi].wavelength_nm-table[lo].wavelength_nm);
    return table[lo].value+t*(table[hi].value-table[lo].value);
}

// Explicit D65-based RGB illuminant model (not a measured emission spectrum).
// D65 is normalized on 360..830 nm so integral(D65*ybar)/integral(ybar) = 1.
// Separate 2*max(RGB) amplitude from its bounded chromatic shape, as in PBRT's
// RGB illuminant reconstruction. The scene intensity supplies radiance units.
static __forceinline__ __device__ float emission(float3 rgb,float wavelength,PRDSpectral& p) {
    if(!finite3(rgb)||rgb.x<0||rgb.y<0||rgb.z<0){
        fail(p,FailureReason::EmissionRgb,{{rgb.x,rgb.y,rgb.z}});return 0;
    }
    const float amplitude=2.0f*fmaxf(rgb.x,fmaxf(rgb.y,rgb.z));
    if(amplitude==0)return 0;
    if(!m1::finite(amplitude)){
        fail(p,FailureReason::EmissionAmplitude,{{rgb.x,rgb.y,rgb.z,amplitude}});return 0;
    }
    const float value=amplitude*reflectance(rgb/amplitude,wavelength,p)*
        d65(wavelength)*optixLaunchParams.light.lightIntensityFactor;
    if(!m1::finite(value)||value<0)
        fail(p,FailureReason::EmissionSpectrum,
             {{value,amplitude,d65(wavelength),optixLaunchParams.light.lightIntensityFactor}});
    return value;
}

static __forceinline__ __device__ float3 cie(float wavelength) {
    const auto* table=optixLaunchParams.m1.cie;
    const unsigned count=optixLaunchParams.m1.cieCount;
    if(count<2||wavelength<table[0].wavelength_nm||wavelength>table[count-1].wavelength_nm)
        return make_float3(0);
    unsigned lo=0,hi=count-1;
    while(hi-lo>1){const unsigned mid=(hi+lo)/2;if(wavelength<table[mid].wavelength_nm)hi=mid;else lo=mid;}
    const float t=(wavelength-table[lo].wavelength_nm)/(table[hi].wavelength_nm-table[lo].wavelength_nm);
    return make_float3(table[lo].x+t*(table[hi].x-table[lo].x),
                       table[lo].y+t*(table[hi].y-table[lo].y),
                       table[lo].z+t*(table[hi].z-table[lo].z));
}

static __forceinline__ __device__ float solar_irradiance(float wavelength) {
    const auto* table=optixLaunchParams.m1.solar;
    const unsigned count=optixLaunchParams.m1.solarCount;
    if(count<2||wavelength<table[0].wavelength_nm||wavelength>table[count-1].wavelength_nm)return 0;
    unsigned lo=0,hi=count-1;
    while(hi-lo>1){const unsigned mid=(hi+lo)/2;if(wavelength<table[mid].wavelength_nm)hi=mid;else lo=mid;}
    const float t=(wavelength-table[lo].wavelength_nm)/(table[hi].wavelength_nm-table[lo].wavelength_nm);
    return table[lo].irradiance_w_m2_nm+t*(table[hi].irradiance_w_m2_nm-table[lo].irradiance_w_m2_nm);
}

static __forceinline__ __device__ Vec3 sun_direction() {
    const float z=optixLaunchParams.sunParams.sunZenithRad;
    const float a=optixLaunchParams.sunParams.sunAzimuthRad;
    const auto zenith=m1::bounded_sincos(z),azimuth=m1::bounded_sincos(a);
    return m1::normalize(Vec3{zenith.sine*azimuth.cosine,zenith.cosine,zenith.sine*azimuth.sine});
}
static __forceinline__ __device__ float sun_pdf() {
    const float s=m1::bounded_sin(0.5f*optixLaunchParams.m1.sunAngularRadiusRad);
    return 1.0f/(4.0f*m1::pi*s*s);
}
static __forceinline__ __device__ float sun_radiance(float wavelength) {
    const float s=m1::bounded_sin(optixLaunchParams.m1.sunAngularRadiusRad);
    // E = integral_disk L*cos(theta) dOmega = pi*L*sin^2(alpha).
    return optixLaunchParams.m1.sunScale*solar_irradiance(wavelength)/(m1::pi*s*s);
}
static __forceinline__ __device__ bool inside_sun(Vec3 direction) {
    const Vec3 d=direction-sun_direction();
    const float half=m1::bounded_sin(0.5f*optixLaunchParams.m1.sunAngularRadiusRad);
    // Chord distance avoids subtracting two nearly equal cosines at 0.25 deg.
    // There is no PDF floor or angular support expansion by a tolerance.
    return fmaf(d.x,d.x,fmaf(d.y,d.y,d.z*d.z))<=4.0f*half*half;
}
static __forceinline__ __device__ Vec3 sample_sun(float u,float v) {
    const float s=m1::bounded_sin(0.5f*optixLaunchParams.m1.sunAngularRadiusRad);
    const float q=u*(2.0f*s*s),r=sqrtf(q*(2.0f-q)),phi=2.0f*m1::pi*v;
    const auto frame=m1::make_frame(sun_direction());
    const auto azimuth=m1::bounded_sincos(phi);
    return m1::normalize(frame.to_world({r*azimuth.cosine,r*azimuth.sine,1.0f-q}));
}

static __forceinline__ __device__ Vec3 offset(Vec3 p,Vec3 normal,Vec3 direction) {
    const float side=m1::dot(normal,direction)>=0 ? 1.0f : -1.0f;
    const Vec3 n=normal*side;
    Vec3 result=p+n*optixLaunchParams.m1.rayEpsilon;
    // Preserve at least one ULP of displacement when world coordinates are large.
    if(n.x!=0)result.x=nextafterf(result.x,n.x>0?CUDART_INF_F:-CUDART_INF_F);
    if(n.y!=0)result.y=nextafterf(result.y,n.y>0?CUDART_INF_F:-CUDART_INF_F);
    if(n.z!=0)result.z=nextafterf(result.z,n.z>0?CUDART_INF_F:-CUDART_INF_F);
    return result;
}

static __forceinline__ __device__ bool visible(PRDSpectral& p,Vec3 origin,Vec3 direction,float distance) {
    if(!m1::finite(origin)||!m1::bsdf_unit_direction(direction)||!(distance>0)){
        fail(p,FailureReason::ShadowRay,{{origin.x,origin.y,origin.z,
            direction.x,direction.y,direction.z,distance,m1::dot(direction,direction)}});return false;
    }
    // A mesh-free M3 scene still contains the analytic planet and atmosphere.
    // Never pass the null IAS handle to optixTrace.
    if(!optixLaunchParams.traversable)return true;
    ShadowPRD shadow;shadow.random=p.random;shadow.visible=false;shadow.m1Invalid=false;shadow.m1Path=&p;
    unsigned u0,u1;packPointer(&shadow,u0,u1);
    optixTrace(optixLaunchParams.traversable,vector(origin),vector(direction),0.0f,distance,0.0f,
               OptixVisibilityMask(MASK_SURFACE),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT|OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               SHADOW_RAY_TYPE,RAY_TYPE_COUNT,SHADOW_RAY_TYPE,u0,u1);
    p.random=shadow.random;
    if(shadow.m1Invalid)fail(p,FailureReason::ShadowUnhandled,{{1}});
    return shadow.visible&&!shadow.m1Invalid;
}

struct SurfaceBsdf {
    unsigned type=0;
    m1::BsdfParameters simple{};
    m1::MetallicParameters metallic{};
    __forceinline__ __device__ m1::BsdfEvaluation evaluate(Vec3 wo,Vec3 wi) const {
        return type==MATERIAL_TYPE_PRINCIPLED_BRDF ? m1::evaluate_metallic(metallic,wo,wi) :
                                                   m1::evaluate_bsdf(simple,wo,wi);
    }
    __forceinline__ __device__ m1::BsdfSample sample(Vec3 wo,float u,float v,float w,PRDSpectral& p) const {
        const auto result=type==MATERIAL_TYPE_PRINCIPLED_BRDF ? m1::sample_metallic(metallic,wo,u,v,w) :
                                                              m1::sample_bsdf(simple,wo,u,v,w);
        if(result.status==m1::BsdfSampleStatus::Invalid)
            fail(p,FailureReason::BsdfSample,{{wo.x,wo.y,wo.z,u,v,w,
                type==MATERIAL_TYPE_PRINCIPLED_BRDF?metallic.color:simple.color,
                type==MATERIAL_TYPE_PRINCIPLED_BRDF?metallic.metallic:0,
                type==MATERIAL_TYPE_PRINCIPLED_BRDF?metallic.roughness:simple.alpha,
                result.wi.x,result.wi.y,result.wi.z,result.pdf,result.weight,result.eta_scale,simple.eta_i_over_t}});
        return result;
    }
};

static __forceinline__ __device__ float light_hit_weight(const PRDSpectral& p,float pdf_light) {
    return m3::sun_emission_weight(p.bounce,p.m1PreviousDelta,
        optixLaunchParams.m1.directMode,p.pdf.bxdf,pdf_light);
}
static __forceinline__ __device__ void add_direct(PRDSpectral& p,const SurfaceBsdf& bsdf,
                                                 Vec3 wo,Vec3 wi,float radiance,float pdf_light) {
    if(wi.z<=0||radiance==0)return;
    const auto v=bsdf.evaluate(wo,wi);
    if(!v.valid) {
        fail(p,FailureReason::DirectBsdf,{{wo.x,wo.y,wo.z,wi.x,wi.y,wi.z,
            m1::dot(wo,wo),m1::dot(wi,wi),v.f,v.pdf,
            bsdf.type==MATERIAL_TYPE_PRINCIPLED_BRDF?bsdf.metallic.color:bsdf.simple.color,
            bsdf.metallic.metallic,bsdf.metallic.roughness}});return;
    }
    if(!m1::finite(radiance)||radiance<0||!m1::finite(pdf_light)||!(pdf_light>0)){
        fail(p,FailureReason::DirectInputs,{{radiance,pdf_light}});return;
    }
    const float weight=m3::nee_weight(optixLaunchParams.m1.directMode,pdf_light,v.pdf);
    p.contribution+=p.albedo*v.f*wi.z*radiance*weight/pdf_light;
    if(!m1::finite(p.contribution)||p.contribution<0)
        fail(p,FailureReason::DirectContribution,{{radiance,pdf_light,v.f,v.pdf,wi.z,weight}});
}

static __forceinline__ __device__ int find_emitter(unsigned mesh,unsigned instance,unsigned primitive) {
    // createLightTable() appends triangles in (flat mesh, instance, primitive)
    // order. Binary search preserves the exact per-triangle PDF without a
    // linear scan through every emissive triangle at each BSDF light hit.
    int lo=0,hi=optixLaunchParams.light.numLights;
    while(lo<hi) {
        const int mid=lo+(hi-lo)/2;
        const auto& e=optixLaunchParams.light.triangleLightData[mid];
        const bool less=e.meshIndex<mesh ||
            (e.meshIndex==mesh && (e.instanceID<instance ||
             (e.instanceID==instance && e.primitiveIndex<primitive)));
        if(less)lo=mid+1;else hi=mid;
    }
    if(lo<optixLaunchParams.light.numLights) {
        const auto& e=optixLaunchParams.light.triangleLightData[lo];
        if(e.meshIndex==mesh&&e.instanceID==instance&&e.primitiveIndex==primitive)return lo;
    }
    return -1;
}

static __forceinline__ __device__ float opacity(TextureSlot texture,float2 uv,
    float factor,float cutoff,PRDSpectral& p) {
    const float a=texture.texture ? tex2D<float4>(texture.texture,uv.x,1.0f-uv.y).w : 1.0f;
    float coverage=0;
    if(!m2::surface_coverage(a,factor,cutoff,coverage)){
        fail(p,FailureReason::SurfaceOpacity,{{a,factor,cutoff,uv.x,uv.y}});return 0;
    }
    return coverage;
}

static __forceinline__ __device__ Vec3 shading_normal(const TriangleMeshGeomData& mesh,
    const MaterialData& material,uint3 ind,float b0,float b1,float b2,Vec3 ng,
    Vec3 edge1,Vec3 edge2,PRDSpectral& p) {
    Vec3 ns=ng;
    Vec3 object_normal{};
    if(mesh.hasNormal) {
        if(!mesh.normal){fail(p,FailureReason::MissingNormals,{{float(mesh.hasNormal)}});return ng;}
        object_normal=vector(b0*mesh.normal[ind.x]+b1*mesh.normal[ind.y]+b2*mesh.normal[ind.z]);
        ns=m1::normalize(vector(optixTransformNormalFromObjectToWorldSpace(vector(object_normal))));
        if(!m1::bsdf_unit_direction(ns)){
            fail(p,FailureReason::InterpolatedNormal,{{ns.x,ns.y,ns.z,
                object_normal.x,object_normal.y,object_normal.z,b0,b1,b2}});return ng;
        }
    }
    if(m1::dot(ns,ng)<0)ns=-ns;
    if(material.normalTexture.texture) {
        if(!mesh.hasTangent||!mesh.tangent||!mesh.hasNormal||!mesh.normalTexcoord){
            fail(p,FailureReason::MissingTangentFrame,{{float(mesh.hasTangent),float(mesh.hasNormal)}});return ng;
        }
        const float2 uv=b0*mesh.normalTexcoord[ind.x]+b1*mesh.normalTexcoord[ind.y]+b2*mesh.normalTexcoord[ind.z];
        const float4 raw=tex2D<float4>(material.normalTexture.texture,uv.x,1.0f-uv.y);
        if(!isfinite(raw.x)||!isfinite(raw.y)||raw.x<0||raw.x>1||raw.y<0||raw.y>1){
            fail(p,FailureReason::NormalTexture,{{raw.x,raw.y,uv.x,uv.y}});return ng;
        }
        const float4 t0=mesh.tangent[ind.x],t1=mesh.tangent[ind.y],t2=mesh.tangent[ind.z];
        const float4 tangent=b0*t0+b1*t1+b2*t2;
        if((t0.w!=1&&t0.w!=-1)||(t1.w!=1&&t1.w!=-1)||(t2.w!=1&&t2.w!=-1)){
            fail(p,FailureReason::TangentHandedness,
                {{tangent.w,t0.w,t1.w,t2.w,b0,b1,b2}});return ng;
        }
        const Vec3 object_tangent=vector(make_float3(tangent));
        Vec3 t{},b{};
        if(t0.w!=t1.w||t0.w!=t2.w) {
            // Handedness is a discrete orientation, not a barycentrically
            // interpolated scalar. Opposite vertex signs can sum to exactly
            // zero on a valid triangle. Use this triangle's UV derivatives for
            // mixed-sign faces, with the pinned Assimp +U/-V convention.
            const float2 uv0=mesh.normalTexcoord[ind.x],uv1=mesh.normalTexcoord[ind.y],uv2=mesh.normalTexcoord[ind.z];
            const auto basis=m2::triangle_uv_tangent_frame(edge1,edge2,
                {uv0.x,uv0.y},{uv1.x,uv1.y},{uv2.x,uv2.y},ns);
            if(!basis.valid) {
                fail(p,FailureReason::TriangleTangentFrame,{{basis.uv_determinant,float(basis.uv_determinant_exponent),ns.x,ns.y,ns.z,
                    uv0.x,uv0.y,uv1.x,uv1.y,uv2.x,uv2.y,
                    m1::fp32::max_abs(edge1),m1::fp32::max_abs(edge2)}});return ng;
            }
            t=basis.tangent;b=basis.bitangent;
        } else {
            t=vector(optixTransformVectorFromObjectToWorldSpace(vector(object_tangent)));
            t=m1::normalize(t-ns*m1::dot(ns,t));
            const Vec3 object_b=m1::cross(object_normal,object_tangent)*t0.w;
            const Vec3 world_b=vector(optixTransformVectorFromObjectToWorldSpace(vector(object_b)));
            const Vec3 cross_b=m1::cross(ns,t);
            b=m1::dot(cross_b,world_b)>=0?cross_b:-cross_b;
        }
        if(!m1::bsdf_unit_direction(t)||!m1::bsdf_unit_direction(b)){
            fail(p,FailureReason::TangentFrame,{{t.x,t.y,t.z,b.x,b.y,b.z,ns.x,ns.y,ns.z,
                m1::dot(t,t),m1::dot(b,b),object_tangent.x,object_tangent.y,object_tangent.z}});return ng;
        }
        // Existing two-channel normal-map convention: signed XY, flipped green,
        // positive Z reconstructed from XY. Normalize after texture filtering.
        const float x=2.0f*raw.x-1.0f,y=1.0f-2.0f*raw.y;
        const float z=sqrtf(fmaxf(0.0f,1.0f-x*x-y*y));
        ns=m1::normalize(x*t+y*b+z*ns);
        if(!m1::bsdf_unit_direction(ns)){
            fail(p,FailureReason::MappedNormal,{{ns.x,ns.y,ns.z,x,y,z}});return ng;
        }
        if(m1::dot(ns,ng)<0)ns=-ns;
    }
    return ns;
}

// Surface polygons and the analytic planet use the same BSDF, NEE and RR path.
static __forceinline__ __device__ void scatter_surface(PRDSpectral& p,Vec3 position,
    Vec3 ng,Vec3 normal,Vec3 ns,Vec3 ray_direction,const SurfaceBsdf& bsdf) {
    const auto oriented=m1::make_bsdf_frame(ns,-ray_direction);
    const auto& frame=oriented.frame;
    const Vec3 wo=oriented.wo;
    p.position=vector(position);

    if(bsdf.type!=MATERIAL_TYPE_GLASS&&optixLaunchParams.m1.directMode!=1) {
        if(optixLaunchParams.m1.sunEnabled) {
            p.m1FailureStage=FailureStage::SunNee;
            const Vec3 direction=sample_sun(p.random(),p.random());
            const Vec3 local=frame.to_local(direction);
            const float le=sun_radiance(p.waveLength),pdf=sun_pdf();
            if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)){
                fail(p,FailureReason::SunParameters,{{le,pdf,optixLaunchParams.m1.sunAngularRadiusRad,
                     optixLaunchParams.m1.sunScale}});return;
            }
            const Vec3 shadow_origin=offset(position,ng,direction);
            if(local.z>0&&m1::dot(normal,direction)>0&&le>0&&
               visible(p,shadow_origin,direction,CUDART_INF_F)) {
                const float transmittance=m3device::connection_transmittance(
                    p,shadow_origin,direction,CUDART_INF_F);
                if(!p.m1Invalid)add_direct(p,bsdf,wo,local,le*transmittance,pdf);
            }
            if(p.m1Invalid)return;
        }
        const int count=optixLaunchParams.light.numLights;
        if(count>0) {
            p.m1FailureStage=FailureStage::AreaNee;
            const int selected=int(p.random()*count);
            if(selected<0||selected>=count){
                fail(p,FailureReason::LightSelection,{{float(selected),float(count)}});return;
            }
            const auto definition=optixLaunchParams.light.lightDefinition[selected];
            if(definition.lightType!=LIGHT_TYPE_TRIANGLE){
                fail(p,FailureReason::LightType,{{float(selected),float(definition.lightType),
                     float(definition.lightIndexInType)}});return;
            }
            const auto& e=optixLaunchParams.light.triangleLightData[definition.lightIndexInType];
            const float root=sqrtf(p.random()),b1=root*(1-p.random()),b2=root-b1,a0=1-root;
            const Vec3 point=a0*vector(e.v0)+b1*vector(e.v1)+b2*vector(e.v2);
            const Vec3 delta=point-position;
            const float distance2=m1::dot(delta,delta);
            if(distance2>0) {
                const Vec3 direction=delta/sqrtf(distance2),en=vector(e.normal);
                const float cos_light=m1::dot(en,-direction);
                const Vec3 local=frame.to_local(direction);
                if(cos_light>0&&local.z>0&&m1::dot(normal,direction)>0) {
                    const float2 light_uv=a0*e.uv0+b1*e.uv1+b2*e.uv2;
                    const float2 alpha_uv=a0*e.opacityUV0+b1*e.opacityUV1+b2*e.opacityUV2;
                    float3 rgb=e.constantEmission;
                    if(e.emissiveTexture.texture)
                        rgb*=make_float3(tex2D<float4>(e.emissiveTexture.texture,light_uv.x,1-light_uv.y));
                    const float le=emission(rgb,p.waveLength,p)*
                        opacity(e.opacityTexture,alpha_uv,e.opacityFactor,e.alphaCutoff,p);
                    const auto density=m1::one_sided_area_light_pdf(distance2,e.area,cos_light,count);
                    const float pdf=density.pdf;
                    if(!m1::finite(le)||le<0||!density.valid||!(pdf>0)){
                        fail(p,FailureReason::AreaLightParameters,
                            {{le,pdf,distance2,e.area,cos_light,float(count),float(selected)}});return;
                    }
                    const Vec3 origin=offset(position,ng,direction);
                    const Vec3 target=offset(point,en,-direction);
                    const Vec3 shadow_delta=target-origin;
                    const float shadow_distance=m1::length(shadow_delta);
                    if(le>0&&shadow_distance>0&&visible(p,origin,shadow_delta/shadow_distance,shadow_distance)) {
                        const float transmittance=m3device::connection_transmittance(p,
                            origin,shadow_delta/shadow_distance,shadow_distance,
                            optixLaunchParams.rayleigh.areaNeeTransmittance!=0);
                        if(!p.m1Invalid)add_direct(p,bsdf,wo,local,le*transmittance,pdf);
                    }
                }
            }
        }
    }
    if(p.m1Invalid)return;
    p.m1FailureStage=FailureStage::Bsdf;
    const auto sampled=bsdf.sample(wo,p.random(),p.random(),p.random(),p);
    if(sampled.status==m1::BsdfSampleStatus::Invalid)return;
    if(sampled.status==m1::BsdfSampleStatus::Null){p.continueTrace=0;return;}
    const Vec3 direction=m1::normalize(frame.to_world(sampled.wi));
    const float geometric_cosine=m1::dot(normal,direction);
    // Same Ng support for BSDF sampling and NEE. Removed directions are null
    // events, not a renormalized sampling distribution. In radiance transport,
    // use the Ns cosine; the adjoint shading-normal correction is one.
    if((sampled.transmission&&geometric_cosine>=0)||(!sampled.transmission&&geometric_cosine<=0)) {
        p.continueTrace=0;return;
    }
    p.albedo*=sampled.weight;p.m1EtaScale*=sampled.eta_scale;
    if(!m1::finite(p.albedo)||p.albedo<0||!m1::finite(p.m1EtaScale)||!(p.m1EtaScale>0)){
        fail(p,FailureReason::Throughput,{{sampled.weight,sampled.eta_scale}});return;
    }
    if(p.albedo==0){p.continueTrace=0;return;}
    // MIS PDFs are conditional on reaching this vertex and exclude RR in BOTH
    // strategies. Surviving path throughput alone is divided by survival.
    if(p.bounce+1>=int(optixLaunchParams.m1.rrDepth)) {
        const float rr_beta=p.albedo*p.m1EtaScale;
        if(!m1::finite(rr_beta)){fail(p,FailureReason::Roulette,{{rr_beta}});return;}
        const float survival=fminf(0.95f,rr_beta);
        if(!(survival>0)||p.random()>=survival){p.continueTrace=0;return;}
        p.albedo/=survival;
    }
    p.pdf.bxdf=sampled.pdf;p.m1PreviousDelta=sampled.delta;p.m1PreviousPosition=vector(position);
    p.m3PreviousMedium=false;
    p.wi=vector(direction);p.position=vector(offset(position,ng,direction));p.continueTrace=1;
    if(!m1::finite(p.albedo)||!finite3(p.position)||!finite3(p.wi))
        fail(p,FailureReason::Continuation,{{p.position.x,p.position.y,p.position.z,p.wi.x,p.wi.y,p.wi.z}});
}

static __forceinline__ __device__ void scatter_medium(PRDSpectral& p,Vec3 position) {
    // One vertex consumes one scattering bounce, exactly as a surface BSDF.
    // Free-flight collision/no-collision probabilities already carry T and
    // sigma_s. Pure Rayleigh has sigma_s/sigma_t=1, so beta is unchanged here.
    p.position=vector(position);
    if(!m1::finite(position)) {
        fail(p,FailureReason::AtmosphereContinuation,{{position.x,position.y,position.z,
             p.wi.x,p.wi.y,p.wi.z}});return;
    }
    if(p.bounce>=optixLaunchParams.frame.maxBounce){p.continueTrace=0;return;}
    const Vec3 incoming=vector(p.wi);
    const auto& settings=optixLaunchParams.m1;
    if(settings.sunEnabled&&settings.directMode!=1) {
        p.m1FailureStage=FailureStage::AtmosphereSunNee;
        const Vec3 direction=sample_sun(p.random(),p.random());
        const float le=sun_radiance(p.waveLength),pdf=sun_pdf();
        const float phase=m3::rayleigh_phase_directions(incoming,direction);
        if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)||!m1::finite(phase)||!(phase>0)){
            fail(p,FailureReason::AtmosphereDirect,{{le,pdf,phase}});return;
        }
        if(le>0&&visible(p,position,direction,CUDART_INF_F)) {
            const float transmittance=m3device::connection_transmittance(p,position,direction,CUDART_INF_F);
            if(p.m1Invalid)return;
            const float weight=m3::nee_weight(settings.directMode,pdf,phase);
            p.contribution+=p.albedo*phase*le*transmittance*weight/pdf;
            if(!m1::finite(p.contribution)||p.contribution<0){
                fail(p,FailureReason::AtmosphereDirect,{{le,pdf,phase,transmittance,weight}});return;
            }
        }
    }
    // There is deliberately no area-light NEE at a medium vertex. Area lights
    // remain reachable by phase continuation; their emission then has weight 1.
    if(p.m1Invalid)return;
    p.m1FailureStage=FailureStage::Atmosphere;
    const float u_mu=p.random(),u_phi=p.random();
    const auto sampled=m3::sample_rayleigh(incoming,u_mu,u_phi);
    if(!sampled.valid||!m1::bsdf_unit_direction(sampled.direction)||
       !m1::finite(sampled.value)||sampled.value<0||!m1::finite(sampled.pdf)||!(sampled.pdf>0)){
        fail(p,FailureReason::AtmospherePhase,{{sampled.direction.x,sampled.direction.y,sampled.direction.z,
             sampled.value,sampled.pdf,u_mu,u_phi}});return;
    }
    p.albedo*=sampled.value/sampled.pdf;
    if(!m1::finite(p.albedo)||p.albedo<0){
        fail(p,FailureReason::Throughput,{{sampled.value/sampled.pdf,1}});return;
    }
    if(p.albedo==0){p.continueTrace=0;return;}
    if(p.bounce+1>=int(settings.rrDepth)) {
        const float rr_beta=p.albedo*p.m1EtaScale;
        if(!m1::finite(rr_beta)){fail(p,FailureReason::Roulette,{{rr_beta}});return;}
        const float survival=fminf(0.95f,rr_beta);
        if(!(survival>0)||p.random()>=survival){p.continueTrace=0;return;}
        p.albedo/=survival;
    }
    p.pdf.bxdf=sampled.pdf;p.m1PreviousDelta=false;p.m3PreviousMedium=true;
    p.m1PreviousPosition=vector(position);p.position=vector(position);
    p.wi=vector(sampled.direction);p.continueTrace=1;
    if(!m1::finite(p.albedo)||!finite3(p.position)||!finite3(p.wi))
        fail(p,FailureReason::AtmosphereContinuation,{{position.x,position.y,position.z,
             sampled.direction.x,sampled.direction.y,sampled.direction.z,sampled.value,sampled.pdf}});
}

static __forceinline__ __device__ m4rain::ScatterResponse rain_query(
    PRDSpectral& p,const m4rain::ScatterRequest& request){
    m4rain::ScatterResponse out{};
    optixDirectCall<void,const m4rain::ScatterRequest*,m4rain::ScatterResponse*>(
        M4_NF_CALLABLE_ID,&request,&out);
    if(out.status==m4nf::Status::success&&
       (!m4nf::unit(out.direction)||!m1::finite(out.value)||!(out.value>0)||
        !m1::finite(out.pdf)||!(out.pdf>0)||!m1::finite(out.weight)||!(out.weight>0)))
        out.status=m4nf::Status::nonfinite;
    if(out.status!=m4nf::Status::success)
        fail(p,FailureReason::RainScatter,{{float(out.status),float(request.op),p.waveLength,
            out.proposalCosine,out.incidentCosine,out.value,out.pdf,out.weight,
            out.logPhysicalPhase,out.logProposal,out.cscaIncident,out.cextCamera,request.u0,request.u1}});
    return out;
}

static __forceinline__ __device__ void scatter_rain(PRDSpectral& p,Vec3 position){
    p.position=vector(position);p.m1FailureStage=FailureStage::Rain;
    if(!m1::finite(position)||!finite3(p.wi)){
        fail(p,FailureReason::AtmosphereContinuation,{{position.x,position.y,position.z,p.wi.x,p.wi.y,p.wi.z}});return;
    }
    if(p.bounce>=optixLaunchParams.frame.maxBounce){p.continueTrace=0;return;}
    const auto& settings=optixLaunchParams.m1;
    m4rain::ScatterRequest request{};request.wavelength=p.waveLength;
    request.cameraDirection={p.wi.x,p.wi.y,p.wi.z};
    if(settings.sunEnabled&&settings.directMode!=1){
        p.m1FailureStage=FailureStage::RainSunNee;
        const Vec3 direction=sample_sun(p.random(),p.random());
        const float le=sun_radiance(p.waveLength),pdf=sun_pdf();
        if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)){
            fail(p,FailureReason::SunParameters,{{le,pdf,settings.sunAngularRadiusRad,settings.sunScale}});return;
        }
        if(le>0&&visible(p,position,direction,CUDART_INF_F)){
            const float transmittance=m3device::connection_transmittance(p,position,direction,CUDART_INF_F);
            if(p.m1Invalid)return;
            if(transmittance>0){
                request.op=m4rain::ScatterOp::Evaluate;
                request.nextDirection={direction.x,direction.y,direction.z};
                const auto q=rain_query(p,request);if(p.m1Invalid)return;
                const float mis=m3::nee_weight(settings.directMode,pdf,q.pdf);
                p.contribution+=p.albedo*q.value*le*transmittance*mis/pdf;
                if(!m1::finite(p.contribution)||p.contribution<0){
                    fail(p,FailureReason::RainDirect,{{le,pdf,q.value,q.pdf,transmittance,mis}});return;
                }
            }
        }
    }
    // The component label was chosen by competing free flights. Both NEE and
    // continuation above/below estimate ONLY the rain summand conditioned on
    // that label. Rayleigh events use the existing analytic branch instead.
    if(p.m1Invalid)return;
    p.m1FailureStage=FailureStage::Rain;request.op=m4rain::ScatterOp::Sample;
    m3device::OpenRandom random{p.random};request.u0=random();request.u1=random();
    const auto q=rain_query(p,request);if(p.m1Invalid)return;
    p.albedo*=q.weight;
    if(!m1::finite(p.albedo)||p.albedo<0){fail(p,FailureReason::Throughput,{{q.weight,1}});return;}
    if(p.albedo==0){p.continueTrace=0;return;}
    if(p.bounce+1>=int(settings.rrDepth)){
        const float rr_beta=p.albedo*p.m1EtaScale;
        if(!m1::finite(rr_beta)){fail(p,FailureReason::Roulette,{{rr_beta}});return;}
        const float survival=fminf(0.95f,rr_beta);
        if(!(survival>0)||p.random()>=survival){p.continueTrace=0;return;}
        p.albedo/=survival;
    }
    p.pdf.bxdf=q.pdf;p.m1PreviousDelta=false;p.m3PreviousMedium=true;
    p.m1PreviousPosition=vector(position);p.position=vector(position);
    p.wi=make_float3(q.direction.x,q.direction.y,q.direction.z);p.continueTrace=1;
    if(!m1::finite(p.albedo)||!finite3(p.wi))
        fail(p,FailureReason::AtmosphereContinuation,{{position.x,position.y,position.z,
            q.direction.x,q.direction.y,q.direction.z,q.value,q.pdf}});
}

static __forceinline__ __device__ void scatter_planet(PRDSpectral& p,
    const m3::tracking::RaySegment& ray,const m3::tracking::ShellClip& segment) {
    p.m1FailureStage=FailureStage::Planet;p.m1FailureContext={};
    const auto& a=optixLaunchParams.rayleigh;
    const auto surface=m3::tracking::planet_hit(a,ray,segment);
    const Vec3 position=surface.position;
    const Vec3 ng=surface.normal;
    if(surface.status!=m3::tracking::Status::Ok||!m1::finite(position)||!m1::bsdf_unit_direction(ng)){
        const auto root_position=m3::tracking::value(m3::tracking::point_at(ray,segment.ground_parameter));
        fail(p,FailureReason::PlanetIntersection,{{segment.ground_t,position.x,position.y,position.z,
             ng.x,ng.y,ng.z,float(surface.status),surface.root_height_m,surface.root_height_bound_m,
             segment.ground_parameter.hi,segment.ground_parameter.lo,surface.radial_correction_m,
             root_position.x,root_position.y,root_position.z}});return;
    }
    p.position=vector(position);p.instanceID=m1::no_index;
    if(p.bounce==0) {
        p.m1PrimaryNormal=vector(ng);p.m1PrimaryAlbedo=make_float3(a.groundAlbedo);
    }
    if(p.bounce>=optixLaunchParams.frame.maxBounce){p.continueTrace=0;return;}
    // Spectrally neutral, opaque Lambertian ground, explicitly parameterized.
    // It uses the surface NEE policy and can exchange light with the atmosphere.
    SurfaceBsdf bsdf;bsdf.type=MATERIAL_TYPE_DIFFUSE;
    bsdf.simple={m1::MaterialKind::Diffuse,a.groundAlbedo,1,1};
    const Vec3 incoming=m1::normalize(vector(p.wi));
    scatter_surface(p,position,ng,ng,ng,incoming,bsdf);
}

// A single existing OptiX traversal first resolves alpha coverage and obtains
// the nearest accepted triangle. We then sample the medium only up to that
// endpoint (or the opaque planet if nearer). No stochastic alpha is re-tested.
// Returning true means this segment was consumed by a scatter/planet/failure.
static __forceinline__ __device__ bool advance_medium(PRDSpectral& p,float t_max) {
    if(!optixLaunchParams.rayleigh.enabled)return false;
    p.m1FailureStage=FailureStage::Atmosphere;p.m1FailureContext={};
    const auto r=m3device::ray(vector(p.position),vector(p.wi),t_max);
    float beta=0.0f;
    if(!m3device::coefficient(p,beta))return true;
    m3device::OpenRandom rng{p.random};
    const auto& rain=optixLaunchParams.rain;
    const auto mixed=m4rain::free_flight(optixLaunchParams.rayleigh,rain.parameters,
                                        rain.model,r,beta,p.waveLength,rng);
    const auto& sampled=mixed.flight;
    if(sampled.status!=m3::tracking::Status::Ok){
        fail(p,FailureReason::RainTracking,{{float(sampled.status),float(mixed.nfStatus),
             sampled.ray_t,float(sampled.candidates),float(mixed.rain.candidates),
             float(mixed.rain.nulls),p.waveLength}});
        return true;
    }
    if(sampled.scattered) {
        const auto exact=m3::tracking::point_at(r,sampled.parameter);
        const auto represented=m3::tracking::represent_medium_point(optixLaunchParams.rayleigh,exact);
        if(represented.status!=m3::tracking::Status::Ok) {
            fail(p,FailureReason::AtmospherePosition,{{float(represented.status),float(sampled.candidates),
                 sampled.ray_t,m3::tracking::value(exact.x),m3::tracking::value(exact.y),m3::tracking::value(exact.z),
                 represented.position.x,represented.position.y,represented.position.z}});
            return true;
        }
        if(mixed.component==m4rain::Component::Rain)scatter_rain(p,represented.position);
        else scatter_medium(p,represented.position);
        return true;
    }
    if(sampled.endpoint==m3::tracking::Endpoint::Ground) {
        scatter_planet(p,r,sampled.clip);
        return true;
    }
    // The probability of reaching this endpoint is already exp(-tau).
    // Adding another T here (or in miss/emitter evaluation) would give T^2.
    return false;
}

static __forceinline__ __device__ void hit(PRDSpectral& p) {
    if(p.m1Invalid){p.continueTrace=0;return;}
    if(advance_medium(p,optixGetRayTmax()))return;
    const auto& sbt=*reinterpret_cast<const HitgroupSBTData*>(optixGetSbtDataPointer());
    m1diagnostics::set_hit_context(p,FailureStage::Surface,sbt);
    if(sbt.geomType!=GeomType::Triangle||sbt.tri.meshIndex>=unsigned(optixLaunchParams.numMeshes)||
       sbt.tri.materialIndex>=unsigned(optixLaunchParams.numMaterials)){
        fail(p,FailureReason::HitSbt,{{float(sbt.geomType),float(optixLaunchParams.numMeshes),
                                     float(optixLaunchParams.numMaterials)}});return;
    }
    const auto& mesh=optixLaunchParams.meshes[sbt.tri.meshIndex];
    const auto& material=optixLaunchParams.materials[sbt.tri.materialIndex];
    const unsigned primitive=optixGetPrimitiveIndex(),instance=optixGetInstanceId();
    const uint3 ind=mesh.index[primitive];
    const Vec3 v0=vector(optixTransformPointFromObjectToWorldSpace(mesh.vertex[ind.x]));
    const Vec3 v1=vector(optixTransformPointFromObjectToWorldSpace(mesh.vertex[ind.y]));
    const Vec3 v2=vector(optixTransformPointFromObjectToWorldSpace(mesh.vertex[ind.z]));
    const Vec3 cross=m1::cross(v1-v0,v2-v0);
    const float twice_area=m1::length(cross);
    if(!m1::finite(twice_area)||!(twice_area>0)){
        fail(p,FailureReason::HitGeometry,{{twice_area,cross.x,cross.y,cross.z}});return;
    }
    const Vec3 ng=cross/twice_area;
    const auto bary=optixGetTriangleBarycentrics();
    const float b0=1.0f-bary.x-bary.y;
    const Vec3 position=b0*v0+bary.x*v1+bary.y*v2;
    const Vec3 ray_direction=m1::normalize(vector(optixGetWorldRayDirection()));
    const bool front=m1::dot(ng,-ray_direction)>0;
    float2 uv=make_float2(0),emissive_uv=make_float2(0);
    if(mesh.texcoord)uv=b0*mesh.texcoord[ind.x]+bary.x*mesh.texcoord[ind.y]+bary.y*mesh.texcoord[ind.z];
    if(mesh.emissiveTexcoord)
        emissive_uv=b0*mesh.emissiveTexcoord[ind.x]+bary.x*mesh.emissiveTexcoord[ind.y]+
                    bary.y*mesh.emissiveTexcoord[ind.z];
    if((material.emissiveTexture.texture&&!mesh.emissiveTexcoord)||
       ((material.diffuseTexture.texture||material.rmTexture.texture)&&!mesh.texcoord)) {
        fail(p,FailureReason::MissingSurfaceUV,{{float(material.diffuseTexture.texture!=0),
             float(material.rmTexture.texture!=0),float(material.emissiveTexture.texture!=0)}});return;
    }
    const Vec3 ns=shading_normal(mesh,material,ind,b0,bary.x,bary.y,ng,v1-v0,v2-v0,p);
    if(p.m1Invalid)return;
    if(p.bounce==0)p.m1PrimaryNormal=vector(ns);

    if(material.hasEmission || material.materialType==MATERIAL_TYPE_LIGHT) {
        if(front) {
            float3 rgb=material.emissive;
            if(material.emissiveTexture.texture)
                rgb*=make_float3(tex2D<float4>(material.emissiveTexture.texture,emissive_uv.x,1-emissive_uv.y));
            const float le=emission(rgb,p.waveLength,p);
            const int emitter=find_emitter(sbt.tri.meshIndex,instance,primitive);
            if(emitter<0){fail(p,FailureReason::EmitterMissing,{{float(optixLaunchParams.light.numLights)}});return;}
            const auto& e=optixLaunchParams.light.triangleLightData[emitter];
            const Vec3 connection=position-vector(p.m1PreviousPosition);
            const float distance2=m1::dot(connection,connection);
            float pdf_light=0;
            if(m3::area_emission_needs_pdf(p.m3PreviousMedium,p.bounce,
                    p.m1PreviousDelta,optixLaunchParams.m1.directMode)) {
                const float cosine=m1::dot(ng,-m1::normalize(connection));
                if(!m1::finite(cosine)||!m1::finite(distance2)||!(distance2>0)||
                   !m1::finite(e.area)||!(e.area>0)||!(optixLaunchParams.light.numLights>0)){
                    fail(p,FailureReason::EmitterGeometry,{{cosine,m1::dot(ng,-ray_direction),
                        e.area,distance2,float(optixLaunchParams.light.numLights),
                        connection.x,connection.y,connection.z,ray_direction.x,ray_direction.y,ray_direction.z,
                        p.m1PreviousPosition.x,p.m1PreviousPosition.y,p.m1PreviousPosition.z}});return;
                }
                // The traced ray starts at an offset point, while area NEE is
                // defined between unoffset vertices. Its one-sided support can
                // therefore be empty even for a front-facing traced hit. Zero
                // competing density is valid and gives the BSDF strategy weight
                // one; abs(cosine) or a cosine floor would invent NEE support.
                const auto competing=m1::one_sided_area_light_pdf(
                    distance2,e.area,cosine,optixLaunchParams.light.numLights);
                pdf_light=competing.pdf;
                if(!competing.valid){
                    fail(p,FailureReason::EmitterPdf,{{pdf_light,cosine,e.area,distance2,
                         float(optixLaunchParams.light.numLights)}});return;
                }
            }
            const float emitter_weight=m3::area_emission_weight(p.m3PreviousMedium,p.bounce,
                p.m1PreviousDelta,optixLaunchParams.m1.directMode,p.pdf.bxdf,pdf_light);
            p.contribution+=p.albedo*le*emitter_weight;
            if(!m1::finite(p.contribution)||p.contribution<0)
                fail(p,FailureReason::EmitterContribution,{{le,pdf_light,emitter_weight}});
        }
        if(p.m1Invalid)return;
        // ORCA windows/signs may both emit and scatter. Continue through their
        // BSDF; the existing M1 pure lamps remain terminal endpoints.
        if(material.materialType==MATERIAL_TYPE_LIGHT){p.continueTrace=0;return;}
    }
    // Single-sided ORCA materials shade the front only. Their back faces still
    // occlude rays; they are not removed from the geometry or made transparent.
    if(!front&&!material.twoSided){p.continueTrace=0;return;}
    // maxBounce counts scattering vertices. A final endpoint may still emit.
    if(p.bounce>=optixLaunchParams.frame.maxBounce){p.continueTrace=0;return;}
    SurfaceBsdf bsdf;bsdf.type=material.materialType;
    if(material.materialType==MATERIAL_TYPE_GLASS) {
        const float ior=1.444829f+3506.29f/(p.waveLength*p.waveLength);
        bsdf.simple={m1::MaterialKind::Dielectric,0,1,front?1.0f/ior:ior};
        if(p.bounce==0)p.m1PrimaryAlbedo=make_float3(1);
    } else {
        float3 rgb=material.color;
        if(material.diffuseTexture.texture)
            rgb=make_float3(tex2D<float4>(material.diffuseTexture.texture,uv.x,1-uv.y));
        const float color=reflectance(rgb,p.waveLength,p);
        if(p.bounce==0)p.m1PrimaryAlbedo=rgb;
        if(material.materialType==MATERIAL_TYPE_DIFFUSE)
            bsdf.simple={m1::MaterialKind::Diffuse,color,1,1};
        else if(material.materialType==MATERIAL_TYPE_PRINCIPLED_BRDF) {
            float roughness=material.roughness,metallic=material.metallic;
            if(material.rmTexture.texture) {
                const auto rm=tex2D<float4>(material.rmTexture.texture,uv.x,1-uv.y);
                roughness=rm.y;metallic=rm.z;
            }
            bsdf.metallic={color,metallic,roughness};
            if(!m1::valid_metallic_parameters(bsdf.metallic))
                fail(p,FailureReason::MetallicParameters,{{color,metallic,roughness,uv.x,uv.y}});
        } else fail(p,FailureReason::UnsupportedMaterial,{{float(material.materialType)}});
    }
    if(p.m1Invalid)return;
    const Vec3 normal=front?ng:-ng;
    p.instanceID=instance;
    scatter_surface(p,position,ng,normal,ns,ray_direction,bsdf);
}

static __forceinline__ __device__ void miss(PRDSpectral& p) {
    if(p.m1Invalid){p.continueTrace=0;return;}
    if(advance_medium(p,CUDART_INF_F))return;
    p.m1FailureStage=FailureStage::Miss;
    p.m1FailureContext={};
    float le=0,pdf=0;
    if(!p.m1Invalid&&optixLaunchParams.m1.sunEnabled&&inside_sun(vector(p.wi))) {
        le=sun_radiance(p.waveLength);pdf=sun_pdf();
        if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)){
            fail(p,FailureReason::SunParameters,{{le,pdf,optixLaunchParams.m1.sunAngularRadiusRad,
                 optixLaunchParams.m1.sunScale}});return;
        }
        p.contribution+=p.albedo*le*light_hit_weight(p,pdf);
    }
    p.continueTrace=0;
    if(!m1::finite(p.contribution)||p.contribution<0)
        fail(p,FailureReason::MissContribution,{{le,pdf}});
}

static __forceinline__ __device__ void render_frame() {
    const unsigned ix=optixGetLaunchIndex().x,iy=optixGetLaunchIndex().y;
    const unsigned index=ix+iy*optixLaunchParams.frame.size.x;
    PRDSpectral p;
    p.random.init(index^m1::mix_bits(optixLaunchParams.m1.seed),optixLaunchParams.frame.frameID);
    unsigned u0,u1;packPointer(&p,u0,u1);
    float3 mean=make_float3(0),normal_mean=make_float3(0),albedo_mean=make_float3(0);
    for(int sample=0;sample<optixLaunchParams.frame.numPixelSamples;++sample) {
        p.contribution=0;p.albedo=1;p.continueTrace=1;p.bounce=0;
        p.pdf.bxdf=0;p.m1PreviousDelta=true;p.m1Invalid=false;p.m1EtaScale=1;
        p.m3PreviousMedium=false;
        p.m1SampleIndex=unsigned(sample);p.m1FailureStage=FailureStage::Camera;p.m1FailureContext={};
        p.m1PrimaryNormal=make_float3(0);p.m1PrimaryAlbedo=make_float3(0);
        const float minimum=optixLaunchParams.spectral.wavelengthMin;
        const float range=optixLaunchParams.spectral.wavelengthMax-minimum;
        p.waveLength=minimum+range*p.random();p.waveLengthNormalized=(p.waveLength-minimum)/range;
        p.pdf.spectral=1.0f/range;
        const float2 screen=make_float2(
            (float(optixLaunchParams.frame.size.x-int(ix)-1)+p.random())/optixLaunchParams.frame.size.x,
            (float(optixLaunchParams.frame.size.y-int(iy)-1)+p.random())/optixLaunchParams.frame.size.y);
        const LensRay ray=optixDirectCall<LensRay,const float2,const float,const float>(
            optixLaunchParams.camera.cameraMode,screen,p.random(),p.random());
        p.position=ray.org;p.wi=vector(m1::normalize(vector(ray.dir)));p.m1PreviousPosition=ray.org;
        if(!finite3(p.position)||!finite3(p.wi)||!m1::finite(ray.weight)||ray.weight<0)
            fail(p,FailureReason::CameraRay,{{ray.org.x,ray.org.y,ray.org.z,
                 ray.dir.x,ray.dir.y,ray.dir.z,ray.weight}});
        while(p.continueTrace&&!p.m1Invalid) {
            if(optixLaunchParams.traversable)
                optixTrace(optixLaunchParams.traversable,p.position,p.wi,0.0f,CUDART_INF_F,0.0f,
                           OptixVisibilityMask(MASK_SURFACE),OPTIX_RAY_FLAG_NONE,
                           RADIANCE_RAY_TYPE,RAY_TYPE_COUNT,RADIANCE_RAY_TYPE,u0,u1);
            else
                miss(p);
            if(p.continueTrace)++p.bounce;
        }
        // Exactly one wavelength-PDF division per camera sample. No HWSS,
        // empirically chosen 0.1 multiplier, radiance clamp or NaN-to-black pass.
        const float3 xyz=cie(p.waveLength)*(p.contribution*ray.weight/p.pdf.spectral);
        const float3 rgb=make_float3(3.24096994f*xyz.x-1.53738318f*xyz.y-0.49861076f*xyz.z,
                                    -0.96924364f*xyz.x+1.87596750f*xyz.y+0.04155506f*xyz.z,
                                    0.05563008f*xyz.x-0.20397696f*xyz.y+1.05697151f*xyz.z);
        p.m1FailureStage=FailureStage::Film;
        if(!finite3(rgb))fail(p,FailureReason::FilmConversion,
            {{xyz.x,xyz.y,xyz.z,rgb.x,rgb.y,rgb.z,ray.weight,p.pdf.spectral}});
        if(p.m1Invalid) {
            if(optixLaunchParams.m1.invalidSamples)atomicAdd(optixLaunchParams.m1.invalidSamples,1u);
            // The host rejects the entire frame when this diagnostic is nonzero.
            continue;
        }
        mean+=(rgb-mean)/float(sample+1);
        normal_mean+=(p.m1PrimaryNormal-normal_mean)/float(sample+1);
        albedo_mean+=(p.m1PrimaryAlbedo-albedo_mean)/float(sample+1);
        if(!finite3(mean)||!finite3(normal_mean)||!finite3(albedo_mean)) {
            fail(p,FailureReason::SampleMean,{{mean.x,mean.y,mean.z,normal_mean.x,normal_mean.y,normal_mean.z,
                 albedo_mean.x,albedo_mean.y,albedo_mean.z}});
            if(optixLaunchParams.m1.invalidSamples)atomicAdd(optixLaunchParams.m1.invalidSamples,1u);
        }
    }
    float4 rgba=make_float4(mean,1);
    if(optixLaunchParams.frame.frameID>0) {
        const float4 previous=optixLaunchParams.frame.colorBuffer[index];
        rgba=previous+(rgba-previous)/float(optixLaunchParams.frame.frameID+1);
    }
    optixLaunchParams.frame.colorBuffer[index]=rgba;
    float4 normal_rgba=make_float4(normal_mean,1),albedo_rgba=make_float4(albedo_mean,1);
    if(optixLaunchParams.frame.frameID>0) {
        if(optixLaunchParams.frame.normalBuffer) {
            const auto previous=optixLaunchParams.frame.normalBuffer[index];
            normal_rgba=previous+(normal_rgba-previous)/float(optixLaunchParams.frame.frameID+1);
        }
        if(optixLaunchParams.frame.albedoBuffer) {
            const auto previous=optixLaunchParams.frame.albedoBuffer[index];
            albedo_rgba=previous+(albedo_rgba-previous)/float(optixLaunchParams.frame.frameID+1);
        }
    }
    if(optixLaunchParams.frame.normalBuffer)optixLaunchParams.frame.normalBuffer[index]=normal_rgba;
    if(optixLaunchParams.frame.albedoBuffer)optixLaunchParams.frame.albedoBuffer[index]=albedo_rgba;
    if(!finite3(make_float3(rgba))||!finite3(make_float3(normal_rgba))||!finite3(make_float3(albedo_rgba))) {
        p.m1FailureStage=FailureStage::Film;
        fail(p,FailureReason::TemporalMean,{{rgba.x,rgba.y,rgba.z,normal_rgba.x,normal_rgba.y,normal_rgba.z,
             albedo_rgba.x,albedo_rgba.y,albedo_rgba.z}});
        if(optixLaunchParams.m1.invalidSamples)atomicAdd(optixLaunchParams.m1.invalidSamples,1u);
    }
}

} // namespace m1surface
