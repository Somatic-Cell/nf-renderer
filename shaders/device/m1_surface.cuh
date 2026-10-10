#pragma once

// M1 is the existing NeuralPFRenderer surface pipeline in vacuum. It uses the
// existing launch symbol, pointer payload, mesh/instance SBT and RGB framebuffer.
#include "../config.cuh"
#include "../params/per_ray_data.cuh"
#include "shader_common.cuh"
#include "../../include/m1/bsdf.hpp"
#include "../../include/m1/rgb2spec_device.hpp"
#include <math_constants.h>

namespace m1surface {
using m1::Vec3;

static __forceinline__ __device__ Vec3 vector(float3 v){return {v.x,v.y,v.z};}
static __forceinline__ __device__ float3 vector(Vec3 v){return make_float3(v.x,v.y,v.z);}
static __forceinline__ __device__ bool finite3(float3 v){return m1::finite(vector(v));}
static __forceinline__ __device__ void fail(PRDSpectral& p){p.m1Invalid=true;p.continueTrace=0;}

static __forceinline__ __device__ float reflectance(float3 rgb,float wavelength,PRDSpectral& p) {
    const auto& m=optixLaunchParams.m1;
    const m1::Rgb2SpecTableView table{m.rgb2specScale,m.rgb2specData,m.rgb2specResolution};
    float value=0;
    if(!m1::rgb2spec_reflectance_device(table,rgb.x,rgb.y,rgb.z,wavelength,value))fail(p);
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
    if(!finite3(rgb)||rgb.x<0||rgb.y<0||rgb.z<0){fail(p);return 0;}
    const float amplitude=2.0f*fmaxf(rgb.x,fmaxf(rgb.y,rgb.z));
    if(amplitude==0)return 0;
    if(!m1::finite(amplitude)){fail(p);return 0;}
    const float value=amplitude*reflectance(rgb/amplitude,wavelength,p)*
        d65(wavelength)*optixLaunchParams.light.lightIntensityFactor;
    if(!m1::finite(value)||value<0)fail(p);
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
    return m1::normalize(Vec3{sinf(z)*cosf(a),cosf(z),sinf(z)*sinf(a)});
}
static __forceinline__ __device__ float sun_pdf() {
    const float s=sinf(0.5f*optixLaunchParams.m1.sunAngularRadiusRad);
    return 1.0f/(4.0f*m1::pi*s*s);
}
static __forceinline__ __device__ float sun_radiance(float wavelength) {
    const float s=sinf(optixLaunchParams.m1.sunAngularRadiusRad);
    // E = integral_disk L*cos(theta) dOmega = pi*L*sin^2(alpha).
    return optixLaunchParams.m1.sunScale*solar_irradiance(wavelength)/(m1::pi*s*s);
}
static __forceinline__ __device__ bool inside_sun(Vec3 direction) {
    const Vec3 d=direction-sun_direction();
    const double half=sin(0.5*double(optixLaunchParams.m1.sunAngularRadiusRad));
    // Chord distance avoids subtracting two nearly equal cosines at 0.25 deg.
    // There is no PDF floor or angular support expansion by a tolerance.
    return double(d.x)*d.x+double(d.y)*d.y+double(d.z)*d.z<=4.0*half*half;
}
static __forceinline__ __device__ Vec3 sample_sun(float u,float v) {
    const float s=sinf(0.5f*optixLaunchParams.m1.sunAngularRadiusRad);
    const float q=u*(2.0f*s*s),r=sqrtf(q*(2.0f-q)),phi=2.0f*m1::pi*v;
    const auto frame=m1::make_frame(sun_direction());
    return m1::normalize(frame.to_world({r*cosf(phi),r*sinf(phi),1.0f-q}));
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
        fail(p);return false;
    }
    ShadowPRD shadow;shadow.random=p.random;shadow.visible=false;shadow.m1Invalid=false;
    unsigned u0,u1;packPointer(&shadow,u0,u1);
    optixTrace(optixLaunchParams.traversable,vector(origin),vector(direction),0.0f,distance,0.0f,
               OptixVisibilityMask(MASK_SURFACE),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT|OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               SHADOW_RAY_TYPE,RAY_TYPE_COUNT,SHADOW_RAY_TYPE,u0,u1);
    p.random=shadow.random;
    if(shadow.m1Invalid)fail(p);
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
    __forceinline__ __device__ m1::BsdfSample sample(Vec3 wo,float u,float v,float w) const {
        return type==MATERIAL_TYPE_PRINCIPLED_BRDF ? m1::sample_metallic(metallic,wo,u,v,w) :
                                                   m1::sample_bsdf(simple,wo,u,v,w);
    }
};

static __forceinline__ __device__ float light_hit_weight(const PRDSpectral& p,float pdf_light) {
    if(p.bounce==0||p.m1PreviousDelta||optixLaunchParams.m1.directMode==1)return 1.0f;
    if(optixLaunchParams.m1.directMode==2)return 0.0f;
    return m1::power_heuristic(p.pdf.bxdf,pdf_light);
}
static __forceinline__ __device__ void add_direct(PRDSpectral& p,const SurfaceBsdf& bsdf,
                                                 Vec3 wo,Vec3 wi,float radiance,float pdf_light) {
    if(wi.z<=0||radiance==0)return;
    const auto v=bsdf.evaluate(wo,wi);
    if(!v.valid||!m1::finite(radiance)||radiance<0||!m1::finite(pdf_light)||!(pdf_light>0)){
        fail(p);return;
    }
    const float weight=optixLaunchParams.m1.directMode==0 ? m1::power_heuristic(pdf_light,v.pdf) : 1.0f;
    p.contribution+=p.albedo*v.f*wi.z*radiance*weight/pdf_light;
    if(!m1::finite(p.contribution)||p.contribution<0)fail(p);
}

static __forceinline__ __device__ int find_emitter(unsigned mesh,unsigned instance,unsigned primitive) {
    for(int i=0;i<optixLaunchParams.light.numLights;++i) {
        const auto definition=optixLaunchParams.light.lightDefinition[i];
        if(definition.lightType!=LIGHT_TYPE_TRIANGLE)continue;
        const auto& e=optixLaunchParams.light.triangleLightData[definition.lightIndexInType];
        if(e.meshIndex==mesh&&e.instanceID==instance&&e.primitiveIndex==primitive)
            return int(definition.lightIndexInType);
    }
    return -1;
}

static __forceinline__ __device__ float opacity(TextureSlot texture,float2 uv,PRDSpectral& p) {
    const float a=texture.texture ? tex2D<float4>(texture.texture,uv.x,1.0f-uv.y).w : 1.0f;
    if(!m1::finite(a)||a<0||a>1){fail(p);return 0;}
    return a;
}

static __forceinline__ __device__ Vec3 shading_normal(const TriangleMeshGeomData& mesh,
    const MaterialData& material,uint3 ind,float b0,float b1,float b2,Vec3 ng,PRDSpectral& p) {
    Vec3 ns=ng;
    Vec3 object_normal{};
    if(mesh.hasNormal) {
        if(!mesh.normal){fail(p);return ng;}
        object_normal=vector(b0*mesh.normal[ind.x]+b1*mesh.normal[ind.y]+b2*mesh.normal[ind.z]);
        ns=m1::normalize(vector(optixTransformNormalFromObjectToWorldSpace(vector(object_normal))));
        if(!m1::bsdf_unit_direction(ns)){fail(p);return ng;}
    }
    if(m1::dot(ns,ng)<0)ns=-ns;
    if(material.normalTexture.texture) {
        if(!mesh.hasTangent||!mesh.tangent||!mesh.hasNormal||!mesh.normalTexcoord){fail(p);return ng;}
        const float2 uv=b0*mesh.normalTexcoord[ind.x]+b1*mesh.normalTexcoord[ind.y]+b2*mesh.normalTexcoord[ind.z];
        const float4 raw=tex2D<float4>(material.normalTexture.texture,uv.x,1.0f-uv.y);
        if(!isfinite(raw.x)||!isfinite(raw.y)||raw.x<0||raw.x>1||raw.y<0||raw.y>1){fail(p);return ng;}
        const float4 tangent=b0*mesh.tangent[ind.x]+b1*mesh.tangent[ind.y]+b2*mesh.tangent[ind.z];
        if(!isfinite(tangent.w)||tangent.w==0){fail(p);return ng;}
        const Vec3 object_tangent=vector(make_float3(tangent));
        Vec3 t=vector(optixTransformVectorFromObjectToWorldSpace(vector(object_tangent)));
        t=m1::normalize(t-ns*m1::dot(ns,t));
        const Vec3 object_b=m1::cross(object_normal,object_tangent)*(tangent.w>0?1.0f:-1.0f);
        const Vec3 world_b=vector(optixTransformVectorFromObjectToWorldSpace(vector(object_b)));
        const Vec3 cross_b=m1::cross(ns,t);
        const Vec3 b=m1::dot(cross_b,world_b)>=0?cross_b:-cross_b;
        if(!m1::bsdf_unit_direction(t)||!m1::bsdf_unit_direction(b)){fail(p);return ng;}
        // Existing two-channel normal-map convention: signed XY, flipped green,
        // positive Z reconstructed from XY. Normalize after texture filtering.
        const float x=2.0f*raw.x-1.0f,y=1.0f-2.0f*raw.y;
        const float z=sqrtf(fmaxf(0.0f,1.0f-x*x-y*y));
        ns=m1::normalize(x*t+y*b+z*ns);
        if(!m1::bsdf_unit_direction(ns)){fail(p);return ng;}
        if(m1::dot(ns,ng)<0)ns=-ns;
    }
    return ns;
}

static __forceinline__ __device__ void hit(PRDSpectral& p) {
    if(p.m1Invalid){p.continueTrace=0;return;}
    const auto& sbt=*reinterpret_cast<const HitgroupSBTData*>(optixGetSbtDataPointer());
    if(sbt.geomType!=GeomType::Triangle||sbt.tri.meshIndex>=unsigned(optixLaunchParams.numMeshes)||
       sbt.tri.materialIndex>=unsigned(optixLaunchParams.numMaterials)){fail(p);return;}
    const auto& mesh=optixLaunchParams.meshes[sbt.tri.meshIndex];
    const auto& material=optixLaunchParams.materials[sbt.tri.materialIndex];
    const unsigned primitive=optixGetPrimitiveIndex(),instance=optixGetInstanceId();
    const uint3 ind=mesh.index[primitive];
    const Vec3 v0=vector(optixTransformPointFromObjectToWorldSpace(mesh.vertex[ind.x]));
    const Vec3 v1=vector(optixTransformPointFromObjectToWorldSpace(mesh.vertex[ind.y]));
    const Vec3 v2=vector(optixTransformPointFromObjectToWorldSpace(mesh.vertex[ind.z]));
    const Vec3 cross=m1::cross(v1-v0,v2-v0);
    const float twice_area=m1::length(cross);
    if(!m1::finite(twice_area)||!(twice_area>0)){fail(p);return;}
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
        fail(p);return;
    }
    const Vec3 ns=shading_normal(mesh,material,ind,b0,bary.x,bary.y,ng,p);
    if(p.m1Invalid)return;
    if(p.bounce==0)p.m1PrimaryNormal=vector(ns);

    if(material.materialType==MATERIAL_TYPE_LIGHT) {
        if(front) {
            float3 rgb=material.emissive;
            if(material.emissiveTexture.texture)
                rgb*=make_float3(tex2D<float4>(material.emissiveTexture.texture,emissive_uv.x,1-emissive_uv.y));
            const float le=emission(rgb,p.waveLength,p);
            const int emitter=find_emitter(sbt.tri.meshIndex,instance,primitive);
            if(emitter<0){fail(p);return;}
            const auto& e=optixLaunchParams.light.triangleLightData[emitter];
            const Vec3 connection=position-vector(p.m1PreviousPosition);
            const float distance2=m1::dot(connection,connection);
            float pdf_light=0;
            if(p.bounce>0) {
                const float cosine=m1::dot(ng,-m1::normalize(connection));
                if(!(cosine>0)||!(e.area>0)||!(optixLaunchParams.light.numLights>0)){fail(p);return;}
                pdf_light=distance2/(e.area*cosine*optixLaunchParams.light.numLights);
                if(!m1::finite(pdf_light)||!(pdf_light>0)){fail(p);return;}
            }
            p.contribution+=p.albedo*le*light_hit_weight(p,pdf_light);
        }
        p.continueTrace=0;
        if(!m1::finite(p.contribution)||p.contribution<0)fail(p);
        return;
    }
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
            if(!m1::valid_metallic_parameters(bsdf.metallic))fail(p);
        } else fail(p);
    }
    if(p.m1Invalid)return;
    const Vec3 normal=front?ng:-ng;
    const Vec3 facing_ns=m1::dot(ns,-ray_direction)>=0?ns:-ns;
    const auto frame=m1::make_frame(facing_ns);
    const Vec3 wo=m1::normalize(frame.to_local(-ray_direction));
    p.position=vector(position);p.instanceID=instance;

    if(material.materialType!=MATERIAL_TYPE_GLASS&&optixLaunchParams.m1.directMode!=1) {
        if(optixLaunchParams.m1.sunEnabled) {
            const Vec3 direction=sample_sun(p.random(),p.random());
            const Vec3 local=frame.to_local(direction);
            const float le=sun_radiance(p.waveLength),pdf=sun_pdf();
            if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)){fail(p);return;}
            if(local.z>0&&m1::dot(normal,direction)>0&&le>0&&
               visible(p,offset(position,ng,direction),direction,CUDART_INF_F))
                add_direct(p,bsdf,wo,local,le,pdf);
            if(p.m1Invalid)return;
        }
        const int count=optixLaunchParams.light.numLights;
        if(count>0) {
            const int selected=int(p.random()*count);
            if(selected<0||selected>=count){fail(p);return;}
            const auto definition=optixLaunchParams.light.lightDefinition[selected];
            if(definition.lightType!=LIGHT_TYPE_TRIANGLE){fail(p);return;}
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
                    const float le=emission(rgb,p.waveLength,p)*opacity(e.opacityTexture,alpha_uv,p);
                    const float pdf=distance2/(e.area*cos_light*count);
                    if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)){fail(p);return;}
                    const Vec3 origin=offset(position,ng,direction);
                    const Vec3 target=offset(point,en,-direction);
                    const Vec3 shadow_delta=target-origin;
                    const float shadow_distance=m1::length(shadow_delta);
                    if(le>0&&shadow_distance>0&&visible(p,origin,shadow_delta/shadow_distance,shadow_distance))
                        add_direct(p,bsdf,wo,local,le,pdf);
                }
            }
        }
    }
    if(p.m1Invalid)return;
    const auto sampled=bsdf.sample(wo,p.random(),p.random(),p.random());
    if(sampled.status==m1::BsdfSampleStatus::Invalid){fail(p);return;}
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
        fail(p);return;
    }
    if(p.albedo==0){p.continueTrace=0;return;}
    // MIS PDFs are conditional on reaching this vertex and exclude RR in BOTH
    // strategies. Surviving path throughput alone is divided by survival.
    if(p.bounce+1>=int(optixLaunchParams.m1.rrDepth)) {
        const float rr_beta=p.albedo*p.m1EtaScale;
        if(!m1::finite(rr_beta)){fail(p);return;}
        const float survival=fminf(0.95f,rr_beta);
        if(!(survival>0)||p.random()>=survival){p.continueTrace=0;return;}
        p.albedo/=survival;
    }
    p.pdf.bxdf=sampled.pdf;p.m1PreviousDelta=sampled.delta;p.m1PreviousPosition=vector(position);
    p.wi=vector(direction);p.position=vector(offset(position,ng,direction));p.continueTrace=1;
    if(!m1::finite(p.albedo)||!finite3(p.position)||!finite3(p.wi))fail(p);
}

static __forceinline__ __device__ void miss(PRDSpectral& p) {
    if(!p.m1Invalid&&optixLaunchParams.m1.sunEnabled&&inside_sun(vector(p.wi))) {
        const float le=sun_radiance(p.waveLength),pdf=sun_pdf();
        if(!m1::finite(le)||le<0||!m1::finite(pdf)||!(pdf>0)){fail(p);return;}
        p.contribution+=p.albedo*le*light_hit_weight(p,pdf);
    }
    p.continueTrace=0;
    if(!m1::finite(p.contribution)||p.contribution<0)fail(p);
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
        if(!finite3(p.position)||!finite3(p.wi)||!m1::finite(ray.weight)||ray.weight<0)fail(p);
        while(p.continueTrace&&!p.m1Invalid) {
            optixTrace(optixLaunchParams.traversable,p.position,p.wi,0.0f,CUDART_INF_F,0.0f,
                       OptixVisibilityMask(MASK_SURFACE),OPTIX_RAY_FLAG_NONE,
                       RADIANCE_RAY_TYPE,RAY_TYPE_COUNT,RADIANCE_RAY_TYPE,u0,u1);
            if(p.continueTrace)++p.bounce;
        }
        // Exactly one wavelength-PDF division per camera sample. No HWSS,
        // empirically chosen 0.1 multiplier, radiance clamp or NaN-to-black pass.
        const float3 xyz=cie(p.waveLength)*(p.contribution*ray.weight/p.pdf.spectral);
        const float3 rgb=make_float3(3.24096994f*xyz.x-1.53738318f*xyz.y-0.49861076f*xyz.z,
                                    -0.96924364f*xyz.x+1.87596750f*xyz.y+0.04155506f*xyz.z,
                                    0.05563008f*xyz.x-0.20397696f*xyz.y+1.05697151f*xyz.z);
        if(!finite3(rgb))p.m1Invalid=true;
        if(p.m1Invalid) {
            if(optixLaunchParams.m1.invalidSamples)atomicAdd(optixLaunchParams.m1.invalidSamples,1u);
            // The host rejects the entire frame when this diagnostic is nonzero.
            continue;
        }
        mean+=(rgb-mean)/float(sample+1);
        normal_mean+=(p.m1PrimaryNormal-normal_mean)/float(sample+1);
        albedo_mean+=(p.m1PrimaryAlbedo-albedo_mean)/float(sample+1);
        if(!finite3(mean)||!finite3(normal_mean)||!finite3(albedo_mean)) {
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
        if(optixLaunchParams.m1.invalidSamples)atomicAdd(optixLaunchParams.m1.invalidSamples,1u);
    }
}

} // namespace m1surface
