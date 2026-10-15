#pragma once

#include "m1/bounded_trig.hpp"

#include "m1/math.hpp"

namespace m1 {

enum class MaterialKind : unsigned { Diffuse, GgxReflection, Dielectric };

// All directions are unit vectors in a frame whose +z is the shading normal
// facing wo. The continuous models reflect only; the ideal dielectric also
// transmits into z < 0. The caller applies geometric-normal support restrictions
// consistently to evaluation and sampling. Nested dielectric media are not used.
struct BsdfParameters {
    MaterialKind kind=MaterialKind::Diffuse;
    float color=0.5f;       // rho for Lambert, Schlick F0 for GGX, unused for glass
    float alpha=0.2f;       // direct GGX slope parameter, in [1e-4, 1]
    float eta_i_over_t=1.0f; // only dielectric; incident / transmitted IOR
};

struct BsdfEvaluation {
    float f=0.0f;          // sr^-1, zero for delta BSDFs
    float pdf=0.0f;        // continuous solid-angle density, never a delta mass
    bool valid=true;
};

enum class BsdfSampleStatus : unsigned { Valid, Null, Invalid };

struct BsdfSample {
    Vec3 wi{};
    float weight=0.0f;     // f * abs(cos theta_i) / pdf, including radiance eta^2
    float pdf=0.0f;        // continuous density; exactly zero for delta events
    float branch_pmf=0.0f; // reflection/transmission mass; only for delta events
    float eta_scale=1.0f;  // multiply into the separate RR eta compensation
    bool delta=false;
    bool transmission=false;
    BsdfSampleStatus status=BsdfSampleStatus::Null;
};

M1_HD inline bool bsdf_unit_direction(Vec3 w) {
    return finite(w) && fabsf(dot(w,w)-1.0f)<=2.0e-4f;
}

struct BsdfFrame {
    Frame frame;
    Vec3 wo;
};

// The caller supplies a unit shading normal and a finite, nonzero world-space
// outgoing direction. Decide the side from the very same local direction that
// the BSDF will consume. Separately faceforwarding Ns and then recomputing this
// dot product can disagree at grazing angles after FP32 rounding/contraction.
M1_HD inline BsdfFrame make_bsdf_frame(Vec3 normal,Vec3 wo_world) {
    BsdfFrame result;
    result.frame=make_frame(normal);
    result.wo=normalize(result.frame.to_local(wo_world));
    if(result.wo.z<0.0f) {
        // A 180-degree basis rotation about y preserves handedness and the
        // represented world direction. Flip both the frame and its coordinate;
        // changing only wo.z would change the direction used by evaluation.
        result.frame.x=-result.frame.x;
        result.frame.z=-result.frame.z;
        result.wo.x=-result.wo.x;
        result.wo.z=-result.wo.z;
    }
    return result;
}

M1_HD inline bool valid_bsdf_parameters(const BsdfParameters& p) {
    if(p.kind==MaterialKind::Dielectric)
        return finite(p.eta_i_over_t) && p.eta_i_over_t>0.0f;
    if(p.kind!=MaterialKind::Diffuse && p.kind!=MaterialKind::GgxReflection)
        return false;
    if(!finite(p.color) || p.color<0.0f || p.color>1.0f)return false;
    return p.kind!=MaterialKind::GgxReflection ||
        (finite(p.alpha) && p.alpha>=1.0e-4f && p.alpha<=1.0f);
}

M1_HD inline float schlick_fresnel(float f0,float cosine) {
    // Unit-vector dot products may exceed one by a few floating-point ulps.
    const float x=1.0f-fminf(1.0f,cosine);
    const float x2=x*x;
    return f0+(1.0f-f0)*x2*x2*x;
}

// Algebraic GGX/Smith forms avoid tan(theta) and the cancellation in
// 1 + (alpha^2 - 1) * cos^2(theta) at very small alpha.
M1_HD inline float ggx_distribution(Vec3 m,float alpha) {
    const float a2=alpha*alpha;
    const float d=m.x*m.x+m.y*m.y+a2*m.z*m.z;
    return a2/(pi*d*d);
}

M1_HD inline float ggx_projected_length(Vec3 w,float alpha) {
    return sqrtf(w.z*w.z+alpha*alpha*(w.x*w.x+w.y*w.y));
}

M1_HD inline BsdfEvaluation evaluate_bsdf(const BsdfParameters& p,Vec3 wo,Vec3 wi) {
    if(!valid_bsdf_parameters(p) || !bsdf_unit_direction(wo) ||
       !bsdf_unit_direction(wi) || wo.z<0.0f)return {0,0,false};
    if(wo.z==0.0f || wi.z<=0.0f || p.kind==MaterialKind::Dielectric)return {};
    if(p.kind==MaterialKind::Diffuse)return {p.color/pi,wi.z/pi,true};

    const Vec3 h=wo+wi;
    const float h2=dot(h,h);
    if(!(h2>0.0f) || !finite(h2))return {0,0,false};
    const Vec3 m=h/sqrtf(h2);
    // For unit directions, dot(wo,m) = length(wo+wi)/2. This symmetric form
    // preserves reciprocity when float normalization leaves slightly unequal
    // lengths at grazing angles.
    const float om=0.5f*sqrtf(h2);
    if(!(m.z>0.0f) || !(om>0.0f))return {};
    const float d=ggx_distribution(m,p.alpha);
    const float so=ggx_projected_length(wo,p.alpha);
    const float si=ggx_projected_length(wi,p.alpha);
    // G2 = 1 / (1 + Lambda(wo) + Lambda(wi)), height-correlated Smith.
    const float f=schlick_fresnel(p.color,om)*d/
        (2.0f*(wo.z*si+wi.z*so));
    // Visible-normal PDF times the reflection Jacobian 1/(4*dot(wo,m)).
    const float pdf=d/(2.0f*(wo.z+so));
    return {f,pdf,finite(f)&&finite(pdf)&&f>=0.0f&&pdf>0.0f};
}

struct DielectricFresnel {
    float reflectance=0.0f;
    float cos_theta_t=0.0f;
    bool total_internal_reflection=false;
    bool valid=true;
};

M1_HD inline DielectricFresnel dielectric_fresnel(float cosine,float eta_i_over_t) {
    if(!finite(cosine) || cosine<0.0f || cosine>1.000002f ||
       !finite(eta_i_over_t) || !(eta_i_over_t>0.0f))return {0,0,false,false};
    const float ci=fminf(cosine,1.0f);
    if(eta_i_over_t==1.0f)return {0,ci,false,true};
    const float st=eta_i_over_t*sqrtf(fmaxf(0.0f,1.0f-ci*ci));
    if(st>=1.0f)return {1,0,true,true};
    const float ct=sqrtf(fmaxf(0.0f,1.0f-st*st));
    const float rp=(ci-eta_i_over_t*ct)/(ci+eta_i_over_t*ct);
    const float rs=(eta_i_over_t*ci-ct)/(eta_i_over_t*ci+ct);
    const float f=0.5f*(rp*rp+rs*rs);
    return {f,ct,false,finite(f)&&f>=0.0f&&f<=1.0f};
}

// Heitz 2018 visible-normal sampling, specialized to isotropic GGX. The max(0)
// operations below only protect square roots/hemisphere reconstruction against
// roundoff. There is no roughness clamp, normal-z floor, or resampling loop.
M1_HD inline Vec3 sample_ggx_visible_normal(Vec3 wo,float alpha,float u1,float u2) {
    const Vec3 v=normalize(Vec3{alpha*wo.x,alpha*wo.y,wo.z});
    const float xy2=v.x*v.x+v.y*v.y;
    const Vec3 t1=xy2>0.0f ? Vec3{-v.y,v.x,0}/sqrtf(xy2) : Vec3{1,0,0};
    const Vec3 t2=cross(v,t1);
    const float radius=sqrtf(u1),phi=2.0f*pi*u2;
    const auto azimuth=m1::bounded_sincos(phi);
    const float x=radius*azimuth.cosine;
    float y=radius*azimuth.sine;
    const float s=0.5f*(1.0f+v.z);
    y=(1.0f-s)*sqrtf(fmaxf(0.0f,1.0f-x*x))+s*y;
    const float z=sqrtf(fmaxf(0.0f,1.0f-x*x-y*y));
    const Vec3 nh=x*t1+y*t2+z*v;
    return normalize(Vec3{alpha*nh.x,alpha*nh.y,fmaxf(0.0f,nh.z)});
}

M1_HD inline BsdfSample sample_bsdf(const BsdfParameters& p,Vec3 wo,
                                    float u_branch,float u1,float u2) {
    BsdfSample r;
    if(!valid_bsdf_parameters(p) || !bsdf_unit_direction(wo) || wo.z<0.0f ||
       !(u_branch>=0.0f&&u_branch<1.0f) || !(u1>=0.0f&&u1<1.0f) ||
       !(u2>=0.0f&&u2<1.0f)) {
        r.status=BsdfSampleStatus::Invalid;return r;
    }
    if(wo.z==0.0f)return r;
    if(p.kind==MaterialKind::Dielectric) {
        const auto fresnel=dielectric_fresnel(wo.z,p.eta_i_over_t);
        if(!fresnel.valid){r.status=BsdfSampleStatus::Invalid;return r;}
        r.delta=true;
        if(fresnel.total_internal_reflection || u_branch<fresnel.reflectance) {
            r.wi={-wo.x,-wo.y,wo.z};
            r.branch_pmf=fresnel.reflectance;
            r.weight=1.0f;
        } else {
            r.wi={-p.eta_i_over_t*wo.x,-p.eta_i_over_t*wo.y,-fresnel.cos_theta_t};
            r.branch_pmf=1.0f-fresnel.reflectance;
            r.weight=p.eta_i_over_t*p.eta_i_over_t;
            r.eta_scale=1.0f/r.weight;
            r.transmission=true;
        }
        r.status=finite(r.weight)&&finite(r.eta_scale)&&r.branch_pmf>0.0f&&
            bsdf_unit_direction(r.wi) ? BsdfSampleStatus::Valid : BsdfSampleStatus::Invalid;
        return r;
    }
    if(p.kind==MaterialKind::Diffuse) {
        const float radius=sqrtf(u1),phi=2.0f*pi*u2;
        const auto azimuth=m1::bounded_sincos(phi);
        r.wi={radius*azimuth.cosine,radius*azimuth.sine,sqrtf(1.0f-u1)};
        r.pdf=r.wi.z/pi;
        r.weight=p.color;
    } else {
        const Vec3 m=sample_ggx_visible_normal(wo,p.alpha,u1,u2);
        if(!bsdf_unit_direction(m)){r.status=BsdfSampleStatus::Invalid;return r;}
        r.wi=2.0f*dot(wo,m)*m-wo;
        if(!finite(r.wi)){r.status=BsdfSampleStatus::Invalid;return r;}
        // A visible microfacet can reflect below the macrosurface. This is a
        // real zero-contribution/null event, not a conditional PDF to renormalize.
        if(r.wi.z<=0.0f)return r;
        const auto e=evaluate_bsdf(p,wo,r.wi);
        if(!e.valid){r.status=BsdfSampleStatus::Invalid;return r;}
        r.pdf=e.pdf;
        // Cancel D analytically in f*cos/pdf for stable near-specular weights.
        const float so=ggx_projected_length(wo,p.alpha);
        const float si=ggx_projected_length(r.wi,p.alpha);
        r.weight=schlick_fresnel(p.color,0.5f*length(wo+r.wi))*r.wi.z*(wo.z+so)/
            (wo.z*si+r.wi.z*so);
    }
    r.status=bsdf_unit_direction(r.wi)&&finite(r.weight)&&r.weight>=0.0f&&
        finite(r.pdf)&&r.pdf>0.0f ? BsdfSampleStatus::Valid : BsdfSampleStatus::Invalid;
    return r;
}

// The existing renderer calls this simplified material "principled". Keep its
// actual model explicit: (1-metallic)*Lambert(rho) + GGX(Schlick F0), with
// F0=(1-metallic)*0.04+metallic*rho. This is NOT the full Disney BSDF and has no
// layered energy compensation. Sampling probabilities are normalized; the
// material lobes themselves are not silently rescaled.
struct MetallicParameters {
    float color=0.5f;
    float metallic=0.0f;
    float roughness=0.2f;
};
M1_HD inline bool valid_metallic_parameters(const MetallicParameters& p) {
    return finite(p.color)&&p.color>=0&&p.color<=1&&finite(p.metallic)&&
        p.metallic>=0&&p.metallic<=1&&finite(p.roughness)&&p.roughness>=0&&p.roughness<=1;
}
M1_HD inline float legacy_principled_alpha(float roughness) {
    // Explicitly preserve the old renderer's regularized roughness convention.
    // It never represents a delta mirror, including when roughness is zero.
    return fmaxf(roughness*roughness,1.0e-3f);
}
M1_HD inline BsdfEvaluation evaluate_metallic(const MetallicParameters& p,Vec3 wo,Vec3 wi) {
    if(!valid_metallic_parameters(p))return {0,0,false};
    const float diffuse=1.0f-p.metallic;
    const float pd=diffuse/(diffuse+1.0f);
    const float f0=0.04f*diffuse+p.color*p.metallic;
    const auto d=evaluate_bsdf({MaterialKind::Diffuse,p.color,1,1},wo,wi);
    const auto s=evaluate_bsdf({MaterialKind::GgxReflection,f0,
                              legacy_principled_alpha(p.roughness),1},wo,wi);
    const float f=diffuse*d.f+s.f,pdf=pd*d.pdf+(1.0f-pd)*s.pdf;
    return {f,pdf,d.valid&&s.valid&&finite(f)&&finite(pdf)};
}
M1_HD inline BsdfSample sample_metallic(const MetallicParameters& p,Vec3 wo,
                                       float branch,float u1,float u2) {
    if(!valid_metallic_parameters(p)) {
        BsdfSample r;r.status=BsdfSampleStatus::Invalid;return r;
    }
    const float diffuse=1.0f-p.metallic,pd=diffuse/(diffuse+1.0f);
    const float f0=0.04f*diffuse+p.color*p.metallic;
    const BsdfParameters lobe=branch<pd ? BsdfParameters{MaterialKind::Diffuse,p.color,1,1} :
        BsdfParameters{MaterialKind::GgxReflection,f0,legacy_principled_alpha(p.roughness),1};
    BsdfSample r=sample_bsdf(lobe,wo,branch,u1,u2);
    if(r.status!=BsdfSampleStatus::Valid)return r;
    const auto v=evaluate_metallic(p,wo,r.wi);
    if(!v.valid || !(v.pdf>0)) {r.status=BsdfSampleStatus::Invalid;return r;}
    r.pdf=v.pdf;r.weight=v.f*r.wi.z/v.pdf;
    if(!finite(r.weight)||r.weight<0)r.status=BsdfSampleStatus::Invalid;
    return r;
}

} // namespace m1
