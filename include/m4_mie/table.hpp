#pragma once
// FP32, device-safe evaluator/sampler for an offline 1-mm Lorenz-Mie table.
// The source angular coordinate is theta, NOT 1-cos(theta): the latter loses
// the narrow forward diffraction peak when evaluated directly in FP32.
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#include "m1/bounded_trig.hpp"
#define M4MIE_HD __host__ __device__
#else
#define M4MIE_HD
#endif

namespace m4mie {
constexpr float pi = 3.14159265358979323846f;
struct Vec3 {float x,y,z;};
struct View {
    const float* wavelengths_nm = nullptr;  // nw
    const float* theta_edges_rad = nullptr; // nbins+1
    const float* masses = nullptr;          // nw*nbins, normalized cell probabilities
    const float* alias_prob = nullptr;      // nw*nbins
    const std::uint32_t* alias = nullptr;   // nw*nbins
    const float* qext = nullptr;            // nw
    const float* qsca = nullptr;            // nw
    const float* g = nullptr;               // nw
    std::uint32_t nw = 0, nbins = 0;
    float radius_m = 0.001f;
};
enum class CrossSection : std::uint32_t {
    GeometricMatched = 0, // explicitly compare phase at common pi*r*r cross section
    PhysicalMie = 1,      // Qsca*area, Qext*area
};
struct Response {
    bool valid = false;
    float phase_sr_inv = 0;
    float csca_m2 = 0;
    float cext_m2 = 0;
};
struct Sample {
    bool valid = false;
    Vec3 direction{};
    float pdf_sr_inv = 0;
    float csca_m2 = 0, cext_m2 = 0;
    std::uint32_t bin = 0;
};
M4MIE_HD inline bool finite(float v) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(v) != 0;
#else
    return std::isfinite(v);
#endif
}
M4MIE_HD inline float sin_bounded(float x) {
#if defined(__CUDA_ARCH__)
    return m1::bounded_sin(x);
#else
    return ::sinf(x);
#endif
}
M4MIE_HD inline float cos_bounded(float x) {
#if defined(__CUDA_ARCH__)
    return m1::bounded_cos(x);
#else
    return ::cosf(x);
#endif
}
M4MIE_HD inline Vec3 mul(Vec3 v,float s){return {v.x*s,v.y*s,v.z*s};}
M4MIE_HD inline Vec3 add(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
M4MIE_HD inline float dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
M4MIE_HD inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
M4MIE_HD inline float norm2(Vec3 a){return dot(a,a);}
M4MIE_HD inline Vec3 normalize(Vec3 v){return mul(v,1.0f/::sqrtf(norm2(v)));}
M4MIE_HD inline bool unit(Vec3 v){return finite(v.x)&&finite(v.y)&&finite(v.z)&&::fabsf(norm2(v)-1)<2e-4f;}

M4MIE_HD inline bool bracket(const View& v,float lambda,std::uint32_t& a,
                              std::uint32_t& b,float& frac) {
    if(!v.wavelengths_nm||!v.theta_edges_rad||!v.masses||!v.qext||!v.qsca||
       v.nw<1||v.nbins<1||!finite(lambda)||
       lambda<v.wavelengths_nm[0]||lambda>v.wavelengths_nm[v.nw-1])return false;
    if(v.nw==1){a=b=0;frac=0;return lambda==v.wavelengths_nm[0];}
    a=0;b=v.nw-1;
    while(b-a>1){const auto mid=(a+b)/2; if(lambda<v.wavelengths_nm[mid])b=mid;else a=mid;}
    const float dl=v.wavelengths_nm[b]-v.wavelengths_nm[a];
    if(!(dl>0))return false;
    frac=(lambda-v.wavelengths_nm[a])/dl;
    return finite(frac)&&frac>=0&&frac<=1;
}
M4MIE_HD inline bool spectral_weights(const View& v,float wavelength,CrossSection cs,
                                       std::uint32_t& a,std::uint32_t& b,
                                       float& wa,float& wb,float& scattering,float& extinction){
    float t=0;
    if(!bracket(v,wavelength,a,b,t))return false;
    const float area=pi*v.radius_m*v.radius_m;
    if(!finite(area)||!(area>0))return false;
    if(cs==CrossSection::PhysicalMie){
        const float sa=area*v.qsca[a],sb=area*v.qsca[b];
        scattering=(1-t)*sa+t*sb;
        extinction=area*((1-t)*v.qext[a]+t*v.qext[b]);
        if(!(scattering>0)||!(extinction>0)||!finite(scattering)||!finite(extinction))return false;
        wa=(1-t)*sa/scattering;
    }else if(cs==CrossSection::GeometricMatched){
        scattering=extinction=area;
        wa=1-t;
    }else return false;
    wb=1-wa;
    return finite(wa)&&wa>=0&&wa<=1&&finite(wb)&&wb>=0&&wb<=1;
}
M4MIE_HD inline float solid_angle_width(float th0,float th1){
    // DeltaOmega = 2pi(cos(th0)-cos(th1)). Avoid cosine cancellation at 0.
    return (4.0f*pi)*sin_bounded(0.5f*(th1+th0))*sin_bounded(0.5f*(th1-th0));
}
M4MIE_HD inline std::uint32_t bin_for_theta(const View& v,float theta){
    if(theta>=v.theta_edges_rad[v.nbins])return v.nbins-1;
    std::uint32_t lo=0,hi=v.nbins;
    while(hi-lo>1){const std::uint32_t m=(lo+hi)/2;
        if(theta<v.theta_edges_rad[m])hi=m;else lo=m;}
    return lo;
}
M4MIE_HD inline bool scattering_angle(Vec3 a,Vec3 b,float& theta) {
    if(!unit(a)||!unit(b))return false;
    const Vec3 v=cross(a,b);
    theta=::atan2f(::sqrtf(norm2(v)),dot(a,b));
    return finite(theta)&&theta>=0&&theta<=pi;
}
M4MIE_HD inline Response evaluate(const View& v,Vec3 camera_direction,Vec3 continuation_direction,
                                   float wavelength,CrossSection cs){
    Response r{};std::uint32_t a=0,b=0;float wa=0,wb=0;
    if(!spectral_weights(v,wavelength,cs,a,b,wa,wb,r.csca_m2,r.cext_m2))return r;
    float theta=0;
    if(!scattering_angle(camera_direction,continuation_direction,theta))return r;
    const auto bin=bin_for_theta(v,theta);
    const float domega=solid_angle_width(v.theta_edges_rad[bin],v.theta_edges_rad[bin+1]);
    if(!finite(domega)||!(domega>0))return r;
    const float mass=wa*v.masses[a*v.nbins+bin]+wb*v.masses[b*v.nbins+bin];
    r.phase_sr_inv=mass/domega;
    r.valid=finite(r.phase_sr_inv)&&r.phase_sr_inv>=0;
    return r;
}
// Sample wavelength endpoint with the Csca-weighted mixture coefficients,
// then Walker alias cell and uniform solid angle within that cell.
// Both wavelength and alias branch consume one uniform by remapping its interval.
M4MIE_HD inline Sample sample(const View& v,Vec3 camera_direction,float wavelength,
                               CrossSection cs,float uniform_bin,float uniform_phi){
    Sample out{};
    if(!unit(camera_direction)||!finite(uniform_bin)||!finite(uniform_phi)||
       uniform_bin<0||uniform_bin>=1||uniform_phi<0||uniform_phi>=1||
       !v.alias_prob||!v.alias)return out;
    std::uint32_t a=0,b=0;float wa=0,wb=0;
    if(!spectral_weights(v,wavelength,cs,a,b,wa,wb,out.csca_m2,out.cext_m2))return out;
    float u=uniform_bin;
    std::uint32_t row=a;
    if(wb>0){
        if(u<wa){row=a;u=u/wa;}
        else{row=b;u=(u-wa)/wb;}
    }
    if(u>=1)u=0x1.fffffep-1f;
    const float z=u*float(v.nbins);
    std::uint32_t column=static_cast<std::uint32_t>(z);
    if(column>=v.nbins)column=v.nbins-1;
    const float fractional=z-float(column);
    const auto offset=row*v.nbins+column;
    const float threshold=v.alias_prob[offset];
    if(!finite(threshold)||threshold<0||threshold>1)return out;
    float cell_u=0;
    if(fractional<threshold){
        out.bin=column;
        if(!(threshold>0))return out;
        cell_u=fractional/threshold;
    }else{
        out.bin=v.alias[offset];
        if(!(threshold<1)||out.bin>=v.nbins)return out;
        cell_u=(fractional-threshold)/(1-threshold);
    }
    if(cell_u>=1)cell_u=0x1.fffffep-1f;
    const float th0=v.theta_edges_rad[out.bin],th1=v.theta_edges_rad[out.bin+1];
    const float sh0=sin_bounded(th0*0.5f),sh1=sin_bounded(th1*0.5f);
    const float s=sh0*sh0+cell_u*(sh1*sh1-sh0*sh0); // sin^2(theta/2), stable at theta≈0
    if(!(s>=0&&s<=1))return out;
    const float cos_th=1-2*s;
    const float sin_th=2*::sqrtf(s)*::sqrtf(1-s);
    const Vec3 helper=::fabsf(camera_direction.z)<0.9f ? Vec3{0,0,1}:Vec3{0,1,0};
    const Vec3 e0=normalize(cross(helper,camera_direction)),e1=cross(camera_direction,e0);
    const float phi=2*pi*uniform_phi-pi;
    const Vec3 w=add(mul(camera_direction,cos_th),
                     add(mul(e0,sin_th*cos_bounded(phi)),mul(e1,sin_th*sin_bounded(phi))));
    if(!(norm2(w)>0)||!finite(norm2(w)))return out;
    out.direction=normalize(w);
    const auto check=evaluate(v,camera_direction,out.direction,wavelength,cs);
    out.valid=check.valid&&check.phase_sr_inv>0;
    out.pdf_sr_inv=check.phase_sr_inv;
    return out;
}
} // namespace m4mie
