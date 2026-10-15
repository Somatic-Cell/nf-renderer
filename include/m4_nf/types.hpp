#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define NF_HD __host__ __device__ __forceinline__
#else
#define NF_HD inline
#endif

namespace m4nf {
constexpr int bins = 64, couplings = 4, hidden = 64, linear_layers = 4;
constexpr int weight_count = 72004;
constexpr float pi = 3.14159265358979323846f;
constexpr float log_four_pi = 2.5310242469692907f;
constexpr float min_bin = 1.0e-5f, min_derivative = 1.0e-4f;
constexpr float derivative_shift = 0.54116665506362915f; // verified by exporter

// Only the physical g/C tables are extended; the NN condition is unchanged.
enum class EndcapPolicy : std::uint32_t { Reject=0, ConstantPhysical=1 };
enum class Status : std::uint32_t {
    success, bad_input, condition_out_of_domain, invalid_asset,
    invalid_spline, nonfinite, unrepresentable_sample
};
NF_HD bool finite(float x) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(x);
#else
    return std::isfinite(x);
#endif
}
NF_HD float nan_value() { return NAN; }
struct Vec3 { float x=0, y=0, z=0; };
NF_HD Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
NF_HD Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
NF_HD Vec3 operator-(Vec3 a){return {-a.x,-a.y,-a.z};}
NF_HD Vec3 operator*(Vec3 a,float b){return {a.x*b,a.y*b,a.z*b};}
NF_HD float dot(Vec3 a,Vec3 b){return ::fmaf(a.x,b.x,::fmaf(a.y,b.y,a.z*b.z));}
NF_HD Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
NF_HD bool unit(Vec3 a){const float n=dot(a,a);return finite(n)&&::fabsf(n-1.0f)<=4.0e-5f;}
NF_HD Vec3 normalize(Vec3 a){return a*(1.0f/::sqrtf(dot(a,a)));}
// Same bounded-libm strategy as M3. No fast-math or approximate intrinsics.
NF_HD float cos_bounded(float a){
    if(!(::fabsf(a)<=2.0f*pi))return nan_value();
    return ::cosf(a);
}
struct SinCos { float sine,cosine; };
NF_HD SinCos sincos_bounded(float a){
    if(!(::fabsf(a)<=2.0f*pi))return {nan_value(),nan_value()};
    SinCos r{};
#if defined(__CUDA_ARCH__)
    ::sincosf(a,&r.sine,&r.cosine);
#else
    r.sine=::sinf(a);r.cosine=::cosf(a);
#endif
    return r;
}
NF_HD float wrap01(float x){return x-::floorf(x);}
NF_HD float softplus(float x){return x>20.0f ? x : ::log1pf(::expf(x));}
NF_HD float silu(float x){return x>=0?x/(1.0f+::expf(-x)):x*::expf(x)/(1.0f+::expf(x));}

// The leading dimension is physical wavelength; every field is [lambda, cosine].
struct ModelView {
    const float* weights;
    const float* wavelengths;
    const float* cosines;
    const float* gs;
    const float* scattering;
    const float* extinction;
    const float* angular_constants;
    std::uint32_t nlambda,ncos;
    float lambda_min,lambda_max;
};
NF_HD int out_dimension(int c,int layer){return layer<3?64:(c%2?97:193);}
NF_HD int in_dimension(int layer){return layer?64:3;}
NF_HD int weight_offset(int c,int layer){
    int at=0;
    for(int i=0;i<c;++i)for(int j=0;j<4;++j)at+=out_dimension(i,j)*(in_dimension(j)+1);
    for(int j=0;j<layer;++j)at+=out_dimension(c,j)*(in_dimension(j)+1);
    return at;
}
struct Context {
    float wavelength=0,cosine=0,external[2]{},a=0,b=0,strength=0;
    float base_logits[64]{};
    float csca=0,cext=0;
    Status status=Status::bad_input;
};
struct Result { float log_pdf=nan_value(); Status status=Status::bad_input; };
struct Sample {
    Vec3 direction{};
    float log_pdf=nan_value();
    Status status=Status::bad_input;
};
struct Frame {
    Vec3 x{},y{},z{};
    NF_HD Vec3 local(Vec3 a)const{return {dot(x,a),dot(y,a),dot(z,a)};}
    NF_HD Vec3 world(Vec3 a)const{return x*a.x+y*a.y+z*a.z;}
};
// Algebraically the projected-axis frame. Cross products retain the small
// transverse component close to axial incidence; no angular threshold is used.
NF_HD bool nf_frame(Vec3 incident,Vec3 axis,Frame& f){
    if(!unit(incident)||!unit(axis))return false;
    f.z=normalize(incident);axis=normalize(axis);
    Vec3 y=cross(f.z,axis);
    const float length=::hypotf(::hypotf(y.x,y.y),y.z);
    if(length>0.0f&&finite(length))f.y=y*(1.0f/length);
    else if(length==0.0f){
        const Vec3 ref=::fabsf(f.z.x)<0.9f?Vec3{1,0,0}:Vec3{0,1,0};
        f.y=normalize(cross(f.z,ref));
    }else return false;
    f.x=normalize(cross(f.y,f.z));f.y=cross(f.z,f.x);
    return unit(f.x)&&unit(f.y);
}
NF_HD float direction_cosine(Vec3 a,Vec3 b){
    if(!unit(a)||!unit(b))return nan_value();
    // Restrict ONLY normalization roundoff to the mathematical [-1,1], not
    // to the narrower physical-table interval. Preserve the signed cosine.
    const float c=dot(normalize(a),normalize(b));
    return ::fminf(1.0f,::fmaxf(-1.0f,c));
}

} // namespace m4nf
