#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define M1_HD __host__ __device__
#else
#define M1_HD
#endif

namespace m1 {
inline constexpr float pi = 3.14159265358979323846f;
struct Vec2 { float x=0, y=0; };
struct Vec3 { float x=0, y=0, z=0; };
struct UInt3 { std::uint32_t x=0, y=0, z=0; };
M1_HD inline Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
M1_HD inline Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
M1_HD inline Vec3 operator-(Vec3 a){return {-a.x,-a.y,-a.z};}
M1_HD inline Vec3 operator*(Vec3 a,float b){return {a.x*b,a.y*b,a.z*b};}
M1_HD inline Vec3 operator*(float a,Vec3 b){return b*a;}
M1_HD inline Vec3 operator/(Vec3 a,float b){return a*(1.0f/b);}
M1_HD inline Vec3& operator+=(Vec3& a,Vec3 b){a=a+b;return a;}
M1_HD inline float dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
M1_HD inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
M1_HD inline float length(Vec3 a){return sqrtf(dot(a,a));}
M1_HD inline Vec3 normalize(Vec3 a){return a/length(a);}
M1_HD inline float clamp(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
M1_HD inline float max_component(Vec3 a){return fmaxf(a.x,fmaxf(a.y,a.z));}
M1_HD inline bool finite(float a){return a==a && fabsf(a)<=3.402823466e38f;}
M1_HD inline bool finite(Vec3 a){return finite(a.x)&&finite(a.y)&&finite(a.z);}
struct Frame {
    Vec3 x,y,z;
    M1_HD Vec3 to_world(Vec3 v) const {return x*v.x+y*v.y+z*v.z;}
    M1_HD Vec3 to_local(Vec3 v) const {return {dot(v,x),dot(v,y),dot(v,z)};}
};
M1_HD inline Frame make_frame(Vec3 z){
    const Vec3 a=fabsf(z.z)<0.999f?Vec3{0,0,1}:Vec3{0,1,0};
    const Vec3 x=normalize(cross(a,z));
    return {x,cross(z,x),z};
}
M1_HD inline float power_heuristic(float a,float b){
    const float scale=fmaxf(a,b);
    if(!(scale>0))return 0;
    a/=scale;b/=scale;return a*a/(a*a+b*b);
}
// One PCG-style integer stream per pixel/sample. Values are strictly inside (0,1).
struct Rng {
    std::uint32_t state;
    M1_HD std::uint32_t next_u32(){
        const std::uint32_t old=state;
        state=old*747796405u+2891336453u;
        const std::uint32_t word=((old>>((old>>28u)+4u))^old)*277803737u;
        return (word>>22u)^word;
    }
    M1_HD float uniform(){return (float(next_u32()>>9u)+0.5f)*(1.0f/8388608.0f);}
};
M1_HD inline std::uint32_t mix_bits(std::uint32_t x){
    x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;
}
}
