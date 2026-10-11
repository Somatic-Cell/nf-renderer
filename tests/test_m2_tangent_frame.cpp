#include "m2/tangent_frame.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

using m1::Vec2;
using m1::Vec3;

static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
static bool close(float a,float b,float tolerance=2.e-6f){return std::abs(a-b)<=tolerance;}
static bool close(Vec3 a,Vec3 b){return close(a.x,b.x)&&close(a.y,b.y)&&close(a.z,b.z);}

// Independent host arithmetic is deliberately wider than the production
// float path, including the exact 1e-60 determinant and grazing projections.
struct R3 {long double x,y,z;};
static R3 ref(Vec3 v){return {v.x,v.y,v.z};}
static R3 sub(R3 a,R3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static R3 mul(R3 v,long double s){return {v.x*s,v.y*s,v.z*s};}
static long double dot(R3 a,R3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static R3 cross(R3 a,R3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static R3 unit(R3 v){return mul(v,1.0L/std::sqrt(dot(v,v)));}
static long double ref_det(Vec2 a,Vec2 b,Vec2 c){
    return (static_cast<long double>(b.x)-a.x)*(static_cast<long double>(c.y)-a.y)-
           (static_cast<long double>(b.y)-a.y)*(static_cast<long double>(c.x)-a.x);
}
static long double component_error(Vec3 actual,R3 expected){
    return std::max({std::abs(actual.x-expected.x),std::abs(actual.y-expected.y),std::abs(actual.z-expected.z)});
}
static void frame_valid(const m2::UvTangentFrame& frame,Vec3 ns){
    require(frame.valid,"expected a valid UV tangent frame");
    const R3 reference_normal=unit(ref(ns));
    ns={float(reference_normal.x),float(reference_normal.y),float(reference_normal.z)};
    require(close(m1::dot(frame.tangent,frame.tangent),1),"tangent is not unit");
    require(close(m1::dot(frame.bitangent,frame.bitangent),1),"bitangent is not unit");
    require(close(m1::dot(frame.tangent,ns),0),"tangent is not perpendicular to Ns");
    require(close(m1::dot(frame.bitangent,ns),0),"bitangent is not perpendicular to Ns");
    require(close(m1::dot(frame.tangent,frame.bitangent),0),"T/B are not orthogonal");
}

static long double check_reference(Vec3 e1,Vec3 e2,Vec2 a,Vec2 b,Vec2 c,Vec3 ns){
    const auto actual=m2::triangle_uv_tangent_frame(e1,e2,a,b,c,ns);
    frame_valid(actual,ns);
    const long double determinant=ref_det(a,b,c);
    const long double decoded=std::ldexp(static_cast<long double>(actual.uv_determinant),actual.uv_determinant_exponent);
    require((decoded>0)==(determinant>0)&&(decoded<0)==(determinant<0),
            "FP32 scaled UV determinant changed the reference sign or zero classification");
    require(actual.used_degenerate_uv_chart==(determinant==0),
            "only the exactly degenerate UV chart may use the imported fallback");
    if(determinant!=0)
        require(std::abs(decoded/determinant-1.0L)<=4.0L*std::numeric_limits<float>::epsilon(),
                "scaled determinant value does not agree with the independent reference");
    const long double orientation=determinant>0?1.0L:-1.0L;
    const R3 raw_u=determinant==0?ref(e2):mul(sub(
        mul(ref(e1),static_cast<long double>(c.y)-a.y),
        mul(ref(e2),static_cast<long double>(b.y)-a.y)),orientation);
    const R3 bitangent=unit(cross(ref(ns),raw_u));
    const R3 tangent=unit(cross(bitangent,unit(ref(ns))));
    const long double error=std::max(component_error(actual.tangent,tangent),
        component_error(actual.bitangent,mul(bitangent,-orientation)));
    require(error<=8.0L*std::numeric_limits<float>::epsilon(),
            "FP32 frame direction disagrees with independent projected UV derivatives");
    return error;
}

static void test_fp32_range_and_cancellation(){
    const Vec3 e1{2,0,0},e2{0,3,0},ns{.3f,.2f,2.f};
    const std::array<int,8> exponents{{-149,-126,-100,-40,0,40,100,126}};
    long double maximum_frame_error=0;
    unsigned count=0;
    for(int eu:exponents)for(int ev:exponents){
        const float u=std::ldexp(1.0f,eu),v=std::ldexp(1.0f,ev);
        maximum_frame_error=std::max(maximum_frame_error,
            check_reference(e1,e2,{0,0},{u,0},{0,v},ns));
        ++count;
    }
    const float epsilon=std::numeric_limits<float>::epsilon();
    // The exact du2 is 2+epsilon, but subtracting in float rounds it to 2.
    // A naive determinant consequently creates a false degenerate chart.
    const Vec2 a{-1,0},b{1,1},c{1+epsilon,1};
    require((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)==0.0f,
            "coordinate-subtraction cancellation regression was not triggered");
    maximum_frame_error=std::max(maximum_frame_error,check_reference(e1,e2,a,b,c,ns));

    // Two rounded products are equal although the exact determinant is -eps^2.
    maximum_frame_error=std::max(maximum_frame_error,
        check_reference(e1,e2,{0,0},{1+epsilon,1},{1,1-epsilon},ns));
    const float half_maximum=(std::numeric_limits<float>::max)()*0.5f;
    maximum_frame_error=std::max(maximum_frame_error,check_reference(e1,e2,
        {-half_maximum,-half_maximum},{half_maximum,-half_maximum},{-half_maximum,half_maximum},ns));

    // Different edge magnitudes may be separated by 60 orders of magnitude.
    // Project before reducing a vector to a single shared exponent.
    maximum_frame_error=std::max(maximum_frame_error,
        check_reference({1e30f,0,0},{0,1e-30f,0},{0,0},{1,0},{0,1},{0,0,1e-30f}));
    maximum_frame_error=std::max(maximum_frame_error,
        check_reference({1e30f,1e-30f,0},{0,0,1},{0,0},{1,0},{0,1},{1,0,0}));

    // U=(1-eps^2,1,0), N=(1,1,0). Rounding U before projection would erase
    // its valid perpendicular component and reject this grazing normal.
    maximum_frame_error=std::max(maximum_frame_error,
        check_reference({1,1,0},{epsilon,0,0},{0,0},{1,epsilon},{0,1},{1,1,0}));
    maximum_frame_error=std::max(maximum_frame_error,
        check_reference(e1,e2,{0,0},{1,0},{2,0},ns));
    std::cout<<"FP32 frame reference: scaled_charts="<<count
             <<" max_component_error="<<std::setprecision(12)<<maximum_frame_error<<'\n';
}

int main(){
    try {
        const Vec3 e1{2,0,0},e2{0,3,0},ns{0,0,1};
        const Vec2 a{0,0},b{1,0},c{0,1};

        // Synthetic mixed vertex signs with a valid interior barycentric point.
        // This is the old failure condition, not an invalid surface direction.
        const float old_sign=.25f*1.f+.5f*(-1.f)+.25f*1.f;
        require(old_sign==0.f,"the original mixed-handedness regression was not triggered");
        const auto regular=m2::triangle_uv_tangent_frame(e1,e2,a,b,c,ns);
        frame_valid(regular,ns);
        require(close(regular.tangent,{1,0,0})&&close(regular.bitangent,{0,-1,0}),
            "positive UV chart must preserve the Assimp +U/-V convention");
        const Vec3 mapped=m1::normalize(regular.tangent*.3f+regular.bitangent*(-.4f)+ns*std::sqrt(.75f));
        require(close(mapped,{.3f,.4f,std::sqrt(.75f)}),"non-flat normal map axes changed");

        const auto mirrored=m2::triangle_uv_tangent_frame(e1,e2,a,{-1,0},c,ns);
        frame_valid(mirrored,ns);
        require(mirrored.uv_determinant<0&&close(mirrored.tangent,{-1,0,0})&&close(mirrored.bitangent,{0,-1,0}),
            "mirrored UV chart has the wrong tangent orientation");

        // Rigid rotation and world-space reflection change the geometry/frame
        // together; orientation is defined against current world winding.
        const auto rotated=m2::triangle_uv_tangent_frame({0,2,0},{0,0,3},a,b,c,{1,0,0});
        frame_valid(rotated,{1,0,0});
        require(close(rotated.tangent,{0,1,0})&&close(rotated.bitangent,{0,0,-1}),"rotated frame is inconsistent");
        const auto reflected=m2::triangle_uv_tangent_frame({-2,0,0},e2,a,b,c,{0,0,-1});
        frame_valid(reflected,{0,0,-1});
        require(close(reflected.tangent,{-1,0,0})&&close(reflected.bitangent,{0,-1,0}),"reflected frame is inconsistent");

        const Vec3 smooth{.3f,.2f,2.f};
        const auto tilted=m2::triangle_uv_tangent_frame(e1,e2,a,b,c,smooth);
        frame_valid(tilted,smooth);
        const auto rescaled=m2::triangle_uv_tangent_frame(e1,e2,a,b,c,smooth*13.f);
        frame_valid(rescaled,smooth);
        require(close(tilted.tangent,rescaled.tangent)&&close(tilted.bitangent,rescaled.bitangent),
            "Ns magnitude must not affect Gram-Schmidt");

        // detUV=1e-60 underflows in float multiplication, but is a regular chart.
        const auto tiny=m2::triangle_uv_tangent_frame(e1,e2,a,{1.e-30f,0},{0,1.e-30f},ns);
        frame_valid(tiny,ns);
        require(tiny.uv_determinant>0&&close(tiny.tangent,regular.tangent)&&close(tiny.bitangent,regular.bitangent),
            "tiny regular UV chart was rejected or reoriented");
        require(std::abs(std::ldexp(double(tiny.uv_determinant),tiny.uv_determinant_exponent)/
            (double(1.e-30f)*double(1.e-30f))-1.0)<=2.0*std::numeric_limits<float>::epsilon(),
            "tiny determinant mantissa/exponent must reconstruct its actual magnitude");

        const auto degenerate=m2::triangle_uv_tangent_frame(e1,e2,a,b,{2,0},ns);
        frame_valid(degenerate,ns);
        require(degenerate.used_degenerate_uv_chart&&degenerate.uv_determinant==0&&
            close(degenerate.tangent,{0,1,0})&&close(degenerate.bitangent,{-1,0,0}),
            "degenerate UVs must inherit Assimp's edge2/-edge1 chart");
        const auto zero_uv=m2::triangle_uv_tangent_frame(e1,e2,a,a,a,ns);
        frame_valid(zero_uv,ns);
        require(zero_uv.used_degenerate_uv_chart&&close(zero_uv.tangent,degenerate.tangent)&&
            close(zero_uv.bitangent,degenerate.bitangent),"all-zero UVs changed the imported chart");
        require(!m2::triangle_uv_tangent_frame(e1,e2,a,a,a,{0,1,0}).valid,
            "zero projected tangent in the degenerate-UV chart was accepted");
        require(!m2::triangle_uv_tangent_frame(e1,e1,a,b,c,ns).valid,"degenerate geometry was accepted");
        require(!m2::triangle_uv_tangent_frame(e1,e2,a,b,c,{0,0,0}).valid,"zero normal was accepted");
        require(!m2::triangle_uv_tangent_frame(e1,e2,a,b,c,{1,0,0}).valid,"zero projected tangent was accepted");
        const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
        require(!m2::triangle_uv_tangent_frame(e1,e2,{nan,0},b,c,ns).valid,"NaN UV was accepted");
        require(!m2::triangle_uv_tangent_frame({inf,0,0},e2,a,b,c,ns).valid,"infinite geometry was accepted");
        require(!m2::triangle_uv_tangent_frame(e1,e2,a,b,c,{0,nan,1}).valid,"NaN normal was accepted");

        test_fp32_range_and_cancellation();

        std::cout<<"M2 tangent frame: mixed-sign interior regression, non-flat normal map, mirrored UV, "
            "world transforms, smooth normal, tiny regular UV, pinned Assimp degenerate-UV chart "
            "and invalid-input checks passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
