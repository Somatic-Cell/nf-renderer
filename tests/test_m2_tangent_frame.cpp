#include "m2/tangent_frame.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using m1::Vec2;
using m1::Vec3;

static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
static bool close(float a,float b,float tolerance=2.e-6f){return std::abs(a-b)<=tolerance;}
static bool close(Vec3 a,Vec3 b){return close(a.x,b.x)&&close(a.y,b.y)&&close(a.z,b.z);}
static void frame_valid(const m2::UvTangentFrame& frame,Vec3 ns){
    require(frame.valid,"expected a valid UV tangent frame");
    ns=m1::normalize(ns);
    require(close(m1::dot(frame.tangent,frame.tangent),1),"tangent is not unit");
    require(close(m1::dot(frame.bitangent,frame.bitangent),1),"bitangent is not unit");
    require(close(m1::dot(frame.tangent,ns),0),"tangent is not perpendicular to Ns");
    require(close(m1::dot(frame.bitangent,ns),0),"bitangent is not perpendicular to Ns");
    require(close(m1::dot(frame.tangent,frame.bitangent),0),"T/B are not orthogonal");
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

        std::cout<<"M2 tangent frame: mixed-sign interior regression, non-flat normal map, mirrored UV, "
            "world transforms, smooth normal, tiny regular UV, pinned Assimp degenerate-UV chart "
            "and invalid-input checks passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
