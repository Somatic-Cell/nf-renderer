#pragma once

#include "m1/math.hpp"

namespace m2 {

struct UvTangentFrame {
    m1::Vec3 tangent{};
    m1::Vec3 bitangent{};
    double uv_determinant=0.0;
    bool used_degenerate_uv_chart=false;
    bool valid=false;
};

namespace tangent_detail {
struct D3 { double x,y,z; };
M1_HD inline D3 as_double(m1::Vec3 v){return {v.x,v.y,v.z};}
M1_HD inline D3 mul(D3 v,double s){return {v.x*s,v.y*s,v.z*s};}
M1_HD inline D3 sub(D3 a,D3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
M1_HD inline double dot(D3 a,D3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
M1_HD inline D3 cross(D3 a,D3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
M1_HD inline bool finite(double x){return x==x && fabs(x)<=1.7976931348623157e308;}
M1_HD inline bool unit(D3 v,D3& result){
    const double length2=dot(v,v);
    if(!finite(length2)||!(length2>0.0))return false;
    result=mul(v,1.0/sqrt(length2));
    return finite(result.x)&&finite(result.y)&&finite(result.z);
}
M1_HD inline m1::Vec3 as_float(D3 v){return {float(v.x),float(v.y),float(v.z)};}
} // namespace tangent_detail

// Used for a triangle whose imported vertex frames have different handedness.
// An orientation sign is discrete; barycentrically interpolating +/-1 creates
// an artificial zero inside a valid triangle. Use its regular UV chart instead.
//
// Edges and the smooth shading normal are in world space. The geometric normal
// is the winding of cross(edge1,edge2), so UV chart orientation is sign(detUV).
// The existing Assimp tangent convention uses +dP/du and -dP/dv. Preserve that
// negative-V bitangent convention here; texture sampling and the DirectX normal
// map's green-channel conversion remain the caller's responsibility.
//
// Exactly degenerate UVs inherit the pinned Assimp CalcTangentsProcess chart:
// (du1,dv1,du2,dv2)=(0,1,1,0), hence T=+edge2 and B=-edge1. This is an explicit
// continuation of the importer, not a world-axis replacement or UV cutoff.
// A non-finite input, degenerate triangle or zero projected tangent is rejected.
M1_HD inline UvTangentFrame triangle_uv_tangent_frame(
    m1::Vec3 edge1,m1::Vec3 edge2,m1::Vec2 uv0,m1::Vec2 uv1,m1::Vec2 uv2,
    m1::Vec3 shading_normal)
{
    using namespace tangent_detail;
    UvTangentFrame result;
    const double du1=double(uv1.x)-uv0.x,dv1=double(uv1.y)-uv0.y;
    const double du2=double(uv2.x)-uv0.x,dv2=double(uv2.y)-uv0.y;
    const double determinant=du1*dv2-dv1*du2;
    result.uv_determinant=determinant;
    if(!m1::finite(edge1)||!m1::finite(edge2)||!m1::finite(shading_normal)||
       !m1::finite(uv0.x)||!m1::finite(uv0.y)||!m1::finite(uv1.x)||!m1::finite(uv1.y)||
       !m1::finite(uv2.x)||!m1::finite(uv2.y)||!finite(determinant))
        return result;
    const D3 e1=as_double(edge1),e2=as_double(edge2),n=as_double(shading_normal);
    const double geometric_length2=dot(cross(e1,e2),cross(e1,e2));
    const double n2=dot(n,n);
    if(!finite(geometric_length2)||!(geometric_length2>0.0)||!finite(n2)||!(n2>0.0))
        return result;

    // A division by detUV is unnecessary when only the +u direction is needed.
    // Compute in double so tiny but regular UV charts remain regular.
    // The fallback chart has determinant -1. Keep the measured determinant in
    // the result for diagnostics, and record that the importer chart was used.
    result.used_degenerate_uv_chart=determinant==0.0;
    const double orientation=determinant>0.0?1.0:-1.0;
    D3 tangent;
    const D3 u_direction=result.used_degenerate_uv_chart?e2:
        mul(sub(mul(e1,dv2),mul(e2,dv1)),orientation);
    if(!unit(u_direction,tangent))return result;
    tangent=sub(tangent,mul(n,dot(n,tangent)/n2));
    if(!unit(tangent,tangent))return result;
    D3 normal;
    if(!unit(n,normal))return result;

    // Normalize the cross product and re-orthogonalize the tangent in double.
    // This is the same Gram-Schmidt frame in exact arithmetic, including near
    // grazing smooth normals; it does not choose a new axis or UV orientation.
    D3 bitangent;
    if(!unit(cross(normal,tangent),bitangent))return result;
    if(!unit(cross(bitangent,normal),tangent))return result;
    bitangent=mul(bitangent,-orientation);
    result.tangent=as_float(tangent);
    result.bitangent=as_float(bitangent);
    result.valid=m1::finite(result.tangent)&&m1::finite(result.bitangent);
    return result;
}

} // namespace m2
