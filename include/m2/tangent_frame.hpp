#pragma once

#include "m1/fp32_numeric.hpp"
#include <cstring>

namespace m2 {

struct UvTangentFrame {
    m1::Vec3 tangent{};
    m1::Vec3 bitangent{};
    // detUV = uv_determinant * 2^uv_determinant_exponent. A signed normalized
    // mantissa preserves orientation even when the determinant is 1e-60.
    // For a valid result, zero means exactly degenerate UVs, not an
    // underflowed float product.
    float uv_determinant=0.0f;
    int uv_determinant_exponent=0;
    bool used_degenerate_uv_chart=false;
    bool valid=false;
};

namespace tangent_detail {
struct Scaled { float value=0.0f; int exponent=0; };
struct Scaled3 { Scaled x,y,z; };

M1_HD inline Scaled scaled(float value,int exponent=0) {
    if(value==0.0f)return {};
    int adjustment=0;
    const float mantissa=frexpf(value,&adjustment);
    return {mantissa,exponent+adjustment};
}
M1_HD inline Scaled3 scaled(m1::Vec3 v) {
    return {scaled(v.x),scaled(v.y),scaled(v.z)};
}
M1_HD inline int maximum_exponent(Scaled a,Scaled b) {
    if(a.value==0.0f)return b.exponent;
    if(b.value==0.0f)return a.exponent;
    return a.exponent>b.exponent?a.exponent:b.exponent;
}

// Multiplication is performed on normalized float mantissas; powers of two
// stay integers. When products nearly cancel their exponents are close, so
// the explicit FMA residual retains the small nonzero difference.
M1_HD inline Scaled difference_products(Scaled a,Scaled b,Scaled c,Scaled d) {
    const bool first_zero=a.value==0.0f||b.value==0.0f;
    const bool second_zero=c.value==0.0f||d.value==0.0f;
    if(first_zero&&second_zero)return {};
    if(first_zero)return scaled(-c.value*d.value,c.exponent+d.exponent);
    if(second_zero)return scaled(a.value*b.value,a.exponent+b.exponent);
    const int first_exponent=a.exponent+b.exponent,second_exponent=c.exponent+d.exponent;
    const int exponent=first_exponent>second_exponent?first_exponent:second_exponent;
    const float av=ldexpf(a.value,first_exponent-exponent);
    const float cv=ldexpf(c.value,second_exponent-exponent);
    return scaled(m1::fp32::difference_of_products(av,b.value,cv,d.value),exponent);
}
M1_HD inline Scaled difference(float a,float b) {
    const Scaled one{0.5f,1};
    return difference_products(scaled(a),one,scaled(b),one);
}
M1_HD inline Scaled3 cross(Scaled3 a,Scaled3 b) {
    return {difference_products(a.y,b.z,a.z,b.y),
            difference_products(a.z,b.x,a.x,b.z),
            difference_products(a.x,b.y,a.y,b.x)};
}
M1_HD inline bool nonzero(Scaled3 v) {
    return v.x.value!=0.0f||v.y.value!=0.0f||v.z.value!=0.0f;
}
M1_HD inline bool unit(Scaled3 v,m1::Vec3& result) {
    if(!nonzero(v))return false;
    Scaled maximum=v.x;
    if(maximum.value==0.0f||(v.y.value!=0.0f&&v.y.exponent>maximum.exponent))maximum=v.y;
    if(maximum.value==0.0f||(v.z.value!=0.0f&&v.z.exponent>maximum.exponent))maximum=v.z;
    return m1::fp32::unit({ldexpf(v.x.value,v.x.exponent-maximum.exponent),
                          ldexpf(v.y.value,v.y.exponent-maximum.exponent),
                          ldexpf(v.z.value,v.z.exponent-maximum.exponent)},result);
}

M1_HD inline uint32_t float_bits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits=0;
    std::memcpy(&bits,&value,sizeof(bits));
    return bits;
#endif
}

// Rare cancellation fallback for a sum of products of ORIGINAL float inputs.
// A float is an integer significand times a power of two. Its product has at
// most 48 significant bits; three factors have at most 72. Twenty-seven uint32
// words on the 2^-447 lattice cover the complete finite-float triple-product
// range and a sum of eight products, with a spare sign region. This predicate
// using integer arithmetic; it neither executes FP64 nor changes a tiny chart
// into the degenerate importer chart. The common path uses scaled FP32 FMAs.
struct ExactProducts {
    uint32_t words[27]{};

    M1_HD void add_word(unsigned index,uint32_t word,bool negative) {
        while(word!=0u&&index<27u) {
            const uint32_t previous=words[index];
            words[index]=negative?previous-word:previous+word;
            word=negative?(previous<word?1u:0u):(words[index]<previous?1u:0u);
            ++index;
        }
    }
    M1_HD void add(float a,float b,bool subtract=false) {
        const uint32_t aa=float_bits(a),bb=float_bits(b);
        const unsigned ae=(aa>>23u)&255u,be=(bb>>23u)&255u;
        const uint32_t am=(aa&0x7fffffu)|(ae!=0u?0x800000u:0u);
        const uint32_t bm=(bb&0x7fffffu)|(be!=0u?0x800000u:0u);
        if(am==0u||bm==0u)return;
        const int exponent_a=ae!=0u?int(ae)-150:-149;
        const int exponent_b=be!=0u?int(be)-150:-149;
        const unsigned shift=unsigned(exponent_a+exponent_b+447);
        const unsigned index=shift/32u,offset=shift%32u;
        const uint64_t product=uint64_t(am)*uint64_t(bm);
        const uint64_t low=product<<offset;
        const uint32_t high=offset==0u?0u:uint32_t(product>>(64u-offset));
        const bool negative=(((aa^bb)>>31u)!=0u)!=subtract;
        add_word(index,uint32_t(low),negative);
        add_word(index+1u,uint32_t(low>>32u),negative);
        add_word(index+2u,high,negative);
    }
    M1_HD void add(float a,float b,float c,bool subtract) {
        const uint32_t aa=float_bits(a),bb=float_bits(b),cc=float_bits(c);
        const unsigned ae=(aa>>23u)&255u,be=(bb>>23u)&255u,ce=(cc>>23u)&255u;
        const uint32_t am=(aa&0x7fffffu)|(ae!=0u?0x800000u:0u);
        const uint32_t bm=(bb&0x7fffffu)|(be!=0u?0x800000u:0u);
        const uint32_t cm=(cc&0x7fffffu)|(ce!=0u?0x800000u:0u);
        if(am==0u||bm==0u||cm==0u)return;
        const int ea=ae!=0u?int(ae)-150:-149,eb=be!=0u?int(be)-150:-149;
        const int ec=ce!=0u?int(ce)-150:-149;
        const unsigned shift=unsigned(ea+eb+ec+447),index=shift/32u,offset=shift%32u;
        const uint64_t ab=uint64_t(am)*uint64_t(bm);
        const uint64_t lower=uint64_t(uint32_t(ab))*uint64_t(cm);
        const uint64_t upper=(ab>>32u)*uint64_t(cm)+(lower>>32u);
        const uint32_t limbs[3]{uint32_t(lower),uint32_t(upper),uint32_t(upper>>32u)};
        const bool negative=(((aa^bb^cc)>>31u)!=0u)!=subtract;
        for(unsigned i=0;i<3u;++i) {
            add_word(index+i,limbs[i]<<offset,negative);
            if(offset!=0u)add_word(index+i+1u,limbs[i]>>(32u-offset),negative);
        }
    }
    M1_HD Scaled value() {
        const bool negative=(words[26]&0x80000000u)!=0u;
        if(negative) {
            for(unsigned i=0;i<27u;++i)words[i]=~words[i];
            add_word(0u,1u,false);
        }
        int top=26;
        while(top>=0&&words[top]==0u)--top;
        if(top<0)return {};
        float leading=static_cast<float>(words[top]);
        if(top>0)leading+=ldexpf(static_cast<float>(words[top-1]),-32);
        return scaled(negative?-leading:leading,32*top-447);
    }
};

M1_HD inline Scaled exact_uv_determinant(m1::Vec2 a,m1::Vec2 b,m1::Vec2 c) {
    ExactProducts sum;
    sum.add(b.x,c.y); sum.add(b.y,c.x,true);
    sum.add(c.x,a.y); sum.add(c.y,a.x,true);
    sum.add(a.x,b.y); sum.add(a.y,b.x,true);
    return sum.value();
}

M1_HD inline Scaled uv_determinant(m1::Vec2 a,m1::Vec2 b,m1::Vec2 c,
                                  Scaled du1,Scaled dv1,Scaled du2,Scaled dv2) {
    const Scaled determinant=difference_products(du1,dv2,dv1,du2);
    const bool first_zero=du1.value==0.0f||dv2.value==0.0f;
    const bool second_zero=dv1.value==0.0f||du2.value==0.0f;
    const Scaled first=first_zero?Scaled{}:Scaled{1.0f,du1.exponent+dv2.exponent};
    const Scaled second=second_zero?Scaled{}:Scaled{1.0f,dv1.exponent+du2.exponent};
    const int product_exponent=maximum_exponent(first,second);
    // Differences of float coordinates have at most one rounding each. A
    // 2^-18 cancellation threshold safely contains their sign uncertainty.
    if(determinant.value!=0.0f&&determinant.exponent>=product_exponent-18)
        return determinant;
    return exact_uv_determinant(a,b,c);
}

M1_HD inline Scaled u_component(float edge1,float edge2,float v0,float v1,float v2,
                               Scaled dv1,Scaled dv2) {
    const Scaled a=scaled(edge1),b=scaled(edge2);
    const Scaled value=difference_products(a,dv2,b,dv1);
    const bool first_zero=a.value==0.0f||dv2.value==0.0f;
    const bool second_zero=b.value==0.0f||dv1.value==0.0f;
    if(first_zero&&second_zero)return {};
    const Scaled first=first_zero?Scaled{}:Scaled{1.0f,a.exponent+dv2.exponent};
    const Scaled second=second_zero?Scaled{}:Scaled{1.0f,b.exponent+dv1.exponent};
    if(value.value!=0.0f&&value.exponent>=maximum_exponent(first,second)-18)
        return value;
    ExactProducts sum;
    sum.add(edge1,v2); sum.add(edge1,v0,true);
    sum.add(edge2,v1,true); sum.add(edge2,v0);
    return sum.value();
}

M1_HD inline Scaled projected_component(Scaled na,Scaled nb,Scaled ua,Scaled ub,
    float normal_a,float normal_b,float edge1_a,float edge1_b,float edge2_a,float edge2_b,
    float v0,float v1,float v2,float orientation) {
    const Scaled value=difference_products(na,ua,nb,ub);
    const bool first_zero=na.value==0.0f||ua.value==0.0f;
    const bool second_zero=nb.value==0.0f||ub.value==0.0f;
    if(first_zero&&second_zero)return {};
    const Scaled first=first_zero?Scaled{}:Scaled{1.0f,na.exponent+ua.exponent};
    const Scaled second=second_zero?Scaled{}:Scaled{1.0f,nb.exponent+ub.exponent};
    if(value.value!=0.0f&&value.exponent>=maximum_exponent(first,second)-18)return value;

    // A nearly parallel N/U can reveal a residual that is smaller than one
    // float ULP of U itself. Evaluate N cross U from the original factors in
    // this exceptional case instead of projecting an already rounded U.
    ExactProducts sum;
    sum.add(normal_a,edge1_a,v2,false); sum.add(normal_a,edge1_a,v0,true);
    sum.add(normal_a,edge2_a,v1,true); sum.add(normal_a,edge2_a,v0,false);
    sum.add(normal_b,edge1_b,v2,true); sum.add(normal_b,edge1_b,v0,false);
    sum.add(normal_b,edge2_b,v1,false); sum.add(normal_b,edge2_b,v0,true);
    Scaled exact=sum.value();
    exact.value*=orientation;
    return exact;
}
} // namespace tangent_detail

// Used for a triangle whose imported vertex frames have different handedness.
// An orientation sign is discrete; barycentrically interpolating +/-1 creates
// an artificial zero inside a valid triangle. Use its regular UV chart instead.
//
// Edges and the smooth shading normal are in world space. The existing Assimp
// convention uses +dP/du and -dP/dv. Texture filtering and DirectX green-channel
// conversion remain the caller's responsibility.
//
// Exactly degenerate UVs inherit the pinned Assimp chart (0,1,1,0), hence
// T=+edge2 and B=-edge1. There is no small-UV cutoff or substitute world axis.
// Invalid input, a degenerate triangle or a zero projected tangent is rejected.
M1_HD inline UvTangentFrame triangle_uv_tangent_frame(
    m1::Vec3 edge1,m1::Vec3 edge2,m1::Vec2 uv0,m1::Vec2 uv1,m1::Vec2 uv2,
    m1::Vec3 shading_normal)
{
    using namespace tangent_detail;
    UvTangentFrame result;
    if(!m1::finite(edge1)||!m1::finite(edge2)||!m1::finite(shading_normal)||
       !m1::finite(uv0.x)||!m1::finite(uv0.y)||!m1::finite(uv1.x)||!m1::finite(uv1.y)||
       !m1::finite(uv2.x)||!m1::finite(uv2.y))return result;
    const Scaled3 e1=scaled(edge1),e2=scaled(edge2),n=scaled(shading_normal);
    if(!nonzero(tangent_detail::cross(e1,e2))||!nonzero(n))return result;

    const Scaled du1=difference(uv1.x,uv0.x),dv1=difference(uv1.y,uv0.y);
    const Scaled du2=difference(uv2.x,uv0.x),dv2=difference(uv2.y,uv0.y);
    const Scaled determinant=uv_determinant(uv0,uv1,uv2,du1,dv1,du2,dv2);
    result.uv_determinant=determinant.value;
    result.uv_determinant_exponent=determinant.exponent;
    result.used_degenerate_uv_chart=determinant.value==0.0f;
    const float orientation=determinant.value>0.0f?1.0f:-1.0f;
    Scaled3 u_direction=e2;
    if(!result.used_degenerate_uv_chart) {
        u_direction={u_component(edge1.x,edge2.x,uv0.y,uv1.y,uv2.y,dv1,dv2),
                     u_component(edge1.y,edge2.y,uv0.y,uv1.y,uv2.y,dv1,dv2),
                     u_component(edge1.z,edge2.z,uv0.y,uv1.y,uv2.y,dv1,dv2)};
        u_direction.x.value*=orientation;
        u_direction.y.value*=orientation;
        u_direction.z.value*=orientation;
    }

    // cross(N,U) projects U without subtracting nearly equal parallel vectors.
    // Each component retains its own exponent until after this projection, so
    // a large parallel component cannot erase a tiny perpendicular component.
    Scaled3 projected=tangent_detail::cross(n,u_direction);
    if(!result.used_degenerate_uv_chart) {
        projected={projected_component(n.y,n.z,u_direction.z,u_direction.y,
                       shading_normal.y,shading_normal.z,edge1.z,edge1.y,edge2.z,edge2.y,
                       uv0.y,uv1.y,uv2.y,orientation),
                   projected_component(n.z,n.x,u_direction.x,u_direction.z,
                       shading_normal.z,shading_normal.x,edge1.x,edge1.z,edge2.x,edge2.z,
                       uv0.y,uv1.y,uv2.y,orientation),
                   projected_component(n.x,n.y,u_direction.y,u_direction.x,
                       shading_normal.x,shading_normal.y,edge1.y,edge1.x,edge2.y,edge2.x,
                       uv0.y,uv1.y,uv2.y,orientation)};
    }
    m1::Vec3 bitangent{},tangent{},normal{};
    if(!unit(projected,bitangent)||!unit(n,normal))return result;
    if(!m1::fp32::unit(m1::fp32::cross(bitangent,normal),tangent))return result;
    result.tangent=tangent;
    result.bitangent=bitangent*(-orientation);
    result.valid=m1::finite(result.tangent)&&m1::finite(result.bitangent);
    return result;
}

} // namespace m2
