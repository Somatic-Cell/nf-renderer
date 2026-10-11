#pragma once

#include <cmath>
#include "m1/math.hpp"

namespace m3::tracking {

// A normalized sum of two FP32 numbers, used only where an Earth-sized
// coordinate or a long ray parameter must retain a small local increment.
// Every operation below is FP32, including the FMA error terms. This is not
// hardware FP64 and does not make arbitrary expressions exact. Compile without
// reassociation/fast-math and with gradual underflow (CUDA's default). The
// error-free transforms below also require separately rounded operations;
// ordinary float temporaries alone do not prevent CUDA FMA contraction.
struct FloatPair { float hi=0, lo=0; };

M1_HD inline bool finite(FloatPair a) { return m1::finite(a.hi)&&m1::finite(a.lo); }
M1_HD inline float value(FloatPair a) { return a.hi+a.lo; }

namespace pair_detail {
// The RN intrinsics prevent the CUDA back end from merging a rounded product
// with a later sum while its FMA residual still refers to the old product.
// The host store similarly preserves binary32 rounding with contraction
// enabled. Only the compensated arithmetic needs these local barriers.
M1_HD inline float add_rn(float a,float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a,b);
#else
    volatile float rounded=a+b;
    return rounded;
#endif
}
M1_HD inline float sub_rn(float a,float b) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a,b);
#else
    volatile float rounded=a-b;
    return rounded;
#endif
}
M1_HD inline float mul_rn(float a,float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a,b);
#else
    volatile float rounded=a*b;
    return rounded;
#endif
}
M1_HD inline FloatPair two_sum(float a,float b) {
    const float s=add_rn(a,b);
    const float v=sub_rn(s,a);
    return {s,add_rn(sub_rn(a,sub_rn(s,v)),sub_rn(b,v))};
}
M1_HD inline FloatPair two_product(float a,float b) {
    const float p=mul_rn(a,b);
    return {p,::fmaf(a,b,-p)};
}
}

M1_HD inline FloatPair normalized(FloatPair a) {
    return pair_detail::two_sum(a.hi,a.lo);
}
M1_HD inline FloatPair negate(FloatPair a) { return {-a.hi,-a.lo}; }
M1_HD inline FloatPair add(FloatPair a,FloatPair b) {
    const auto s=pair_detail::two_sum(a.hi,b.hi);
    const auto t=pair_detail::two_sum(a.lo,b.lo);
    const auto u=pair_detail::two_sum(s.hi,pair_detail::add_rn(s.lo,t.hi));
    return pair_detail::two_sum(u.hi,pair_detail::add_rn(u.lo,t.lo));
}
M1_HD inline FloatPair add(FloatPair a,float b) { return add(a,FloatPair{b,0}); }
M1_HD inline FloatPair sub(FloatPair a,FloatPair b) { return add(a,negate(b)); }
M1_HD inline FloatPair sub(FloatPair a,float b) { return add(a,FloatPair{-b,0}); }
M1_HD inline FloatPair mul(FloatPair a,FloatPair b) {
    const auto p=pair_detail::two_product(a.hi,b.hi);
    float correction=::fmaf(a.hi,b.lo,p.lo);
    correction=::fmaf(a.lo,b.hi,correction);
    correction=::fmaf(a.lo,b.lo,correction);
    return pair_detail::two_sum(p.hi,correction);
}
M1_HD inline FloatPair mul(FloatPair a,float b) {
    const auto p=pair_detail::two_product(a.hi,b);
    return pair_detail::two_sum(p.hi,::fmaf(a.lo,b,p.lo));
}
M1_HD inline FloatPair product(float a,float b) { return pair_detail::two_product(a,b); }
M1_HD inline FloatPair divide(FloatPair a,FloatPair b) {
    const float q=a.hi/b.hi;
    const auto remainder=sub(a,mul(b,q));
    const float correction=value(remainder)/b.hi;
    return pair_detail::two_sum(q,correction);
}
M1_HD inline FloatPair divide(FloatPair a,float b) { return divide(a,FloatPair{b,0}); }
M1_HD inline FloatPair square_root(FloatPair a) {
    if (a.hi == 0 && a.lo == 0) return {};
    const float root=::sqrtf(a.hi);
    const auto remainder=sub(a,product(root,root));
    return add(FloatPair{root,0},value(remainder)/(2.0f*root));
}

// Comparisons require normalized inputs. Public arithmetic preserves this.
M1_HD inline int compare(FloatPair a,FloatPair b) {
    if(a.hi < b.hi)return -1;
    if(a.hi > b.hi)return 1;
    return a.lo < b.lo ? -1 : (a.lo > b.lo ? 1 : 0);
}
M1_HD inline int compare(FloatPair a,float b) { return compare(a,FloatPair{b,0}); }
M1_HD inline FloatPair minimum(FloatPair a,FloatPair b) { return compare(a,b)<0 ? a : b; }
M1_HD inline FloatPair maximum(FloatPair a,FloatPair b) { return compare(a,b)>0 ? a : b; }

// Compensated product-and-add before narrowing a coordinate to one float.
M1_HD inline FloatPair multiply_add(float a,float b,float c) {
    return add(product(a,b),FloatPair{c,0});
}

} // namespace m3::tracking
