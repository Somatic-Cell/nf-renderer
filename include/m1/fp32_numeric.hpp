#pragma once

#include "m1/math.hpp"

namespace m1::fp32 {

// Explicit fmaf keeps the error term even when surrounding contraction is off.
// Both products must have representable finite intermediates. Callers scale
// their vectors before using this difference of products.
M1_HD inline float difference_of_products(float a, float b, float c, float d) {
    const float cd = c * d;
    const float error = fmaf(-c, d, cd);
    return fmaf(a, b, -cd) + error;
}

M1_HD inline Vec3 cross(Vec3 a, Vec3 b) {
    return {difference_of_products(a.y, b.z, a.z, b.y),
            difference_of_products(a.z, b.x, a.x, b.z),
            difference_of_products(a.x, b.y, a.y, b.x)};
}

M1_HD inline float dot(Vec3 a, Vec3 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

M1_HD inline float max_abs(Vec3 v) {
    return fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
}

M1_HD inline Vec3 scale_pow2(Vec3 v, int exponent) {
    return {ldexpf(v.x, exponent), ldexpf(v.y, exponent), ldexpf(v.z, exponent)};
}

// Scaling each component directly avoids an overflowing reciprocal for tiny
// vectors and a squared norm overflow for large vectors. No length in world
// units is discarded: this routine is used only to obtain a direction.
M1_HD inline bool unit(Vec3 v, Vec3& result) {
    if (!m1::finite(v)) return false;
    const float maximum = max_abs(v);
    if (!(maximum > 0.0f)) return false;
    int exponent = 0;
    frexpf(maximum, &exponent);
    const Vec3 scaled = scale_pow2(v, -exponent);
    const float length_squared = m1::fp32::dot(scaled, scaled);
    if (!m1::finite(length_squared) || !(length_squared > 0.0f)) return false;
    const float inverse_length = 1.0f / sqrtf(length_squared);
    result = scaled * inverse_length;
    return m1::finite(result);
}

} // namespace m1::fp32
