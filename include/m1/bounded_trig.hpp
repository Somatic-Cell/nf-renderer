#pragma once

#include "m1/math.hpp"

namespace m1 {

// The spectral integrator's radian angles are in [-2*pi,2*pi]: sun angles,
// sampled azimuths and camera half-FOV. Keep the ordinary float libm accuracy.
// An explicit range guard lets CUDA remove its huge-argument reduction path,
// which otherwise contains FP64 instructions even for sinf/cosf/tanf.
// This is not fast-math or a clamp; out-of-contract input stays invalid.
M1_HD inline float bounded_sin(float radians) {
    if (!(::fabsf(radians) <= 2.0f * pi)) return NAN;
    return ::sinf(radians);
}
M1_HD inline float bounded_cos(float radians) {
    if (!(::fabsf(radians) <= 2.0f * pi)) return NAN;
    return ::cosf(radians);
}
struct SinCos { float sine, cosine; };
M1_HD inline SinCos bounded_sincos(float radians) {
    if (!(::fabsf(radians) <= 2.0f * pi)) return {NAN, NAN};
    SinCos result{};
#if defined(__CUDA_ARCH__)
    // A single call preserves the common range guard through CUDA's
    // optimization. Separate guarded sinf/cosf calls can reintroduce a
    // redundant huge-angle slow path during common-subexpression folding.
    ::sincosf(radians, &result.sine, &result.cosine);
#else
    result.sine = ::sinf(radians);
    result.cosine = ::cosf(radians);
#endif
    return result;
}
M1_HD inline float bounded_tan(float radians) {
    if (!(::fabsf(radians) <= 2.0f * pi)) return NAN;
    return ::tanf(radians);
}

} // namespace m1
