#pragma once

#include "m1/math.hpp"

namespace m2 {

// A negative cutoff denotes the existing stochastic coverage model. ORCA's
// explicit material profile uses the basic alpha-mask test at cutoff 0.5.
// Camera hits, shadow hits and sampled-emitter radiance share this definition.
// There is no additional opacity multiplier after a hit has been accepted.
M1_HD inline bool surface_coverage(float texture_alpha, float opacity_factor,
                                   float cutoff, float& coverage) {
    if (!m1::finite(texture_alpha) || texture_alpha < 0.0f || texture_alpha > 1.0f ||
        !m1::finite(opacity_factor) || opacity_factor < 0.0f || opacity_factor > 1.0f ||
        !m1::finite(cutoff) || (cutoff != -1.0f && (cutoff < 0.0f || cutoff > 1.0f)))
        return false;
    const float alpha = texture_alpha * opacity_factor;
    coverage = cutoff < 0.0f ? alpha : (alpha < cutoff ? 0.0f : 1.0f);
    return true;
}

} // namespace m2
