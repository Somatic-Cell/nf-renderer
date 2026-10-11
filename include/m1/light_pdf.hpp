#pragma once

#include "m1/math.hpp"
#include <cfloat>

namespace m1 {

struct AreaLightPdfResult {
    float pdf = 0.0f; // Solid-angle density; zero can be a valid support result.
    bool valid = false;
};

// The existing strategy chooses one of light_count triangles uniformly, then
// samples its full area uniformly. light_cosine is the signed cosine from the
// emitter's geometric normal and the connection toward the previous vertex.
// The caller supplies unit directions/normals; no absolute value, angular
// floor, area renormalization, or sampler change is performed here.
//
// A finite nonpositive cosine is outside one-sided emission support. It has
// zero competing NEE density even if a displaced traced ray hits the front.
// Validate the remaining geometry first so a zero-support result cannot hide
// nonfinite inputs or degenerate geometry.
M1_HD inline AreaLightPdfResult one_sided_area_light_pdf(
    float distance_squared, float area, float light_cosine, int light_count) {
    if (!finite(distance_squared) || !(distance_squared > 0.0f) ||
        !finite(area) || !(area > 0.0f) || !finite(light_cosine) || light_count <= 0)
        return {};
    if (light_cosine <= 0.0f) return {0.0f, true};

    // Separate powers of two before forming the area-to-solid-angle Jacobian.
    // All mantissas are in [1/2,1), so their product and quotient remain normal
    // even when area*cosine*count itself is outside the float exponent range.
    // The final ldexpf is the only operation allowed to underflow/overflow;
    // either event is still rejected, never reclassified as zero support.
    int distance_exponent = 0, area_exponent = 0, cosine_exponent = 0, count_exponent = 0;
    const float distance_mantissa = frexpf(distance_squared, &distance_exponent);
    const float area_mantissa = frexpf(area, &area_exponent);
    const float cosine_mantissa = frexpf(light_cosine, &cosine_exponent);
    const float count_mantissa = frexpf(static_cast<float>(light_count), &count_exponent);
    const float mantissa = distance_mantissa /
        ((area_mantissa * cosine_mantissa) * count_mantissa);
    const int exponent = distance_exponent - area_exponent - cosine_exponent - count_exponent;
    const float pdf = ldexpf(mantissa, exponent);
    if (!finite(pdf) || !(pdf > 0.0f)) return {};
    return {pdf, true};
}

// direct_mode has already been validated by the host: 0=MIS, 1=BSDF, 2=NEE.
// Primary/delta hits do not compete with a continuous light sample. In either
// non-MIS mode the competing density is unused and need not be evaluated.
M1_HD inline bool light_hit_needs_pdf(int bounce, bool previous_delta,
                                    unsigned direct_mode) {
    return bounce > 0 && !previous_delta && direct_mode == 0u;
}

} // namespace m1
