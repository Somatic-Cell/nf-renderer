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

    // Evaluate the same area-to-solid-angle Jacobian with wider intermediate
    // range. Valid float inputs can underflow/overflow in area*cosine*count
    // even when their final quotient is representable as a positive float.
    const double value = double(distance_squared) /
        (double(area) * double(light_cosine) * double(light_count));
    if (!(value > 0.0) || value > double(FLT_MAX))
        return {};
    const float pdf = static_cast<float>(value);
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
