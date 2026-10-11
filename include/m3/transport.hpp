#pragma once

#include "m1/bounded_trig.hpp"

#include "m1/fp32_numeric.hpp"

namespace m3 {

inline constexpr float transport_pi = m1::pi;

namespace transport_detail {
M1_HD inline float invalid() { return NAN; }
} // namespace transport_detail

// Scalar, unpolarized Rayleigh phase density per unit solid angle. The direct
// cosine API requires its physical domain; invalid inputs remain observable.
M1_HD inline float rayleigh_phase(float cosine) {
    if (!m1::finite(cosine) || cosine < -1.0f || cosine > 1.0f)
        return transport_detail::invalid();
    return (3.0f / (16.0f * transport_pi)) * fmaf(cosine, cosine, 1.0f);
}

// Marginal CDF in mu: azimuth has already been integrated over [0, 2*pi).
M1_HD inline float rayleigh_mu_cdf(float cosine) {
    if (!m1::finite(cosine) || cosine < -1.0f || cosine > 1.0f)
        return transport_detail::invalid();
    return fmaf(cosine, fmaf(cosine, cosine, 3.0f) * 0.125f, 0.5f);
}

M1_HD inline float sample_rayleigh_mu(float u) {
    if (!m1::finite(u) || u < 0.0f || u > 1.0f)
        return transport_detail::invalid();
    if (u == 0.0f) return -1.0f;
    if (u == 1.0f) return 1.0f;
    // mu^3 + 3*mu = 8*u - 4. This is the real, monotone inverse on [-1, 1].
    // One Newton correction removes most of the asinhf/sinhf rounding error.
    // The centered cubic avoids subtracting two CDFs close to one at the tail.
    const float centered = fmaf(4.0f, u, -2.0f);
    float cosine = 2.0f * sinhf(asinhf(centered) / 3.0f);
    const float residual = fmaf(cosine, fmaf(cosine, cosine, 3.0f), -2.0f * centered);
    cosine -= residual / (3.0f * fmaf(cosine, cosine, 1.0f));
    return cosine;
}

// OptiX ray directions need not have exactly unit norm. Scale and normalize
// each vector before its dot product, including finite tiny/large inputs.
// For finite nonzero vectors Cauchy-Schwarz guarantees [-1,1]; only roundoff
// can cross that range.
// This is not an angular support tolerance, a PDF floor, or an invalid-ray fix.
M1_HD inline float rayleigh_direction_cosine(m1::Vec3 a, m1::Vec3 b) {
    m1::Vec3 a_unit{}, b_unit{};
    if (!m1::fp32::unit(a, a_unit) || !m1::fp32::unit(b, b_unit))
        return transport_detail::invalid();
    return m1::clamp(m1::fp32::dot(a_unit, b_unit), -1.0f, 1.0f);
}

M1_HD inline float rayleigh_phase_directions(m1::Vec3 a, m1::Vec3 b) {
    return rayleigh_phase(rayleigh_direction_cosine(a, b));
}

struct RayleighDirectionSample {
    m1::Vec3 direction{};
    float value = 0.0f;
    float pdf = 0.0f;
    bool valid = false;
};

// previous_ray_direction and the returned direction follow the camera path,
// toward successive vertices. For scalar Rayleigh, reciprocity makes this
// backward kernel equal to the forward phase density. Do not assume the same
// convention for a future oriented, conditional neural scattering kernel.
// value and the actual proposal pdf remain distinct fields for that reason.
M1_HD inline RayleighDirectionSample sample_rayleigh(
    m1::Vec3 previous_ray_direction, float u_mu, float u_phi) {
    if (!m1::finite(previous_ray_direction) || !m1::finite(u_mu) || !m1::finite(u_phi) ||
        u_mu < 0.0f || u_mu > 1.0f || u_phi < 0.0f || u_phi >= 1.0f)
        return {};
    m1::Vec3 axis{};
    if (!m1::fp32::unit(previous_ray_direction, axis)) return {};
    const float cosine = sample_rayleigh_mu(u_mu);
    if (!m1::finite(cosine) || cosine < -1.0f || cosine > 1.0f)
        return {};
    const float sine = sqrtf((1.0f - cosine) * (1.0f + cosine));
    const float azimuth = (2.0f * transport_pi) * u_phi;
    const auto phi = m1::bounded_sincos(azimuth);
    const m1::Vec3 local{sine * phi.cosine, sine * phi.sine, cosine};
    const auto frame = m1::make_frame(axis);
    m1::Vec3 direction{};
    if (!m1::fp32::unit(frame.to_world(local), direction)) return {};
    const float density = rayleigh_phase_directions(previous_ray_direction, direction);
    if (!m1::finite(direction) || !m1::finite(density) || !(density > 0.0f)) return {};
    return {direction, density, density, true};
}

// Existing render modes: 0 = MIS, 1 = continuation sampling, 2 = NEE.
// Callers validate nonnegative finite PDFs and the mode. These weights use
// angular densities conditional on a reached scattering vertex. Free-flight
// survival and roulette are represented by their sampling/throughput factors;
// they must not be inserted a second time in an endpoint contribution.
M1_HD inline float sun_emission_weight(int bounce, bool previous_delta,
    unsigned direct_mode, float scattering_pdf, float sun_pdf) {
    if (bounce == 0 || previous_delta || direct_mode == 1u) return 1.0f;
    if (direct_mode == 2u) return 0.0f;
    return m1::power_heuristic(scattering_pdf, sun_pdf);
}

M1_HD inline float nee_weight(unsigned direct_mode, float light_pdf, float scattering_pdf) {
    if (direct_mode == 1u) return 0.0f;
    if (direct_mode == 2u) return 1.0f;
    return m1::power_heuristic(light_pdf, scattering_pdf);
}

M1_HD inline bool area_emission_needs_pdf(bool previous_medium, int bounce,
    bool previous_delta, unsigned direct_mode) {
    return !previous_medium && bounce > 0 && !previous_delta && direct_mode == 0u;
}

M1_HD inline float area_emission_weight(bool previous_medium, int bounce,
    bool previous_delta, unsigned direct_mode, float scattering_pdf, float light_pdf) {
    // M3 intentionally has no area-light NEE at a medium vertex. A phase path
    // that reaches an emitter is therefore the sole strategy, even in mode 2.
    if (previous_medium) return 1.0f;
    return sun_emission_weight(bounce, previous_delta, direct_mode, scattering_pdf, light_pdf);
}

} // namespace m3
