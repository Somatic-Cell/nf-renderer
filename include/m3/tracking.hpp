#pragma once

#include "m3/tracking_geometry.hpp"
#include <cstdint>

namespace m3::tracking {

// CPU and CUDA execute the same FP32 estimator. RNG::operator() must return a
// fresh uniform float strictly inside (0,1). The caller's stream is advanced;
// null collisions are not vertices and never evaluate a phase or trace a mesh.
enum class Endpoint : std::uint32_t { None, SegmentEnd, AtmosphereExit, Ground };

struct FreeFlightResult {
    Status status = Status::Ok;
    bool scattered = false;
    Endpoint endpoint = Endpoint::None;
    FloatPair parameter{}; // Raw OptiX ray parameter, retaining small increments.
    float ray_t = 0.0f;    // Rounded diagnostic; do not reconstruct the point from it.
    std::uint32_t candidates = 0;
    std::uint32_t null_collisions = 0;
    ShellClip clip{};
};

struct RatioResult {
    Status status = Status::Ok;
    float transmittance = 1.0f;
    bool ground_occluded = false;
    std::uint32_t candidates = 0;
    std::uint32_t roulette_tests = 0;
    bool roulette_terminated = false;
    ShellClip clip{};
};

M1_HD inline Status validate_parameters(const TrackingParameters& p) {
    const Status geometry = validate_geometry_parameters(p);
    if (geometry != Status::Ok) return geometry;
    if (p.enabled > 1u || p.areaNeeTransmittance > 1u ||
        !m1::finite(p.scaleHeight) || !(p.scaleHeight > 0.0f) ||
        !m1::finite(p.beta550) || p.beta550 < 0.0f ||
        !m1::finite(p.groundAlbedo) || p.groundAlbedo < 0.0f || p.groundAlbedo > 1.0f ||
        !m1::finite(p.ratioRrThreshold) || p.ratioRrThreshold < 0.0f || p.ratioRrThreshold > 1.0f ||
        p.maxTrackingEvents == 0u)
        return Status::InvalidParameters;
    return Status::Ok;
}

// The global majorant is the wavelength's sea-level coefficient itself.
// One wavelength per path avoids a majorant chosen for a different wavelength.
M1_HD inline bool rayleigh_coefficient(const TrackingParameters& p,
                                      float wavelength_nm, float& beta_per_m) {
    beta_per_m = 0.0f;
    if (!m1::finite(p.beta550) || p.beta550 < 0.0f ||
        !m1::finite(wavelength_nm) || !(wavelength_nm > 0.0f)) return false;
    if (p.beta550 == 0.0f) return true;
    const float ratio = 550.0f / wavelength_nm;
    const float square = ratio * ratio;
    const float beta = (p.beta550 * square) * square;
    if (!m1::finite(beta) || !(beta > 0.0f)) return false;
    beta_per_m = beta;
    return true;
}

namespace tracking_detail {

template<class RNG>
M1_HD inline Status uniform_open(RNG& rng, float& u) {
    u = rng();
    return m1::finite(u) && u > 0.0f && u < 1.0f ? Status::Ok : Status::InvalidRandom;
}

M1_HD inline Status basic_ray(const RaySegment& ray) {
    if (!m1::finite(ray.origin) || !m1::finite(ray.direction) ||
        !(m1::max_component({::fabsf(ray.direction.x), ::fabsf(ray.direction.y),
                             ::fabsf(ray.direction.z)}) > 0.0f) ||
        ray.t_max != ray.t_max || ray.t_max < 0.0f)
        return Status::InvalidRay;
    return Status::Ok;
}

// Add an exponential step without rounding a long accumulated t to one float.
// An overflowing positive step is necessarily beyond the finite shell endpoint.
// A non-advancing positive increment is an explicit failure, never an epsilon
// displacement that would silently alter the sampling distribution.
template<class RNG>
M1_HD inline Status next_candidate(RNG& rng, float rate_per_t, FloatPair current,
    FloatPair end, FloatPair& next, bool& beyond) {
    float u = 0.0f;
    const Status random = uniform_open(rng, u);
    if (random != Status::Ok) return random;
    const float step = -::log1pf(-u) / rate_per_t;
    if (step != step || !(step > 0.0f)) return Status::TrackingRoundoff;
    if (!m1::finite(step)) { beyond = true; return Status::Ok; }
    next = add(current, step);
    if (!finite(next)) { beyond = true; return Status::Ok; }
    if (compare(next, current) <= 0) return Status::TrackingRoundoff;
    beyond = compare(next, end) >= 0;
    return Status::Ok;
}

// h is evaluated before converting the candidate point to the renderer's float
// position. No density clipping or probability clamp hides a majorant violation.
M1_HD inline Status height_exponent(const TrackingParameters& p,
    const RaySegment& ray, FloatPair parameter, float& exponent) {
    // Parameters and the ray were checked once at the sampler entry/clip.
    // Avoid repeating radius-domain validation at every null collision.
    const auto h = geometry_detail::height_unchecked(p, point_at(ray, parameter));
    if (h.status != Status::Ok) return h.status;
    if (compare(h.squared_ground_distance, 0.0f) < 0 || !(h.height_m >= 0.0f))
        return Status::InsideGround;
    exponent = -h.height_m / p.scaleHeight;
    if (exponent != exponent || exponent > 0.0f) return Status::MajorantViolation;
    return Status::Ok;
}

M1_HD inline void set_endpoint(const RaySegment& ray, FreeFlightResult& result) {
    if (result.clip.ground_occludes) {
        result.endpoint = Endpoint::Ground;
        result.parameter = result.clip.ground_parameter;
    } else if (m1::finite(ray.t_max)) {
        result.endpoint = Endpoint::SegmentEnd;
        result.parameter = {ray.t_max, 0.0f};
    } else {
        result.endpoint = Endpoint::AtmosphereExit;
        result.parameter = result.clip.has_medium ? result.clip.end_parameter : FloatPair{};
    }
    result.ray_t = value(result.parameter);
}

} // namespace tracking_detail

// Analog delta tracking: a real collision has density sigma_t(s)*T(s), while
// no collision before the endpoint has probability T(L). Pure Rayleigh has
// sigma_s/sigma_t == 1, so BOTH outcomes have distance-sampling weight one.
// This API deliberately does not fabricate an uncomputed tau, PDF or T(L).
template<class RNG>
M1_HD inline FreeFlightResult delta_free_flight(const TrackingParameters& p,
    const RaySegment& ray, float beta_per_m, RNG& rng) {
    FreeFlightResult result;
    result.status = validate_parameters(p);
    if (result.status != Status::Ok) return result;
    result.status = tracking_detail::basic_ray(ray);
    if (result.status != Status::Ok) return result;
    if (!m1::finite(beta_per_m) || beta_per_m < 0.0f) {
        result.status = Status::InvalidCoefficient; return result;
    }
    if (!p.enabled) {
        tracking_detail::set_endpoint(ray, result);
        return result;
    }
    result.clip = clip(p, ray);
    result.status = result.clip.status;
    if (result.status != Status::Ok) return result;
    tracking_detail::set_endpoint(ray, result);
    if (!result.clip.has_medium || beta_per_m == 0.0f) return result;
    const float rate = beta_per_m * result.clip.direction_length;
    if (!m1::finite(rate) || !(rate > 0.0f)) {
        result.status = Status::InvalidCoefficient; return result;
    }
    FloatPair current = result.clip.begin_parameter;
    for (;;) {
        FloatPair next{};
        bool beyond = false;
        result.status = tracking_detail::next_candidate(rng, rate, current,
            result.clip.end_parameter, next, beyond);
        if (result.status != Status::Ok || beyond) return result;
        if (result.candidates == p.maxTrackingEvents) {
            result.status = Status::TrackingLimit; return result;
        }
        ++result.candidates;
        float exponent = 0.0f;
        result.status = tracking_detail::height_exponent(p, ray, next, exponent);
        if (result.status != Status::Ok) return result;
        const float accept = ::expf(exponent);
        if (!m1::finite(accept) || accept < 0.0f || accept > 1.0f) {
            result.status = Status::MajorantViolation; return result;
        }
        float u = 0.0f;
        result.status = tracking_detail::uniform_open(rng, u);
        if (result.status != Status::Ok) return result;
        if (u < accept) {
            result.scattered = true;
            result.endpoint = Endpoint::None;
            result.parameter = next;
            result.ray_t = value(next);
            return result;
        }
        ++result.null_collisions;
        current = next;
    }
}

// Ratio tracking is used ONLY on explicit light connections. With roulette
// disabled, it is the product over a Poisson process of rate beta(lambda)*|d|.
// Optional roulette preserves its conditional expectation; a small value is
// never simply cut off. Opaque ground is a separate geometric visibility test.
template<class RNG>
M1_HD inline RatioResult ratio_transmittance(const TrackingParameters& p,
    const RaySegment& ray, float beta_per_m, RNG& rng) {
    RatioResult result;
    result.status = validate_parameters(p);
    if (result.status != Status::Ok) return result;
    result.status = tracking_detail::basic_ray(ray);
    if (result.status != Status::Ok) return result;
    if (!m1::finite(beta_per_m) || beta_per_m < 0.0f) {
        result.status = Status::InvalidCoefficient; return result;
    }
    if (!p.enabled) return result;
    result.clip = clip(p, ray);
    result.status = result.clip.status;
    if (result.status != Status::Ok) return result;
    result.ground_occluded = result.clip.ground_occludes;
    if (result.ground_occluded) { result.transmittance = 0.0f; return result; }
    if (!result.clip.has_medium || beta_per_m == 0.0f) return result;
    const float rate = beta_per_m * result.clip.direction_length;
    if (!m1::finite(rate) || !(rate > 0.0f)) {
        result.status = Status::InvalidCoefficient; return result;
    }
    FloatPair current = result.clip.begin_parameter;
    for (;;) {
        FloatPair next{};
        bool beyond = false;
        result.status = tracking_detail::next_candidate(rng, rate, current,
            result.clip.end_parameter, next, beyond);
        if (result.status != Status::Ok || beyond) return result;
        if (result.candidates == p.maxTrackingEvents) {
            result.status = Status::TrackingLimit; return result;
        }
        ++result.candidates;
        float exponent = 0.0f;
        result.status = tracking_detail::height_exponent(p, ray, next, exponent);
        if (result.status != Status::Ok) return result;
        // Stable even when sigma/M is close to one near the ground.
        const float factor = -::expm1f(exponent);
        if (!m1::finite(factor) || factor < 0.0f || factor > 1.0f) {
            result.status = Status::MajorantViolation; return result;
        }
        if (factor == 0.0f) { result.transmittance = 0.0f; return result; }
        const float threshold = p.ratioRrThreshold;
        if (threshold > 0.0f && factor < threshold / result.transmittance) {
            // Compute the survival probability before multiplying tiny values.
            const float survival = (result.transmittance / threshold) * factor;
            if (!m1::finite(survival) || !(survival > 0.0f) || survival > 1.0f) {
                result.status = Status::NumericalUnderflow; return result;
            }
            ++result.roulette_tests;
            float u = 0.0f;
            result.status = tracking_detail::uniform_open(rng, u);
            if (result.status != Status::Ok) return result;
            if (u >= survival) {
                result.transmittance = 0.0f;
                result.roulette_terminated = true;
                return result;
            }
            result.transmittance = threshold;
        } else {
            result.transmittance *= factor;
            if (!(result.transmittance > 0.0f)) {
                result.status = Status::NumericalUnderflow; return result;
            }
        }
        current = next;
    }
}

} // namespace m3::tracking
