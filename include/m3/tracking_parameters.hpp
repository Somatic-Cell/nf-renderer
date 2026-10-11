#pragma once

#include <cstdint>
#include <type_traits>

namespace m3 {

// GPU launch data for the scalar, pure-scattering Rayleigh atmosphere.
// Coordinates are metres with +Y up. The planet centre is
// (0, groundY - planetRadius, 0); wavelength is in nanometres.
// The numerical-reference parameters in parameters.hpp are CPU-only.
struct TrackingParameters {
    std::uint32_t enabled = 0;
    std::uint32_t areaNeeTransmittance = 1;
    float planetRadius = 6'371'000.0f;
    float atmosphereHeight = 100'000.0f;
    float scaleHeight = 8'000.0f;
    float groundY = -6.0f;
    float beta550 = 1.3e-5f;
    float groundAlbedo = 0.1f;
    // Unbiased Russian roulette of a small ratio-tracking product. Zero
    // disables roulette for comparisons. This is not a transmittance cutoff.
    float ratioRrThreshold = 0.01f;
    // Exhausting this diagnostic budget rejects the frame. It never accepts a
    // truncated flight or transmittance product as a valid estimator.
    std::uint32_t maxTrackingEvents = 1'048'576;
};

static_assert(std::is_standard_layout<TrackingParameters>::value,
              "Tracking parameters cross the host/device ABI verbatim");
static_assert(std::is_trivially_copyable<TrackingParameters>::value,
              "Tracking parameters must be uploadable as bytes");
static_assert(sizeof(TrackingParameters) == 40, "Unexpected FP32 launch layout");

} // namespace m3
