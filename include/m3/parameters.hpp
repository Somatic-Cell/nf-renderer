#pragma once

#if defined(__CUDACC__)
#error "CPU numerical-reference parameters must not enter a GPU module. Use m3/tracking_parameters.hpp."
#endif

#include <cstdint>
#include <type_traits>

namespace m3 {

// CPU numerical-reference data (not the GPU launch ABI). Coordinates are
// metres, +Y is up, and
// the opaque planet centre is (0, groundY - planetRadius, 0). The atmosphere is
// truncated at planetRadius + atmosphereHeight. M3 has pure Rayleigh scattering
// with beta(lambda_nm) = beta550 * (550 / lambda_nm)^4 and no absorption.
struct AtmosphereParameters {
    std::uint32_t enabled = 0;
    // Disabling only area-light shadow transmittance is an explicit rendering
    // approximation. The physically consistent default includes it.
    std::uint32_t areaNeeTransmittance = 1;
    double planetRadius = 6'371'000.0;
    double atmosphereHeight = 100'000.0;
    double scaleHeight = 8'000.0;
    double groundY = -6.0;
    double beta550 = 1.3e-5; // Sea-level scattering/extinction coefficient [1/m].
    float groundAlbedo = 0.1f;

    // Optical-depth error targets are dimensionless. The quadrature returns an
    // error estimate, not a certified interval-arithmetic bound. An unresolved
    // integral/quantile is an explicit failure, never a vacuum fallback.
    double tauAbsoluteTolerance = 1e-10;
    double tauRelativeTolerance = 1e-8;
    double inverseTauTolerance = 1e-9;
    // A millimetre allows a roundoff-resolved inverse even for rare events near
    // the 100 km top, where the local density is extremely small. This is still
    // below the float OptiX position spacing there; it is an explicit tolerance.
    double distanceAbsoluteTolerance = 1e-3; // Metres.
    double distanceRelativeTolerance = 1e-10;
    std::uint32_t maxQuadratureDepth = 16;
    // Per optical-depth evaluation. Inversion has its own bounded iteration
    // count; its result also reports the total number of density evaluations.
    std::uint32_t maxQuadratureEvaluations = 4096;
    std::uint32_t maxInverseIterations = 64;
};

static_assert(std::is_standard_layout<AtmosphereParameters>::value,
              "Reference parameters have a stable CPU layout");
static_assert(std::is_trivially_copyable<AtmosphereParameters>::value,
              "Reference parameters are plain CPU data");

} // namespace m3
