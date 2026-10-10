#pragma once

#include <cstdint>
#include "spectral_types.hpp"

// Additional launch data for the M1 surface integrator.  The existing renderer,
// SBT, geometry/material buffers and two-word PRD payload ABI remain in use.
// Pointer members refer to allocations owned by Renderer (CUDABuffer).
struct M1RenderParams {
    uint32_t enabled = 0;
    uint32_t directMode = 0; // 0: power-heuristic MIS, 1: BSDF only, 2: NEE only
    uint32_t rrDepth = 5;
    uint32_t seed = 1;
    uint32_t sunEnabled = 1;
    float sunAngularRadiusRad = 0.004363323129985824f; // 0.25 degree
    float sunScale = 1.0f;
    float rayEpsilon = 1.0e-4f; // scene units; M1 fixtures use metres

    const m1::CieSample* cie = nullptr;
    uint32_t cieCount = 0;
    const m1::SolarSample* solar = nullptr;
    uint32_t solarCount = 0;
    const m1::IlluminantSample* d65 = nullptr;
    uint32_t d65Count = 0;
    uint32_t rgb2specResolution = 0;
    const float* rgb2specScale = nullptr;
    const float* rgb2specData = nullptr;

    // Counts rejected path samples and invalid film events; inspected by host.
    // Rejecting a non-finite sample is a reported failure, not a bias fix.
    uint32_t* invalidSamples = nullptr;
};
