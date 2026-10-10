#pragma once

#include "m1/math.hpp"
#include <cstdint>

namespace m1 {

// Wavelengths are nm on the published source coordinates (no air/vacuum shift).
// A constant represents exact neutral reflectance.
struct SpectralColor {
    Vec3 coefficients{}; // quadratic, linear, constant in physical wavelength nm
    float constant_value = 0.5f;
    std::uint32_t use_constant = 1;
};

M1_HD inline float evaluate_color(SpectralColor c, float wavelength_nm) {
    if (c.use_constant) return c.constant_value;
    const float p = fmaf(fmaf(c.coefficients.x, wavelength_nm, c.coefficients.y), wavelength_nm, c.coefficients.z);
    return 0.5f * p / hypotf(p, 1.0f) + 0.5f;
}

struct CieSample { float wavelength_nm = 0, x = 0, y = 0, z = 0; };
struct SolarSample { float wavelength_nm = 0, irradiance_w_m2_nm = 0; };
struct IlluminantSample { float wavelength_nm = 0, value = 0; };

} // namespace m1
