#pragma once

// Adapted from mitsuba-renderer/rgb2spec rgb2spec.c, commit
// 721145dedf2491851bd46ab8fd165955cb38ddaf.
// Copyright (c) 2020, Wenzel Jakob. All rights reserved.
// BSD-3-Clause: retain third_party/rgb2spec/LICENSE with source/binary releases.
// Changes: host/device annotations, explicit range/error status, bounded M1
// table layout, stable evaluation, and exact neutral reflectance evaluation.

#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define M1_RGB2SPEC_HD __host__ __device__
#else
#define M1_RGB2SPEC_HD
#endif

namespace m1 {

struct Rgb2SpecTableView {
    const float* scale = nullptr;
    const float* data = nullptr;
    std::uint32_t resolution = 0;
};

M1_RGB2SPEC_HD inline bool rgb2spec_finite(float v) {
    return v == v && v >= -FLT_MAX && v <= FLT_MAX;
}

// Caller uploads the arrays from a validated RgbToSpectrum host object. This
// function preserves the official maximum-component tie rule and index order.
// Invalid input returns false rather than upstream's implicit RGB clamping.
M1_RGB2SPEC_HD inline bool rgb2spec_fetch_device(Rgb2SpecTableView table,
                                                float r, float g, float b,
                                                float out[3]) {
    if (!out || !table.scale || !table.data || table.resolution != 64) return false;
    const float rgb[3] = {r, g, b};
    for (int j = 0; j < 3; ++j)
        if (!rgb2spec_finite(rgb[j]) || rgb[j] < 0.0f || rgb[j] > 1.0f) return false;
    if (r == g && g == b) {
        out[0] = out[1] = 0.0f;
        out[2] = r <= 0.0f ? -8192.0f : r >= 1.0f ? 8192.0f :
                 (r - 0.5f) / sqrtf(r * (1.0f - r));
        return rgb2spec_finite(out[2]);
    }
    int i = 0;
    for (int j = 1; j < 3; ++j) if (rgb[j] >= rgb[i]) i = j;
    const std::uint32_t res = table.resolution;
    const float z = rgb[i];
    // Equivalent ratio order avoids overflow for positive subnormal RGB.
    // Normal inputs retain the exact operation order in the official runtime.
    const bool tiny = z < 63.0f * FLT_MIN;
    const float factor = tiny ? 0.0f : float(res - 1) / z;
    const float x = tiny ? (rgb[(i + 1) % 3] / z) * float(res - 1) : rgb[(i + 1) % 3] * factor;
    const float y = tiny ? (rgb[(i + 2) % 3] / z) * float(res - 1) : rgb[(i + 2) % 3] * factor;
    const std::uint32_t raw_xi = static_cast<std::uint32_t>(x);
    const std::uint32_t raw_yi = static_cast<std::uint32_t>(y);
    const std::uint32_t xi = raw_xi < res - 2 ? raw_xi : res - 2;
    const std::uint32_t yi = raw_yi < res - 2 ? raw_yi : res - 2;
    int left = 0, size = int(res) - 2;
    while (size > 0) {
        const int half = size >> 1;
        const int middle = left + half + 1;
        if (table.scale[middle] <= z) { left = middle; size -= half + 1; }
        else size = half;
    }
    const std::uint32_t zi = std::uint32_t(left) < res - 2 ? std::uint32_t(left) : res - 2;
    std::uint32_t offset = (((std::uint32_t(i) * res + zi) * res + yi) * res + xi) * 3;
    const std::uint32_t dx = 3, dy = 3 * res, dz = 3 * res * res;
    const float x1 = x - float(xi), x0 = 1.0f - x1;
    const float y1 = y - float(yi), y0 = 1.0f - y1;
    const float z1 = (z - table.scale[zi]) / (table.scale[zi + 1] - table.scale[zi]);
    const float z0 = 1.0f - z1;
    for (int j = 0; j < 3; ++j, ++offset) {
        out[j] = ((table.data[offset] * x0 + table.data[offset + dx] * x1) * y0 +
                  (table.data[offset + dy] * x0 + table.data[offset + dy + dx] * x1) * y1) * z0 +
                 ((table.data[offset + dz] * x0 + table.data[offset + dz + dx] * x1) * y0 +
                  (table.data[offset + dz + dy] * x0 + table.data[offset + dz + dy + dx] * x1) * y1) * z1;
        if (!rgb2spec_finite(out[j])) return false;
    }
    return true;
}

M1_RGB2SPEC_HD inline float rgb2spec_eval_device(const float coefficients[3], float wavelength_nm) {
    const float p = fmaf(fmaf(coefficients[0], wavelength_nm, coefficients[1]), wavelength_nm, coefficients[2]);
    return 0.5f * p / hypotf(p, 1.0f) + 0.5f;
}

// Apply to linear-sRGB texture values after RGB filtering. No coefficients or
// spectra are interpolated across texels. The nonlinear upsampler is applied
// once to that filtered color, which matches the declared texture policy.
M1_RGB2SPEC_HD inline bool rgb2spec_reflectance_device(Rgb2SpecTableView table,
                                                      float r, float g, float b,
                                                      float wavelength_nm, float& value) {
    if (!rgb2spec_finite(wavelength_nm) || wavelength_nm < 360.0f || wavelength_nm > 830.0f) return false;
    if (!table.scale || !table.data || table.resolution != 64) return false;
    if (!rgb2spec_finite(r) || !rgb2spec_finite(g) || !rgb2spec_finite(b) ||
        r < 0.0f || g < 0.0f || b < 0.0f || r > 1.0f || g > 1.0f || b > 1.0f) return false;
    if (r == g && g == b) { value = r; return true; }
    float c[3];
    if (!rgb2spec_fetch_device(table, r, g, b, c)) return false;
    value = rgb2spec_eval_device(c, wavelength_nm);
    return rgb2spec_finite(value) && value >= 0.0f && value <= 1.0f;
}

} // namespace m1

#undef M1_RGB2SPEC_HD
