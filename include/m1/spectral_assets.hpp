#pragma once

#include "m1/spectral_types.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace m1 {

// Table values retain their physical meaning: CIE 1931 2-degree color matching
// functions and AM0 irradiance on a plane normal to the solar center direction,
// in W m^-2 nm^-1. No white balance, Y normalization, or 683 factor is applied.
struct SpectralAssets {
    std::vector<CieSample> cie;
    std::vector<SolarSample> solar;
    // Relative CIE D65, normalized over the fixed 360..830 nm interval so that
    // trapz(D65*ybar)/trapz(ybar) == 1. The raw CIE source is also preserved.
    std::vector<IlluminantSample> d65;
};

// Loads the canonical CSV files written by tools/prepare_spectral_assets.py.
// The M1 package covers 360..830 nm inclusively. Missing, malformed, unordered,
// non-finite, negative, or incomplete data are errors; there is no fallback SPD.
SpectralAssets load_spectral_assets(const std::filesystem::path& asset_directory);

// Host-only adapter around the vendored, pinned official RGB2Spec runtime.
// This M1 adapter accepts its resolution-64 sRGB model with the forward-model
// block. It does not silently accept an older format or a different RGB gamut.
class RgbToSpectrum {
public:
    explicit RgbToSpectrum(const std::filesystem::path& model_path);
    ~RgbToSpectrum();
    RgbToSpectrum(RgbToSpectrum&&) noexcept;
    RgbToSpectrum& operator=(RgbToSpectrum&&) noexcept;
    RgbToSpectrum(const RgbToSpectrum&) = delete;
    RgbToSpectrum& operator=(const RgbToSpectrum&) = delete;

    // RGB is linear sRGB reflectance, not encoded sRGB and not emitted radiance.
    // Every component must be finite and in [0,1]; invalid values are rejected.
    // Coefficients are (quadratic, linear, constant), evaluated at wavelength nm.
    std::array<float, 3> reflectance_coefficients(std::array<float, 3> linear_srgb) const;

    // Exact neutral RGB uses SpectralColor's constant mode. This makes zero,
    // one, and neutral reflectances exact across all wavelengths, including the
    // extreme endpoints where finite polynomial coefficients are approximate.
    SpectralColor reflectance(std::array<float, 3> linear_srgb) const;

    // Borrowed host arrays for a single cudaMalloc/cudaMemcpy upload. Pointers
    // remain valid until this object is destroyed or moved from. Counts are in
    // floats: resolution and 9*resolution^3. The GPU does not need the optional
    // forward-model block, because it implements rgb2spec_fetch (not fetch_opt).
    std::uint32_t resolution() const;
    const float* scale_data() const;
    const float* coefficients_data() const;
    std::size_t scale_count() const;
    std::size_t coefficients_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace m1
