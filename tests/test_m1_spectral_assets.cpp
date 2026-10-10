#include "m1/spectral_assets.hpp"
#include "m1/rgb2spec_device.hpp"
#include "rgb2spec.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::abs(actual - expected) <= tolerance, message + ": " + std::to_string(actual));
}
void rejects(const std::function<void()>& operation, const std::string& message) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        path = std::filesystem::temp_directory_path() /
               ("m1-spectral-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(path);
    }
    ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

void test_real_tables(const std::filesystem::path& directory) {
    const auto assets = m1::load_spectral_assets(directory);
    require(assets.cie.size() == 471 && assets.solar.size() == 4701 && assets.d65.size() == 471, "real table sample counts");
    near(assets.cie.front().wavelength_nm, 360.0, 0, "CIE minimum wavelength");
    near(assets.cie.back().wavelength_nm, 830.0, 0, "CIE maximum wavelength");
    near(assets.cie.front().x, 0.0001299, 1e-11, "official CIE 360 nm X");
    near(assets.cie[80].z, 1.74706, 1e-7, "official CIE 440 nm Z");
    near(assets.cie[195].y, 1.0, 0, "official CIE 555 nm Y");
    double sum_y = 0;
    for (const auto& c : assets.cie) sum_y += c.y;
    // Independent validation value supplied by CIE's source metadata.
    near(sum_y, 106.8569171011719, 1e-6, "CIE metadata column sum");
    near(assets.solar.front().irradiance_w_m2_nm, 1.1270878767034793, 1e-7, "TSIS-1 HSRS 360 nm");
    near(assets.solar[1].irradiance_w_m2_nm, 1.1149154674796427, 1e-7, "TSIS-1 HSRS 360.1 nm");
    near(assets.solar.back().irradiance_w_m2_nm, 1.0560676282192256, 1e-7, "TSIS-1 HSRS 830 nm");
    double visible_irradiance = 0;
    for (std::size_t i = 1; i < assets.solar.size(); ++i) {
        const auto& a = assets.solar[i - 1];
        const auto& b = assets.solar[i];
        require(b.wavelength_nm > a.wavelength_nm, "float wavelength ordering");
        visible_irradiance += double(b.wavelength_nm - a.wavelength_nm) *
                              .5 * (double(a.irradiance_w_m2_nm) + double(b.irradiance_w_m2_nm));
    }
    // Source decimal values integrated by the preparation audit, in W/m^2.
    // This is only 360..830 nm, deliberately not the total solar irradiance.
    near(visible_irradiance, 739.3906237986226, 1e-4, "solar units / visible-band integral");
    double y_integral = 0, d65_y_integral = 0;
    for (std::size_t i = 0; i < assets.cie.size(); ++i) {
        const double weight = (i == 0 || i + 1 == assets.cie.size()) ? .5 : 1.0;
        y_integral += weight * assets.cie[i].y;
        d65_y_integral += weight * assets.cie[i].y * assets.d65[i].value;
    }
    near(d65_y_integral / y_integral, 1.0, 1e-7, "D65 weighted-mean normalization");
    TemporaryDirectory temporary;
    rejects([&] { (void)m1::load_spectral_assets(temporary.path); }, "missing data must fail");
    std::filesystem::copy_file(directory / "solar_am0.csv", temporary.path / "solar_am0.csv");
    std::filesystem::copy_file(directory / "d65_normalized.csv", temporary.path / "d65_normalized.csv");
    {
        std::ofstream broken(temporary.path / "cie_1931_2deg.csv");
        broken << "wavelength_nm,x,y,z\n360,1,1,1\n360,1,1,1\n";
    }
    rejects([&] { (void)m1::load_spectral_assets(temporary.path); }, "duplicate/incomplete CIE table must fail");
    {
        std::ofstream broken(temporary.path / "cie_1931_2deg.csv");
        broken << "wavelength_nm,x,y,z\n360,nan,1,1\n361,1,1,1\n";
    }
    rejects([&] { (void)m1::load_spectral_assets(temporary.path); }, "non-finite CIE table must fail");
}

void test_rgb2spec(const std::filesystem::path& directory) {
    const auto path = directory / "srgb.coeff";
    m1::RgbToSpectrum adapter(path);
    require(adapter.resolution() == 64 && adapter.scale_count() == 64 &&
            adapter.coefficients_count() == 2359296, "full resolution-64 model");
    m1::Rgb2SpecTableView view{adapter.scale_data(), adapter.coefficients_data(), adapter.resolution()};
    std::unique_ptr<RGB2Spec, decltype(&rgb2spec_free)> upstream(rgb2spec_load(path.string().c_str()), rgb2spec_free);
    require(upstream != nullptr, "official loader");
    std::vector<std::array<float, 3>> colors;
    for (float r : {0.0f, 0.25f, 0.5f, 0.999999f, 1.0f})
        for (float g : {0.0f, 0.25f, 0.5f, 0.999999f, 1.0f})
            for (float b : {0.0f, 0.25f, 0.5f, 0.999999f, 1.0f}) colors.push_back({r, g, b});
    colors.push_back({1e-10f, 5e-11f, 0.0f});
    std::mt19937 generator(1729);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < 10000; ++i) colors.push_back({uniform(generator), uniform(generator), uniform(generator)});
    double largest_evaluation_error = 0;
    for (auto rgb : colors) {
        float official[3];
        rgb2spec_fetch(upstream.get(), rgb.data(), official);
        const auto host = adapter.reflectance_coefficients(rgb);
        const auto color = adapter.reflectance(rgb);
        float common[3];
        require(m1::rgb2spec_fetch_device(view, rgb[0], rgb[1], rgb[2], common), "valid GPU common lookup");
        for (int j = 0; j < 3; ++j) {
            near(host[j], official[j], 0, "host wrapper must call the official implementation");
            near(common[j], official[j], 1e-6 + 2e-6 * std::abs(official[j]), "GPU/common lookup parity");
        }
        for (float wavelength : {360.0f, 412.5f, 500.0f, 555.0f, 701.25f, 830.0f}) {
            const float expected = rgb2spec_eval_precise(official, wavelength);
            const float polynomial = m1::rgb2spec_eval_device(common, wavelength);
            float reflected = -1;
            require(m1::rgb2spec_reflectance_device(view, rgb[0], rgb[1], rgb[2], wavelength, reflected),
                    "finite bounded reflectance");
            require(reflected >= 0.0f && reflected <= 1.0f, "reflectance bounds");
            const double error = std::abs(double(polynomial) - expected);
            largest_evaluation_error = std::max(largest_evaluation_error, error);
            // The common evaluator explicitly uses FMA and hypot; upstream
            // enables FMA conditionally. Allow rounding, not a different model.
            near(polynomial, expected, 2e-4, "GPU/common polynomial parity");
            near(m1::evaluate_color(color, wavelength), reflected, 2e-6, "host coefficient GPU evaluation");
        }
    }
    for (float neutral : {0.0f, 0.125f, 0.5f, 1.0f}) {
        const auto color = adapter.reflectance({neutral, neutral, neutral});
        require(color.use_constant == 1, "neutral uses constant representation");
        for (float wavelength : {360.0f, 555.0f, 830.0f}) {
            float value = -1;
            require(m1::rgb2spec_reflectance_device(view, neutral, neutral, neutral, wavelength, value), "neutral common evaluation");
            near(value, neutral, 0, "neutral exact common value");
            near(m1::evaluate_color(color, wavelength), neutral, 0, "neutral exact constant value");
        }
    }
    // Independent colorimetric check with the real 1 nm CIE/D65 tables. This
    // catches a wrong gamut, units, or D65 convention even if host and device
    // accidentally shared the same error. RGB->spectrum is metameric, so this
    // checks color reproduction, not a unique material spectrum.
    const auto assets = m1::load_spectral_assets(directory);
    for (auto rgb : {std::array<float, 3>{.1f, .2f, .3f}, {.8f, .2f, .1f}, {.1f, .7f, .3f}, {.5f, .5f, .5f}}) {
        const auto color = adapter.reflectance(rgb);
        double xyz[3]{}, normalizer = 0;
        for (std::size_t i = 0; i < assets.cie.size(); ++i) {
            const auto& c = assets.cie[i];
            const double weight = (i == 0 || i + 1 == assets.cie.size()) ? .5 : 1.0;
            const double spd = m1::evaluate_color(color, c.wavelength_nm) * double(assets.d65[i].value);
            xyz[0] += weight * spd * c.x;
            xyz[1] += weight * spd * c.y;
            xyz[2] += weight * spd * c.z;
            normalizer += weight * c.y;
        }
        for (double& x : xyz) x /= normalizer;
        const double rgb_matrix[3][3] = {{3.240479, -1.537150, -.498535},
                                         {-.969256, 1.875991, .041556},
                                         {.055648, -.204043, 1.057311}};
        for (int j = 0; j < 3; ++j)
            near(rgb_matrix[j][0] * xyz[0] + rgb_matrix[j][1] * xyz[1] + rgb_matrix[j][2] * xyz[2],
                 rgb[j], .008, "real CIE/D65 RGB reproduction");
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const auto bad : {std::array<float, 3>{-0.01f, 0, 0}, {1.01f, 0, 0}, {nan, 0, 0}, {inf, 0, 0}}) {
        rejects([&] { (void)adapter.reflectance(bad); }, "invalid host RGB must fail");
        float c[3], value = 1;
        require(!m1::rgb2spec_fetch_device(view, bad[0], bad[1], bad[2], c), "invalid common RGB must fail");
        require(!m1::rgb2spec_reflectance_device(view, bad[0], bad[1], bad[2], 555, value), "invalid reflectance RGB must fail");
    }
    float value = 0, c[3];
    require(!m1::rgb2spec_reflectance_device(view, .5f, .2f, .1f, 359, value), "out-of-range wavelength");
    require(!m1::rgb2spec_reflectance_device(view, .5f, .2f, .1f, nan, value), "nonfinite wavelength");
    require(!m1::rgb2spec_fetch_device({}, .5f, .2f, .1f, c), "missing model view");
    auto unsupported = view;
    unsupported.resolution = 32;
    require(!m1::rgb2spec_fetch_device(unsupported, .5f, .2f, .1f, c), "unsupported model resolution");
    // Positive subnormal input must not overflow (63 / max(RGB)). Upstream has
    // no guarantee here; this is an explicitly documented robustness fix.
    const float tiny = std::numeric_limits<float>::denorm_min();
    require(m1::rgb2spec_reflectance_device(view, tiny, 0, 0, 555, value), "subnormal RGB robustness");
    TemporaryDirectory temporary;
    const auto bad_model = temporary.path / "bad.coeff";
    {
        std::ofstream out(bad_model, std::ios::binary);
        out << "SPEC";
    }
    rejects([&] { m1::RgbToSpectrum invalid(bad_model); }, "truncated/old model rejected before allocation");
    std::filesystem::copy_file(path, bad_model, std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream out(bad_model, std::ios::binary | std::ios::in | std::ios::out);
        constexpr std::size_t matrix = 8 + 4 * 64 + 36 * 64 * 64 * 64 + 4 + 16 * 288;
        out.seekp(std::streamoff(matrix));
        const float wrong_gamut = 0.9f;
        out.write(reinterpret_cast<const char*>(&wrong_gamut), sizeof(wrong_gamut));
    }
    rejects([&] { m1::RgbToSpectrum invalid(bad_model); }, "same-size wrong-gamut model rejected");
    std::cout << "RGB2Spec official/common parity: " << colors.size()
              << " colors, max evaluation abs error " << largest_evaluation_error << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path directory = argc > 1 ? argv[1] : "spectrum/m1";
        test_real_tables(directory);
        test_rgb2spec(directory);
        std::cout << "M1 spectral asset tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
