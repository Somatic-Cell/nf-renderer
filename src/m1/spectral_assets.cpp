#include "m1/spectral_assets.hpp"

#include "rgb2spec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace m1 {
namespace {

[[noreturn]] void invalid_file(const std::filesystem::path& p, const std::string& why) {
    throw std::runtime_error("Spectral asset '" + p.string() + "': " + why);
}

template <std::size_t N>
std::vector<std::array<double, N>> read_csv(const std::filesystem::path& path,
                                          const char* expected_header) {
    std::ifstream in(path);
    if (!in) invalid_file(path, "cannot open; run tools/prepare_spectral_assets.py");
    in.imbue(std::locale::classic());
    std::vector<std::array<double, N>> rows;
    std::string line;
    std::size_t line_number = 0;
    bool header_seen = false;
    while (std::getline(in, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        if (!header_seen) {
            if (line != expected_header) invalid_file(path, "unsupported CSV header");
            header_seen = true;
            continue;
        }
        std::array<double, N> row{};
        std::istringstream fields(line);
        fields.imbue(std::locale::classic());
        std::string field;
        for (std::size_t c = 0; c < N; ++c) {
            if (!std::getline(fields, field, ','))
                invalid_file(path, "missing column at line " + std::to_string(line_number));
            std::istringstream number(field);
            number.imbue(std::locale::classic());
            if (!(number >> row[c]) || !std::isfinite(row[c]) || row[c] < 0.0 ||
                row[c] > std::numeric_limits<float>::max())
                invalid_file(path, "invalid nonnegative finite value at line " + std::to_string(line_number));
            number >> std::ws;
            if (!number.eof()) invalid_file(path, "trailing text in numeric field");
        }
        if (std::getline(fields, field, ',') || (!line.empty() && line.back() == ','))
            invalid_file(path, "extra CSV column");
        if (!rows.empty() && !(row[0] > rows.back()[0]))
            invalid_file(path, "wavelengths must be strictly increasing");
        rows.push_back(row);
        if (rows.size() > 100000) invalid_file(path, "unexpected table size");
    }
    if (!in.eof()) invalid_file(path, "I/O error");
    if (!header_seen || rows.size() < 2) invalid_file(path, "empty or incomplete table");
    return rows;
}

template <std::size_t N>
void require_grid(const std::filesystem::path& path,
                  const std::vector<std::array<double, N>>& rows,
                  std::size_t count, double step) {
    if (rows.size() != count) invalid_file(path, "unexpected sample count");
    for (std::size_t i = 0; i < count; ++i) {
        if (std::abs(rows[i][0] - (360.0 + step * double(i))) > 1e-7)
            invalid_file(path, "expected complete 360..830 nm grid");
    }
}

void validate_rgb(const std::array<float, 3>& rgb) {
    for (float x : rgb)
        if (!std::isfinite(x) || x < 0.0f || x > 1.0f)
            throw std::invalid_argument("RGB2Spec reflectance requires finite linear sRGB in [0,1]");
}

// Upstream's binary is a native-endian format without a gamut identifier.
// Inspect the entire small, fixed M1 format before calling its allocation code.
// The appended RGB-to-XYZ matrix identifies the supported sRGB gamut. Checksums
// and the exact upstream revision are verified by the preparation script.
void validate_model(const std::filesystem::path& path) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "RGB2Spec needs IEEE-754 binary32");
    const std::uint32_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1)
        invalid_file(path, "M1 coefficient package requires a little-endian host");
    constexpr std::size_t res = 64;
    constexpr std::size_t nfine = 288;
    constexpr std::size_t forward_offset = 8 + 4 * res + 36 * res * res * res;
    constexpr std::size_t total_size = forward_offset + 4 + 4 * (4 * nfine + 12);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) invalid_file(path, "cannot open RGB2Spec model");
    if (in.tellg() != std::streamoff(total_size))
        invalid_file(path, "unsupported RGB2Spec model size/format; expected M1 sRGB resolution 64");
    in.seekg(0);
    std::vector<unsigned char> bytes(total_size);
    if (!in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
        invalid_file(path, "truncated RGB2Spec model");
    auto u32 = [&](std::size_t pos) {
        std::uint32_t value;
        std::memcpy(&value, bytes.data() + pos, sizeof(value));
        return value;
    };
    auto f32 = [&](std::size_t pos) {
        float value;
        std::memcpy(&value, bytes.data() + pos, sizeof(value));
        return value;
    };
    if (std::memcmp(bytes.data(), "SPEC", 4) || u32(4) != res || u32(forward_offset) != nfine)
        invalid_file(path, "unsupported RGB2Spec model header or forward block");
    for (std::size_t pos = 8; pos < total_size; pos += 4) {
        if (pos != forward_offset && !std::isfinite(f32(pos)))
            invalid_file(path, "non-finite RGB2Spec coefficient");
    }
    if (f32(8) != 0.0f || f32(8 + 4 * (res - 1)) != 1.0f)
        invalid_file(path, "invalid RGB2Spec scale endpoints");
    for (std::size_t i = 1; i < res; ++i)
        if (!(f32(8 + 4 * i) > f32(8 + 4 * (i - 1))))
            invalid_file(path, "RGB2Spec scales must be strictly increasing");
    // Official details/cie1931.h sRGB/D65 transform, serialized by the pinned
    // optimizer. Other gamut models have the same magic and must be rejected.
    constexpr float srgb_matrix[9] = {
        .412453f, .357580f, .180423f,
        .212671f, .715160f, .072169f,
        .019334f, .119193f, .950227f
    };
    const std::size_t matrix_offset = forward_offset + 4 + 16 * nfine;
    for (std::size_t i = 0; i < 9; ++i)
        if (std::abs(f32(matrix_offset + 4 * i) - srgb_matrix[i]) > 1e-7f)
            invalid_file(path, "model gamut is not the supported linear sRGB/D65");
}

} // namespace

SpectralAssets load_spectral_assets(const std::filesystem::path& directory) {
    const auto cie_path = directory / "cie_1931_2deg.csv";
    const auto solar_path = directory / "solar_am0.csv";
    const auto d65_path = directory / "d65_normalized.csv";
    const auto cie_rows = read_csv<4>(cie_path, "wavelength_nm,x,y,z");
    const auto solar_rows = read_csv<2>(solar_path, "wavelength_nm,irradiance_w_m2_nm");
    const auto d65_rows = read_csv<2>(d65_path, "wavelength_nm,value");
    require_grid(cie_path, cie_rows, 471, 1.0);
    require_grid(solar_path, solar_rows, 4701, 0.1);
    require_grid(d65_path, d65_rows, 471, 1.0);
    SpectralAssets result;
    result.cie.reserve(cie_rows.size());
    for (const auto& v : cie_rows)
        result.cie.push_back({float(v[0]), float(v[1]), float(v[2]), float(v[3])});
    result.solar.reserve(solar_rows.size());
    for (const auto& v : solar_rows) {
        if (!(v[1] > 0.0)) invalid_file(solar_path, "visible solar irradiance must be positive");
        result.solar.push_back({float(v[0]), float(v[1])});
    }
    result.d65.reserve(d65_rows.size());
    double integral_y = 0.0, integral_d65_y = 0.0;
    for (std::size_t i = 0; i < d65_rows.size(); ++i) {
        const auto& v = d65_rows[i];
        if (!(v[1] > 0.0)) invalid_file(d65_path, "D65 must be positive over the visible interval");
        result.d65.push_back({float(v[0]), float(v[1])});
        const double weight = (i == 0 || i + 1 == d65_rows.size()) ? 0.5 : 1.0;
        integral_y += weight * cie_rows[i][2];
        integral_d65_y += weight * cie_rows[i][2] * v[1];
    }
    if (std::abs(integral_d65_y / integral_y - 1.0) > 1e-6)
        invalid_file(d65_path, "unexpected D65 normalization");
    return result;
}

struct RgbToSpectrum::Impl {
    RGB2Spec* model = nullptr;
    ~Impl() { if (model) rgb2spec_free(model); }
};

RgbToSpectrum::RgbToSpectrum(const std::filesystem::path& model_path)
    : impl_(std::make_unique<Impl>()) {
    validate_model(model_path);
    impl_->model = rgb2spec_load(model_path.string().c_str());
    if (!impl_->model) invalid_file(model_path, "official RGB2Spec loader failed");
}
RgbToSpectrum::~RgbToSpectrum() = default;
RgbToSpectrum::RgbToSpectrum(RgbToSpectrum&&) noexcept = default;
RgbToSpectrum& RgbToSpectrum::operator=(RgbToSpectrum&&) noexcept = default;

std::array<float, 3> RgbToSpectrum::reflectance_coefficients(std::array<float, 3> rgb) const {
    if (!impl_ || !impl_->model) throw std::logic_error("RGB2Spec adapter has been moved from");
    validate_rgb(rgb);
    std::array<float, 3> coeff{};
    rgb2spec_fetch(impl_->model, rgb.data(), coeff.data());
    for (float x : coeff)
        if (!std::isfinite(x)) throw std::runtime_error("RGB2Spec produced a non-finite coefficient");
    return coeff;
}

SpectralColor RgbToSpectrum::reflectance(std::array<float, 3> rgb) const {
    if (!impl_ || !impl_->model) throw std::logic_error("RGB2Spec adapter has been moved from");
    validate_rgb(rgb);
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) return {{}, rgb[0], 1};
    const auto coeff = reflectance_coefficients(rgb);
    return {{coeff[0], coeff[1], coeff[2]}, 0.0f, 0};
}

std::uint32_t RgbToSpectrum::resolution() const {
    if (!impl_ || !impl_->model) throw std::logic_error("RGB2Spec adapter has been moved from");
    return impl_->model->res;
}
const float* RgbToSpectrum::scale_data() const {
    (void) resolution();
    return impl_->model->scale;
}
const float* RgbToSpectrum::coefficients_data() const {
    (void) resolution();
    return impl_->model->data;
}
std::size_t RgbToSpectrum::scale_count() const { return resolution(); }
std::size_t RgbToSpectrum::coefficients_count() const {
    const std::size_t n = resolution();
    return 9 * n * n * n;
}

} // namespace m1
