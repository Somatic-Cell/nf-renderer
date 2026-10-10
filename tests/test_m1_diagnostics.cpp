#include "m1/failure_report.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void contains(const std::string& text, const std::string& needle, const char* message) {
    require(text.find(needle) != std::string::npos, message);
}
std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
float float_after(const std::string& text, const std::string& label) {
    const auto offset = text.find(label);
    require(offset != std::string::npos, "Missing numeric field");
    std::istringstream input(text.substr(offset + label.size()));
    input.imbue(std::locale::classic());
    float value = 0;
    require(bool(input >> value), "Numeric field does not parse in the classic locale");
    return value;
}

void handedness_cause_survives_formatting() {
    m1::FailureRecord record;
    record.reason = m1::FailureReason::TangentHandedness;
    record.stage = m1::FailureStage::Surface;
    record.context = {20, 17, 20, 110};
    record.frame = 0; record.pixel_x = 125; record.pixel_y = 72;
    record.sample = 0; record.bounce = 2; record.seed = 17;
    record.wavelength_nm = 550; record.throughput = .625f;
    record.radiance = 1.5f; record.bsdf_pdf = .25f; record.eta_scale = 1;
    record.current_position[0] = -38.375f;
    record.current_direction[2] = -1;
    // A legal barycentric midpoint with conflicting endpoint handedness. The
    // diagnosis depends on preserving zero, both signs, and the barycentrics.
    const float values[] = {0, 1, -1, 1, .5f, .5f, 0};
    for (unsigned i = 0; i < 7; ++i) record.values.v[i] = values[i];
    record.values.v[15] = -2.5f;
    const auto report = m1::format_failure(record);
    contains(report, "reason=TangentHandedness (", "Lost the specific failure cause");
    contains(report, "stage=surface (", "Lost the shader stage");
    contains(report, "zero-based frame/pixel/sample", "Ambiguous index convention");
    contains(report, "frame=0, pixel=(125, 72), sample=0, bounce=2, seed=17", "Changed the reproduction indices");
    contains(report, "mesh=20, material=17, instance=20, primitive=110", "Lost primitive identity");
    contains(report, "throughput=0.625, radiance=1.5, bsdf_pdf=0.25, eta_scale=1", "Lost transport state");
    contains(report, "position=(-38.375, 0, 0), direction=(0, 0, -1)", "Lost the current ray");
    contains(report, "[0] tangent.w=0", "Zero handedness was repaired or omitted");
    contains(report, "[2] w1=-1", "Negative handedness was lost");
    contains(report, "[4] b0=0.5", "Payload label/value mapping changed");
    contains(report, "[15] value[15]=-2.5", "Unlabeled payload values were discarded");
}

void invalid_values_and_missing_ids_survive_formatting() {
    m1::FailureRecord record;
    record.reason = m1::FailureReason::ReflectanceSpectrum;
    record.stage = m1::FailureStage::AreaNee;
    record.context.material = 0;
    record.bounce = -3;
    record.wavelength_nm = std::nextafter(550.f, 551.f);
    record.throughput = -0.0f;
    record.radiance = std::numeric_limits<float>::infinity();
    record.bsdf_pdf = -std::numeric_limits<float>::infinity();
    record.values.v[0] = -std::numeric_limits<float>::epsilon();
    record.values.v[1] = std::nextafter(1.f, 2.f);
    const std::uint32_t nanBits = 0x7fc12345u;
    std::memcpy(&record.values.v[2], &nanBits, sizeof(nanBits));
    record.values.v[3] = record.wavelength_nm;
    record.values.v[4] = -1;
    const auto report = m1::format_failure(record);
    contains(report, "mesh=none, material=0, instance=none, primitive=none", "Missing IDs were confused with index zero");
    contains(report, "bounce=-3", "A negative recorded bounce was silently changed");
    contains(report, "throughput=-0", "Signed zero was lost");
    contains(report, "radiance=inf[bits=0x7f800000]", "Positive infinity was lost");
    contains(report, "bsdf_pdf=-inf[bits=0xff800000]", "Negative infinity was lost");
    contains(report, "[2] rgb.b=nan[bits=0x7fc12345]", "NaN cause/payload was lost");
    require(bits(float_after(report, "wavelength_nm=")) == bits(record.wavelength_nm), "Wavelength lost binary32 precision");
    require(bits(float_after(report, "[0] rgb.r=")) == bits(record.values.v[0]), "Negative RGB lost binary32 precision");
    require(bits(float_after(report, "[1] rgb.g=")) == bits(record.values.v[1]), "Above-one RGB was clamped or rounded to one");
    contains(report, "[4] result=-1", "The failed API result was omitted");
}

void unknown_enums_and_locale_are_safe() {
    struct CommaDecimal : std::numpunct<char> {
        char do_decimal_point() const override { return ','; }
    };
    const auto previous = std::locale();
    std::locale::global(std::locale(previous, new CommaDecimal));
    m1::FailureRecord record;
    record.reason = static_cast<m1::FailureReason>(0xfffffffeu);
    record.stage = static_cast<m1::FailureStage>(123456u);
    record.seed = 0xffffffffu;
    record.wavelength_nm = 550.125f;
    record.values.v[0] = -7.25f;
    const auto report = m1::format_failure(record);
    std::locale::global(previous);
    contains(report, "reason=UnknownReason (4294967294)", "Unknown reason numeric ID was lost");
    contains(report, "stage=unknown (123456)", "Unknown stage numeric ID was lost");
    contains(report, "seed=4294967295", "A valid full-range seed was treated as missing");
    contains(report, "wavelength_nm=550.125", "Locale changed the diagnostic decimal separator");
    contains(report, "[0] value[0]=-7.25", "Unknown payload semantics were invented");
    contains(report, "[15] value[15]=0", "Unknown reason did not retain all 16 slots");
    m1::FailureRecord empty;
    contains(m1::format_failure(empty), "reason=None (0)", "Empty record was disguised as a real reason");
}
} // namespace

int main() {
    try {
        handedness_cause_survives_formatting();
        invalid_values_and_missing_ids_survive_formatting();
        unknown_enums_and_locale_are_safe();
        std::cout << "M1/M2 failure report checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Diagnostic formatting test failed: " << error.what() << '\n';
        return 1;
    }
}
