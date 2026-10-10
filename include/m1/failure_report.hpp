#pragma once

// Host-only formatting of the verbatim GPU diagnostic. This header has no CUDA
// dependencies and never repairs, clamps, or changes the recorded values.
#include "diagnostics.hpp"

#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace m1 {
namespace failure_report_detail {

inline std::string number(float value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (std::isfinite(value)) {
        // max_digits10 permits binary32 round trips, including small negative
        // values. A negative zero is intentionally not changed to zero.
        out << value;
    } else {
        if (std::signbit(value)) out << '-';
        out << (std::isnan(value) ? "nan" : "inf");
        // NaN payload/sign and the exact infinity bits remain inspectable;
        // stream implementations otherwise spell non-finite values differently.
        static_assert(sizeof(float) == sizeof(std::uint32_t), "Diagnostic float must be binary32-sized");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        out << "[bits=0x" << std::hex << std::setfill('0') << std::setw(8) << bits << ']';
    }
    return out.str();
}

inline std::string index(std::uint32_t value) {
    return value == no_index ? "none" : std::to_string(value);
}

inline std::vector<std::string> value_labels(FailureReason reason) {
    const std::string labels = failure_value_labels(reason);
    // Unknown/absent reasons have no known field semantics. Keep all payload
    // slots under their numeric indices instead of inventing parameter names.
    if (labels == "unknown" || reason == FailureReason::None) return {};
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin < labels.size()) {
        const auto comma = labels.find(',', begin);
        const auto end = comma == std::string::npos ? labels.size() : comma;
        std::size_t first = begin, last = end;
        while (first < last && std::isspace(static_cast<unsigned char>(labels[first]))) ++first;
        while (last > first && std::isspace(static_cast<unsigned char>(labels[last - 1]))) --last;
        result.push_back(labels.substr(first, last - first));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return result;
}

inline std::string vector3(const float value[3]) {
    return '(' + number(value[0]) + ", " + number(value[1]) + ", " + number(value[2]) + ')';
}

} // namespace failure_report_detail

inline std::string format_failure(const FailureRecord& record) {
    using failure_report_detail::index;
    using failure_report_detail::number;
    using failure_report_detail::vector3;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "reason=" << failure_reason_name(record.reason)
        << " (" << static_cast<std::uint32_t>(record.reason) << "), stage="
        << failure_stage_name(record.stage) << " (" << static_cast<std::uint32_t>(record.stage) << ")\n";
    out << "location (zero-based frame/pixel/sample): frame=" << record.frame
        << ", pixel=(" << record.pixel_x << ", " << record.pixel_y << ")"
        << ", sample=" << record.sample << ", bounce=" << record.bounce
        << ", seed=" << record.seed << '\n';
    out << "context (none=no_index): mesh=" << index(record.context.mesh)
        << ", material=" << index(record.context.material)
        << ", instance=" << index(record.context.instance)
        << ", primitive=" << index(record.context.primitive) << '\n';
    out << "state: wavelength_nm=" << number(record.wavelength_nm)
        << ", throughput=" << number(record.throughput)
        << ", radiance=" << number(record.radiance)
        << ", bsdf_pdf=" << number(record.bsdf_pdf)
        << ", eta_scale=" << number(record.eta_scale) << '\n';
    out << (record.stage == FailureStage::ShadowAnyHit ? "shadow_ray: origin=" : "path: current_position=")
        << vector3(record.current_position)
        << ", direction=" << vector3(record.current_direction) << '\n';
    const auto labels = failure_report_detail::value_labels(record.reason);
    out << "payload (all 16 slots; unlabeled slots retain their raw values):\n";
    for (std::size_t i = 0; i < 16; ++i) {
        const std::string label = i < labels.size() && !labels[i].empty()
            ? labels[i] : "value[" + std::to_string(i) + ']';
        out << "  [" << i << "] " << label << '=' << number(record.values.v[i]);
        if (i + 1 < 16) out << '\n';
    }
    return out.str();
}

} // namespace m1
