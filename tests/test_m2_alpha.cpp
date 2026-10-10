#include "m2/alpha.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance, const char* message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

float coverage(float textureAlpha, float opacity, float cutoff) {
    float value = -2.0f;
    require(m2::surface_coverage(textureAlpha, opacity, cutoff, value),
            "valid alpha input must produce coverage");
    return value;
}

void test_mask_boundaries() {
    const float below = std::nextafter(0.5f, 0.0f);
    const float above = std::nextafter(0.5f, 1.0f);
    close(coverage(below, 1.0f, 0.5f), 0.0, 0.0, "mask rejects alpha below cutoff");
    close(coverage(0.5f, 1.0f, 0.5f), 1.0, 0.0, "mask accepts alpha equal to cutoff");
    close(coverage(above, 1.0f, 0.5f), 1.0, 0.0, "mask accepts alpha above cutoff");
    close(coverage(0.75f, 0.5f, 0.5f), 0.0, 0.0, "cutoff applies after opacity multiplication");
    close(coverage(1.0f, 0.5f, 0.5f), 1.0, 0.0, "combined alpha equal to cutoff is opaque");
    close(coverage(0.75f, 1.0f, 0.5f) * 8.0, 8.0, 0.0,
          "accepted mask endpoint emits full radiance, without raw alpha weighting");
    close(coverage(0.0f, 1.0f, -1.0f), 0.0, 0.0, "legacy zero coverage is transparent");
    close(coverage(1.0f, 1.0f, -1.0f), 1.0, 0.0, "legacy unit coverage is opaque");
}

void test_legacy_estimator_consistency() {
    // Two emissive patches with different proposal probabilities. Enumerating
    // independent, stratified direction/alpha variates gives exact acceptance
    // frequencies for these binary fractions, with no stochastic test tolerance.
    // This tests the endpoint estimator contract, not GPU intersection traversal.
    constexpr unsigned count = 64;
    constexpr std::array<float, 2> textureAlpha{0.25f, 0.75f};
    constexpr float opacity = 0.5f;
    constexpr std::array<double, 2> unmaskedContribution{2.0, 10.0};
    constexpr std::array<float, 2> bsdfPmf{0.75f, 0.25f};
    constexpr std::array<float, 2> lightPmf{0.25f, 0.75f};
    const std::array<float, 2> a{coverage(textureAlpha[0], opacity, -1.0f),
                                coverage(textureAlpha[1], opacity, -1.0f)};
    close(a[0], 0.125, 0.0, "legacy first patch coverage is texture alpha times opacity");
    close(a[1], 0.375, 0.0, "legacy second patch coverage is texture alpha times opacity");

    double bsdfOnly = 0.0, neeOnly = 0.0, mis = 0.0, doubleWeighted = 0.0;
    for (unsigned i = 0; i < count; ++i) {
        const float directionVariate = (float(i) + 0.5f) / float(count);
        const unsigned b = directionVariate < bsdfPmf[0] ? 0u : 1u;
        const unsigned l = directionVariate < lightPmf[0] ? 0u : 1u;
        for (unsigned j = 0; j < count; ++j) {
            const float alphaVariate = (float(j) + 0.5f) / float(count);
            const double hit = alphaVariate < a[b] ? unmaskedContribution[b] / bsdfPmf[b] : 0.0;
            const double nee = a[l] * unmaskedContribution[l] / lightPmf[l];
            bsdfOnly += hit;
            neeOnly += nee;
            mis += hit * m1::power_heuristic(bsdfPmf[b], lightPmf[b]) +
                   nee * m1::power_heuristic(lightPmf[l], bsdfPmf[l]);
            // Deliberate mutation: a second alpha factor after hit acceptance.
            // The analytic target below must distinguish this erroneous alpha^2.
            doubleWeighted += a[b] * hit;
        }
    }
    const double trials = double(count) * count;
    bsdfOnly /= trials;
    neeOnly /= trials;
    mis /= trials;
    doubleWeighted /= trials;
    constexpr double exact = 0.125 * 2.0 + 0.375 * 10.0;
    close(bsdfOnly, exact, 1.0e-12, "accepted hits need no second alpha multiplier");
    close(neeOnly, exact, 1.0e-12, "NEE endpoint weights emission by coverage once");
    close(mis, exact, 1.0e-6, "alpha hit and NEE estimates retain complementary MIS weights");
    close(doubleWeighted, 1.4375, 1.0e-12, "double-alpha mutation has the predicted wrong mean");
    require(std::abs(doubleWeighted - exact) > 1.0, "regression fixture must detect double weighting");
    std::cout << "legacy alpha: BSDF=" << bsdfOnly << " NEE=" << neeOnly
              << " MIS=" << mis << " exact=" << exact
              << "; double-alpha mutation=" << doubleWeighted << '\n';
}

void test_invalid_inputs() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (float invalid : std::array<float, 5>{-0.01f, 1.01f, nan, infinity, -infinity}) {
        float out = 0.0f;
        require(!m2::surface_coverage(invalid, 1.0f, -1.0f, out), "invalid texture alpha is rejected");
        require(!m2::surface_coverage(1.0f, invalid, -1.0f, out), "invalid opacity factor is rejected");
    }
    for (float invalid : std::array<float, 6>{-2.0f, -0.5f, 1.01f, nan, infinity, -infinity}) {
        float out = 0.0f;
        require(!m2::surface_coverage(1.0f, 1.0f, invalid, out), "invalid cutoff is rejected");
    }
}
} // namespace

int main() {
    try {
        test_mask_boundaries();
        test_legacy_estimator_consistency();
        test_invalid_inputs();
        std::cout << "M2 alpha: cutoff boundaries, opacity composition, endpoint/MIS consistency, "
                     "double-weighting sensitivity and invalid-input rejection passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M2 alpha check failed: " << error.what() << '\n';
        return 1;
    }
}
