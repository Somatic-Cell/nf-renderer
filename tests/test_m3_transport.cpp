#include "m3/atmosphere.hpp"
#include "m3/transport.hpp"
#include "m1/bsdf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
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

// Independent host-only reference. The production phase header contains only
// FP32 arithmetic; these formulas must not call its float CDF or density.
constexpr double reference_pi = 3.141592653589793238462643383279502884;
constexpr double float_epsilon = std::numeric_limits<float>::epsilon();
double reference_phase(double mu) { return 3.0 * (1.0 + mu * mu) / (16.0 * reference_pi); }
double reference_cdf(double mu) { return (mu * mu * mu + 3.0 * mu + 4.0) / 8.0; }
double reference_quantile(double u) { return 2.0 * std::sinh(std::asinh(4.0 * u - 2.0) / 3.0); }
double reference_cosine(m1::Vec3 a, m1::Vec3 b) {
    const double aa = double(a.x)*a.x + double(a.y)*a.y + double(a.z)*a.z;
    const double bb = double(b.x)*b.x + double(b.y)*b.y + double(b.z)*b.z;
    return (double(a.x)*b.x + double(a.y)*b.y + double(a.z)*b.z) / std::sqrt(aa*bb);
}

template<class Function>
double simpson(Function function, double begin, double end, unsigned count) {
    require(count > 0 && count % 2 == 0, "Simpson integration needs an even interval count");
    const double step = (end - begin) / count;
    double total = function(begin) + function(end);
    for (unsigned i = 1; i < count; ++i)
        total += (i % 2 ? 4.0 : 2.0) * function(begin + step * i);
    return total * step / 3.0;
}

void test_rayleigh_scalar_distribution() {
    const auto density = [](double mu) { return 2.0 * reference_pi * m3::rayleigh_phase(float(mu)); };
    close(simpson(density, -1, 1, 128), 1.0, float_epsilon, "FP32 Rayleigh solid-angle normalization");
    close(simpson([&](double mu) { return mu * density(mu); }, -1, 1, 128),
          0.0, 2e-15, "Rayleigh first angular moment");
    close(simpson([&](double mu) { return mu * mu * density(mu); }, -1, 1, 4096),
          0.4, float_epsilon, "FP32 Rayleigh second angular moment is 2/5");

    constexpr unsigned count = 16384;
    double first = 0, second = 0, maximum_cdf_error = 0, maximum_quantile_error = 0;
    double maximum_phase_relative_error = 0, maximum_cdf_evaluation_error = 0;
    double previous = -1;
    for (unsigned i = 0; i < count; ++i) {
        const float u = (float(i) + 0.5f) / float(count);
        const float mu = m3::sample_rayleigh_mu(u);
        require(mu > previous && mu > -1 && mu < 1, "Rayleigh inverse CDF is strictly ordered");
        previous = mu;
        maximum_cdf_error = std::max(maximum_cdf_error, std::abs(reference_cdf(mu) - u));
        maximum_quantile_error = std::max(maximum_quantile_error, std::abs(mu - reference_quantile(u)));
        maximum_phase_relative_error = std::max(maximum_phase_relative_error,
            std::abs(m3::rayleigh_phase(mu) / reference_phase(mu) - 1.0));
        maximum_cdf_evaluation_error = std::max(maximum_cdf_evaluation_error,
            std::abs(m3::rayleigh_mu_cdf(mu) - reference_cdf(mu)));
        first += mu / count;
        second += double(mu) * mu / count;
    }
    close(first, 0, 2e-15, "Rayleigh quantile mean is zero");
    close(second, 0.4, 1.5e-8, "Rayleigh FP32 quantile second moment");
    require(maximum_cdf_error <= float_epsilon, "Rayleigh quantiles agree with independent double CDF");
    require(maximum_quantile_error <= float_epsilon, "Rayleigh quantiles agree with independent double inverse");
    require(maximum_phase_relative_error <= 2.0 * float_epsilon, "Rayleigh phase relative FP32 error");
    require(maximum_cdf_evaluation_error <= float_epsilon, "Rayleigh CDF absolute FP32 error");
    close(m3::sample_rayleigh_mu(0), -1, 0, "Rayleigh lower endpoint");
    close(m3::sample_rayleigh_mu(1), 1, 0, "Rayleigh upper endpoint");
    std::cout << "Rayleigh quantiles: mean=" << first << " second_moment=" << second
              << " max_cdf_error=" << maximum_cdf_error
              << " max_quantile_error=" << maximum_quantile_error
              << " max_phase_relative_error=" << maximum_phase_relative_error
              << " max_cdf_evaluation_error=" << maximum_cdf_evaluation_error << '\n';

    // Exercise the actual 23-bit PCG probability lattice at its tails and
    // around the symmetric center, including adjacent representable floats.
    const std::array<float, 12> boundary_probabilities{{0.0f,
        std::numeric_limits<float>::denorm_min(), 0x1p-24f, 0x1p-23f,
        std::nextafter(0.5f, 0.0f), 0.5f, std::nextafter(0.5f, 1.0f),
        0.75f, 1.0f-0x1p-23f, 1.0f-0x1p-24f, std::nextafter(1.0f, 0.0f), 1.0f}};
    for (float u : boundary_probabilities) {
        const float mu = m3::sample_rayleigh_mu(u);
        require(std::isfinite(mu) && mu >= -1.0f && mu <= 1.0f,
                "tail quantile remains in the physical interval");
        close(mu, reference_quantile(u), float_epsilon, "boundary quantile agrees with double reference");
    }
}

void test_rayleigh_direction_sampling() {
    const std::array<m1::Vec3, 9> axes{{{0, 1, 0}, {0, 0, 1}, {0, 0, -1},
        {1, 2, 3}, {1e30f, -2e30f, 3e30f}, {1e-30f, -2e-30f, 3e-30f},
        {0.70710677f, 0.70710677f, 0},
        {std::numeric_limits<float>::denorm_min(), 0, 0},
        {(std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)(), 0}}};
    double maximum_cdf_error = 0, maximum_norm_error = 0, maximum_cosine_error = 0;
    for (const auto axis : axes) {
        close(m3::rayleigh_direction_cosine(axis, axis), 1, 3.0 * float_epsilon, "same-direction FP32 cosine");
        close(m3::rayleigh_direction_cosine(axis, -axis), -1, 3.0 * float_epsilon, "opposite-direction FP32 cosine");
        for (unsigned i = 0; i < 256; ++i) {
            const float u = (float(i) + 0.5f) / 256.0f;
            const float v = (float((i * 73) % 256) + 0.5f) / 256.0f;
            const auto sampled = m3::sample_rayleigh(axis, u, v);
            require(sampled.valid && sampled.value > 0 && sampled.pdf > 0,
                    "Rayleigh direction sample is valid");
            const double cosine = reference_cosine(axis, sampled.direction);
            maximum_cdf_error = std::max(maximum_cdf_error, std::abs(reference_cdf(cosine) - u));
            maximum_cosine_error = std::max(maximum_cosine_error,
                std::abs(m3::rayleigh_direction_cosine(axis, sampled.direction) - cosine));
            maximum_norm_error = std::max(maximum_norm_error,
                std::abs(double(m1::dot(sampled.direction, sampled.direction)) - 1.0));
            close(sampled.value, m3::rayleigh_phase_directions(axis, sampled.direction), 0,
                  "returned phase is evaluated at the final float direction");
            close(sampled.pdf, sampled.value, 0, "Rayleigh samples its normalized phase");
            close(sampled.pdf / reference_phase(cosine), 1.0, 4.0 * float_epsilon,
                  "final direction PDF agrees with independently normalized double geometry");
        }
    }
    require(maximum_cdf_error < 2e-7, "float direction retains the sampled Rayleigh quantile");
    require(maximum_norm_error < 5e-7, "Rayleigh output directions remain unit length");
    require(maximum_cosine_error <= 3.0 * float_epsilon, "scaled FP32 cosine error against double geometry");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (const auto invalid : std::array<m1::Vec3, 3>{{{}, {nan, 0, 0}, {infinity, 0, 0}}}) {
        require(!std::isfinite(m3::rayleigh_direction_cosine(invalid, {0, 1, 0})),
                "invalid ray direction remains observable");
        require(!m3::sample_rayleigh(invalid, 0.5f, 0.5f).valid,
                "invalid ray direction is not silently replaced");
    }
    for (const float invalid : std::array<float, 4>{{-1.001f, 1.001f, nan, infinity}}) {
        require(!std::isfinite(m3::rayleigh_phase(invalid)), "phase rejects an invalid cosine");
        require(!std::isfinite(m3::rayleigh_mu_cdf(invalid)), "CDF rejects an invalid cosine");
    }
    for (const float invalid : std::array<float, 4>{{-0.01f, 1.01f, nan, infinity}})
        require(!std::isfinite(m3::sample_rayleigh_mu(invalid)), "inverse CDF rejects invalid probability");
    require(!m3::sample_rayleigh({0, 1, 0}, 0.5f, 1.0f).valid,
            "azimuth random input follows the documented half-open interval");
    std::cout << "Rayleigh float directions: max_cdf_error=" << maximum_cdf_error
              << " max_norm_error=" << maximum_norm_error
              << " max_cosine_error=" << maximum_cosine_error << '\n';
}

void test_strategy_support() {
    for (unsigned mode = 0; mode < 3; ++mode) {
        for (const int bounce : std::array<int, 3>{{0, 1, 7}}) {
            for (const bool delta : std::array<bool, 2>{{false, true}}) {
                require(!m3::area_emission_needs_pdf(true, bounce, delta, mode),
                        "medium area hits have no competing light strategy");
                close(m3::area_emission_weight(true, bounce, delta, mode, 0.1f, 12.0f),
                      1.0, 0, "medium area hits retain weight one in every mode");
                require(m3::area_emission_needs_pdf(false, bounce, delta, mode) ==
                        (bounce > 0 && !delta && mode == 0), "surface area PDF is evaluated only when needed");
                if (bounce == 0 || delta) {
                    close(m3::sun_emission_weight(bounce, delta, mode, 0, 12), 1, 0,
                          "camera and delta solar endpoints are retained");
                    close(m3::area_emission_weight(false, bounce, delta, mode, 0, 12), 1, 0,
                          "camera and delta area endpoints are retained");
                }
            }
        }
        for (const float scattering_pdf : std::array<float, 4>{{0.001f, 0.1f, 1, 1000}})
            for (const float light_pdf : std::array<float, 4>{{0.001f, 0.1f, 1, 1000}}) {
                const float a = m3::sun_emission_weight(1, false, mode, scattering_pdf, light_pdf);
                const float b = m3::nee_weight(mode, light_pdf, scattering_pdf);
                close(double(a) + b, 1, 1e-7, "solar continuation and NEE weights partition unity");
                close(m3::area_emission_weight(false, 1, false, mode, scattering_pdf, light_pdf),
                      a, 0, "surface endpoints use the same angular MIS convention");
            }
    }
    close(m3::area_emission_weight(false, 1, false, 0, 0.1f, 0), 1, 0,
          "zero competing area support is valid, without a PDF floor");
    close(m3::nee_weight(0, 12, 0), 1, 0, "light-only angular support has weight one");
}

m3::AtmosphereParameters reference_atmosphere() {
    m3::AtmosphereParameters parameters;
    parameters.enabled = 1;
    parameters.atmosphereHeight = 16000;
    parameters.scaleHeight = 8000;
    return parameters;
}

void test_vertical_ballistic_transport() {
    const auto parameters = reference_atmosphere();
    constexpr double height = 250, distance = 6000, target_tau = 0.8;
    const double column = std::exp(-height / parameters.scaleHeight) * parameters.scaleHeight *
                          (-std::expm1(-distance / parameters.scaleHeight));
    const double beta = target_tau / column;
    const m3::RaySegment segment{{0, parameters.groundY + height, 0}, {0, 1, 0}, distance};
    const auto attenuation = m3::transmittance(parameters, segment, beta);
    require(attenuation.status == m3::Status::Ok && !attenuation.ground_occluded,
            "vertical reference transmittance is valid and unobstructed");
    close(attenuation.tau, target_tau, 2e-9, "vertical exponential column has analytic optical depth");
    const double expected = std::exp(-target_tau);
    close(attenuation.transmittance, expected, 2e-9, "vertical Beer-Lambert transmittance");

    constexpr unsigned count = 1024;
    unsigned escaped = 0;
    double wrong_double_attenuation = 0, maximum_distance_error = 0, maximum_tau_residual = 0;
    for (unsigned i = 0; i < count; ++i) {
        const double xi = (i + 0.5) / count;
        const auto sample = m3::sample_free_flight(parameters, segment, beta, xi);
        require(sample.status == m3::Status::Ok, "vertical free-flight quantile converges");
        if (sample.scattered) {
            const double tau = -std::log1p(-xi);
            const double coefficient = beta * std::exp(-height / parameters.scaleHeight) * parameters.scaleHeight;
            const double exact_distance = -parameters.scaleHeight * std::log1p(-tau / coefficient);
            maximum_distance_error = std::max(maximum_distance_error, std::abs(sample.distance_m - exact_distance));
            maximum_tau_residual = std::max(maximum_tau_residual, std::abs(sample.tau_residual));
            close(sample.endpoint_mass, 0, 0, "a medium event has no endpoint mass");
            close(sample.transmittance, std::exp(-tau), 2e-9, "event transmittance follows the sampled optical depth");
            close(sample.pdf_per_m, sample.sigma_s_per_m * sample.transmittance, 2e-12,
                  "pure Rayleigh collision density is sigma_s times T");
        } else {
            // This is the zero-scattering endpoint estimator. A surviving path
            // contributes one, not T: its survival probability already is T.
            ++escaped;
            wrong_double_attenuation += sample.transmittance / count;
            close(sample.endpoint_mass, expected, 2e-9, "no-collision probability equals T");
        }
    }
    const double escaped_fraction = double(escaped) / count;
    close(escaped_fraction, expected, 1.0 / count, "ballistic endpoint estimator has expectation T");
    require(std::abs(wrong_double_attenuation - expected) > 0.2,
            "regression distinguishes an erroneous additional endpoint T factor");
    close(wrong_double_attenuation, expected * expected, 1.0 / count,
          "double attenuation gives T squared rather than the required T");
    require(maximum_distance_error < 5e-4, "vertical free-flight distance agrees with analytic inverse");

    // The same zero-scattering term for the sun uses an infinite geometric ray
    // clipped by the atmosphere; no surface endpoint is needed.
    const double top_distance = parameters.atmosphereHeight - height;
    const double sun_tau = beta * std::exp(-height / parameters.scaleHeight) * parameters.scaleHeight *
                           (-std::expm1(-top_distance / parameters.scaleHeight));
    const m3::RaySegment sun_ray{segment.origin, {0, 1, 0}, std::numeric_limits<double>::infinity()};
    const auto sun = m3::transmittance(parameters, sun_ray, beta);
    require(sun.status == m3::Status::Ok && !sun.ground_occluded, "vertical sun ray exits the atmosphere");
    close(sun.tau, sun_tau, 2e-9, "sun transmittance includes exactly the truncated atmosphere column");
    unsigned sun_escaped = 0;
    for (unsigned i = 0; i < count; ++i) {
        const auto sample = m3::sample_free_flight(parameters, sun_ray, beta, (i + 0.5) / count);
        require(sample.status == m3::Status::Ok, "solar free-flight quantile converges");
        if (!sample.scattered) ++sun_escaped;
    }
    close(double(sun_escaped) / count, std::exp(-sun_tau), 1.0 / count,
          "unscattered solar endpoint receives survival only once");
    std::cout << "Vertical transport: analytic_T=" << expected << " escaped_fraction=" << escaped_fraction
              << " erroneous_T_twice=" << wrong_double_attenuation
              << " max_distance_error_m=" << maximum_distance_error
              << " max_tau_residual=" << maximum_tau_residual
              << " sun_T=" << std::exp(-sun_tau)
              << " sun_escaped_fraction=" << double(sun_escaped) / count << '\n';
}

void test_solar_mis_expectations() {
    const auto parameters = reference_atmosphere();
    constexpr double height = 800, beta = 1.3e-5;
    const m3::DVec3 origin{0, parameters.groundY + height, 0};
    const auto transmittance = [&](m1::Vec3 direction) {
        const m3::RaySegment ray{origin, {direction.x, direction.y, direction.z},
                                 std::numeric_limits<double>::infinity()};
        const auto result = m3::transmittance(parameters, ray, beta);
        require(result.status == m3::Status::Ok && !result.ground_occluded,
                "solar MIS direction has valid atmosphere visibility");
        return result.transmittance;
    };
    const auto direction_at = [](double mu) {
        return m1::Vec3{float(std::sqrt((1 - mu) * (1 + mu))), float(mu), 0};
    };
    // A deliberately wide finite cap gives both strategies appreciable density
    // in this mathematical regression. It does not change the renderer's sun.
    constexpr double minimum_mu = 0.4;
    const float light_pdf = float(1.0 / (2.0 * reference_pi * (1 - minimum_mu)));
    constexpr unsigned count = 2048;
    const double surface_reference = simpson([&](double mu) {
        return 2.0 * mu * transmittance(direction_at(mu));
    }, minimum_mu, 1, count);
    const double medium_reference = simpson([&](double mu) {
        return 2.0 * reference_pi * reference_phase(mu) * transmittance(direction_at(mu));
    }, minimum_mu, 1, count);

    double maximum_error = 0;
    const m1::BsdfParameters diffuse{m1::MaterialKind::Diffuse, 1, 1, 1};
    for (unsigned mode = 0; mode < 3; ++mode) {
        double surface_nee = 0, surface_path = 0, medium_nee = 0, medium_path = 0;
        const double surface_cap_mass = 1 - minimum_mu * minimum_mu;
        const double phase_cdf_begin = reference_cdf(minimum_mu);
        const double medium_cap_mass = 1 - phase_cdf_begin;
        for (unsigned i = 0; i < count; ++i) {
            const double u = (i + 0.5) / count;
            const double mu = minimum_mu + (1 - minimum_mu) * u;
            const auto direction = direction_at(mu);
            const double attenuation = transmittance(direction);
            const float surface_pdf = float(mu / reference_pi);
            const float phase_pdf = m3::rayleigh_phase_directions({0, 1, 0}, direction);
            surface_nee += (mu / reference_pi) * attenuation / light_pdf *
                m3::nee_weight(mode, light_pdf, surface_pdf) / count;
            medium_nee += phase_pdf * attenuation / light_pdf *
                m3::nee_weight(mode, light_pdf, phase_pdf) / count;

            // Integrate continuation samples over the cap's known probability
            // mass. T here marginalizes the unsampled free-flight survival;
            // the preceding test explicitly samples it with endpoint weight 1.
            const auto surface = m1::sample_bsdf(diffuse, {0, 0, 1}, 0.5f,
                float(surface_cap_mass * u), 0.31f);
            require(surface.status == m1::BsdfSampleStatus::Valid, "Lambert continuation sample");
            const m1::Vec3 surface_world{surface.wi.x, surface.wi.z, surface.wi.y};
            surface_path += surface_cap_mass * surface.weight * transmittance(surface_world) *
                m3::sun_emission_weight(1, false, mode, surface.pdf, light_pdf) / count;

            const auto medium = m3::sample_rayleigh({0, 1, 0},
                float(phase_cdf_begin + medium_cap_mass * u), 0.31f);
            require(medium.valid, "Rayleigh continuation sample");
            medium_path += medium_cap_mass * double(medium.value / medium.pdf) * transmittance(medium.direction) *
                m3::sun_emission_weight(1, false, mode, medium.pdf, light_pdf) / count;
        }
        maximum_error = std::max(maximum_error, std::abs(surface_nee + surface_path - surface_reference));
        maximum_error = std::max(maximum_error, std::abs(medium_nee + medium_path - medium_reference));
        close(surface_nee + surface_path, surface_reference, 2e-6,
              "surface solar NEE plus attenuated continuation has the reference expectation");
        close(medium_nee + medium_path, medium_reference, 2e-6,
              "medium solar NEE plus phase continuation has the reference expectation");
    }
    std::cout << "Solar angular MIS: surface_reference=" << surface_reference
              << " medium_reference=" << medium_reference
              << " max_expected_value_error=" << maximum_error << '\n';
}
} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        test_rayleigh_scalar_distribution();
        test_rayleigh_direction_sampling();
        test_strategy_support();
        test_vertical_ballistic_transport();
        test_solar_mis_expectations();
        std::cout << "M3 transport checks passed: Rayleigh normalization/sampling, ballistic survival, "
                     "sun MIS expectation, and absent medium-to-area NEE support\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M3 transport check failed: " << error.what() << '\n';
        return 1;
    }
}
