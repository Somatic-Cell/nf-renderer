#include "m3/position.hpp"

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

double height(const m3::AtmosphereParameters& parameters, m3::DVec3 point) {
    double result = 0, radius = 0;
    require(m3::detail::height_at(parameters, point, result, radius) == m3::Status::Ok,
            "test point has a finite spherical height");
    return result;
}

void verify_representation(const m3::AtmosphereParameters& parameters,
                           m3::DVec3 exact, const m3::MediumPointResult& result) {
    require(result.status == m3::Status::Ok && m1::finite(result.position),
            "valid medium point has a finite float representation");
    require(height(parameters, {result.position.x, result.position.y, result.position.z}) >= 0,
            "represented medium point remains outside the opaque ground");
    const double source[3] = {exact.x, exact.y, exact.z};
    const float nearest[3] = {float(exact.x), float(exact.y), float(exact.z)};
    const float represented[3] = {result.position.x, result.position.y, result.position.z};
    const double relative[3] = {exact.x, parameters.planetRadius + (exact.y - parameters.groundY), exact.z};
    bool changed = false;
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (represented[axis] == nearest[axis]) continue;
        changed = true;
        const float outward = represented[axis] > nearest[axis] ?
            std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
        require(represented[axis] == std::nextafter(nearest[axis], outward),
                "only one adjacent float step is used for each changed coordinate");
        require(relative[axis] * (double(represented[axis]) - nearest[axis]) > 0,
                "changed coordinate moves toward the planetary exterior");
        require((relative[axis] > 0 && double(nearest[axis]) < source[axis]) ||
                (relative[axis] < 0 && double(nearest[axis]) > source[axis]),
                "coordinates already rounded outward are not moved");
        const double spacing = std::abs(double(represented[axis]) - nearest[axis]);
        require(std::abs(double(represented[axis]) - source[axis]) <= spacing,
                "directed representation differs from the double coordinate by at most one float step");
    }
    require(changed == result.rounded_outward, "outward rounding is reported explicitly");
    for (const auto direction : std::array<m3::DVec3, 6>{{{1,0,0},{-1,0,0},{0,1,0},
                                                        {0,-1,0},{0,0,1},{0,0,-1}}}) {
        const auto clip = m3::clip_atmosphere(parameters,
            {{result.position.x, result.position.y, result.position.z}, direction,
             std::numeric_limits<double>::infinity()});
        require(clip.status == m3::Status::Ok, "represented point can start another physical ray");
    }
}

void test_actual_freeflight_counterexample() {
    m3::AtmosphereParameters parameters;
    parameters.enabled = 1;
    // A valid 24-bit PCG probability cell, not an arbitrary value closer to an
    // endpoint than the renderer's RNG can produce. The local x position is
    // 250 km from the tangent point on an Earth-radius sphere.
    const m3::RaySegment ray{{250000, -4902.92822265625, 0}, {0, -1, 0},
                             std::numeric_limits<double>::infinity()};
    double beta = 0;
    require(m3::rayleigh_coefficient(parameters, 360, beta), "Rayleigh coefficient at 360 nm");
    const double xi = (11870.0 + 0.5) / 16777216.0;
    const auto sampled = m3::sample_free_flight(parameters, ray, beta, xi);
    require(sampled.status == m3::Status::Ok && sampled.scattered,
            "the real free-flight counterexample is a valid scattering event");
    const double norm = m3::length(ray.direction);
    const auto unit = ray.direction / norm;
    const m3::DVec3 exact{std::fma(unit.x, sampled.distance_m, ray.origin.x),
                         std::fma(unit.y, sampled.distance_m, ray.origin.y),
                         std::fma(unit.z, sampled.distance_m, ray.origin.z)};
    const m3::DVec3 nearest{double(float(exact.x)), double(float(exact.y)), double(float(exact.z))};
    const double exact_height = height(parameters, exact);
    const double nearest_height = height(parameters, nearest);
    require(exact_height > 0 && nearest_height < 0,
            "nearest float conversion reproduces an outside-to-inside domain change");
    const auto rejected = m3::clip_atmosphere(parameters, {nearest, {0, 1, 0}, 1});
    require(rejected.status == m3::Status::InsideGround, "uncorrected next ray is rejected");
    const auto represented = m3::represent_medium_point(parameters, exact);
    verify_representation(parameters, exact, represented);
    require(represented.rounded_outward, "counterexample needs directed representation");
    require(represented.position.x == float(exact.x) && represented.position.z == float(exact.z),
            "exactly representable coordinates remain unchanged");
    std::cout << "Medium point counterexample: exact_height_m=" << exact_height
              << " nearest_height_m=" << nearest_height
              << " represented_height_m=" << height(parameters,
                  {represented.position.x, represented.position.y, represented.position.z})
              << " y_step_m=" << double(represented.position.y) - nearest.y << '\n';
}

void test_spherical_sides_and_unchanged_points() {
    m3::AtmosphereParameters parameters;
    parameters.enabled = 1;
    unsigned corrected = 0, checked = 0;
    for (const double theta : std::array<double, 7>{{0, 0.25, 0.8, 1.4, 1.7, 2.6, 3.0}})
        for (const double phi : std::array<double, 5>{{0, 0.7, 1.6, 3.2, 4.8}}) {
            const double radius = parameters.planetRadius + 0.0001;
            const m3::DVec3 exact{radius * std::sin(theta) * std::cos(phi),
                parameters.groundY - parameters.planetRadius + radius * std::cos(theta),
                radius * std::sin(theta) * std::sin(phi)};
            require(height(parameters, exact) > 0, "constructed spherical point is outside");
            const auto result = m3::represent_medium_point(parameters, exact);
            verify_representation(parameters, exact, result);
            corrected += result.rounded_outward ? 1u : 0u;
            ++checked;
        }
    require(corrected > 0, "hemisphere cases exercise outward representation");

    for (const auto exact : std::array<m3::DVec3, 5>{{{0, parameters.groundY, 0},
        {0, parameters.groundY + 0.01, 0}, {100, 200, 30}, {100000, -750, 0},
        {0, parameters.groundY + parameters.atmosphereHeight + 100, 0}}}) {
        const auto result = m3::represent_medium_point(parameters, exact);
        verify_representation(parameters, exact, result);
        require(!result.rounded_outward && result.position.x == float(exact.x) &&
                result.position.y == float(exact.y) && result.position.z == float(exact.z),
                "nearest representations already outside the planet stay unchanged");
    }
    std::cout << "Spherical representation: checked=" << checked << " corrected=" << corrected << '\n';
}

void test_invalid_points_are_not_repaired() {
    m3::AtmosphereParameters parameters;
    parameters.enabled = 1;
    require(m3::represent_medium_point(parameters, {0, parameters.groundY - 0.0001, 0}).status ==
            m3::Status::InsideGround, "a truly inside point is never moved to the surface");
    bool checked_inside_with_outside_image = false;
    for (const double x : std::array<double, 7>{{1000, 10000, 50000, 100000, 150000, 500000, 1000000}}) {
        const double sag = x * x / (parameters.planetRadius +
            std::sqrt(parameters.planetRadius * parameters.planetRadius - x * x));
        const m3::DVec3 exact{x, parameters.groundY - sag - 1e-6, 0};
        require(height(parameters, exact) < 0, "adversarial point is truly inside the sphere");
        const double nearest_height = height(parameters,
            {double(float(exact.x)), double(float(exact.y)), 0});
        checked_inside_with_outside_image |= nearest_height >= 0;
        require(m3::represent_medium_point(parameters, exact).status == m3::Status::InsideGround,
                "inside double point is rejected even if its nearest float happens to be outside");
    }
    require(checked_inside_with_outside_image, "test includes an invalid point whose float image looks valid");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const auto invalid : std::array<m3::DVec3, 5>{{{nan, 0, 0}, {0, infinity, 0},
        {0, 0, -infinity}, {double(FLT_MAX) * 2, 0, 0}, {DBL_MAX, 0, 0}}})
        require(m3::represent_medium_point(parameters, invalid).status == m3::Status::NonfiniteGeometry,
                "nonfinite or unrepresentable positions are rejected");
    parameters.scaleHeight = 0;
    require(m3::represent_medium_point(parameters, {}).status == m3::Status::InvalidParameters,
            "invalid atmosphere parameters remain observable");
}
} // namespace

int main() {
    try {
        std::cout << std::setprecision(12);
        test_actual_freeflight_counterexample();
        test_spherical_sides_and_unchanged_points();
        test_invalid_points_are_not_repaired();
        std::cout << "M3 position checks passed: real PCG free-flight counterexample, sphere-domain "
                     "preservation, one-float-step bound, unchanged valid points, and invalid-input rejection\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M3 position check failed: " << error.what() << '\n';
        return 1;
    }
}
