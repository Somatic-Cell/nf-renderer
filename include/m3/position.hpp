#pragma once

#include "m1/math.hpp"
#include "m3/atmosphere.hpp"
#include <cfloat>

namespace m3 {

struct MediumPointResult {
    m1::Vec3 position{};
    Status status = Status::NonfiniteGeometry;
    bool rounded_outward = false;
};

// OptiX positions are floats. A valid double-precision scattering point close
// to the opaque planet can round inside it, particularly far from the local
// scene origin. Preserve the spherical domain when choosing its float image.
// This does not move a truly invalid point out of the planet.
//
// Usually nearest rounding is returned unchanged. Only if that image is inside
// the ground, round the inward-rounded coordinates toward the planet's exterior
// by one adjacent float step. Coordinates already rounded outward stay intact.
// Thus each absolute planet-relative coordinate is at least its exact value,
// which preserves the sphere's exterior. This is a representational displacement
// of at most one float step per changed coordinate, with no metric epsilon.
M1_HD inline MediumPointResult represent_medium_point(
    const AtmosphereParameters& parameters, DVec3 exact) {
    MediumPointResult result;
    result.status = validate_parameters(parameters);
    if (result.status != Status::Ok) return result;
    if (!finite(exact)) {
        result.status = Status::NonfiniteGeometry;
        return result;
    }
    double exact_height = 0, radius = 0;
    result.status = detail::height_at(parameters, exact, exact_height, radius);
    if (result.status != Status::Ok) return result;
    if (exact_height < 0) {
        result.status = Status::InsideGround;
        return result;
    }
    // Reject values outside the float representation before performing a
    // narrowing conversion; the renderer cannot trace such a position.
    if (fabs(exact.x) > double(FLT_MAX) || fabs(exact.y) > double(FLT_MAX) ||
        fabs(exact.z) > double(FLT_MAX)) {
        result.status = Status::NonfiniteGeometry;
        return result;
    }
    result.position = {float(exact.x), float(exact.y), float(exact.z)};
    double rounded_height = 0;
    result.status = detail::height_at(parameters,
        {result.position.x, result.position.y, result.position.z}, rounded_height, radius);
    if (result.status != Status::Ok || rounded_height >= 0) return result;

    const double relative_y = parameters.planetRadius + (exact.y - parameters.groundY);
    const double outward[3] = {exact.x, relative_y, exact.z};
    const double coordinate[3] = {exact.x, exact.y, exact.z};
    float rounded[3] = {result.position.x, result.position.y, result.position.z};
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (outward[axis] > 0 && double(rounded[axis]) < coordinate[axis]) {
            rounded[axis] = nextafterf(rounded[axis], HUGE_VALF);
            result.rounded_outward = true;
        } else if (outward[axis] < 0 && double(rounded[axis]) > coordinate[axis]) {
            rounded[axis] = nextafterf(rounded[axis], -HUGE_VALF);
            result.rounded_outward = true;
        }
    }
    result.position = {rounded[0], rounded[1], rounded[2]};
    if (!m1::finite(result.position)) {
        result.status = Status::NonfiniteGeometry;
        return result;
    }
    result.status = detail::height_at(parameters,
        {rounded[0], rounded[1], rounded[2]}, rounded_height, radius);
    if (result.status == Status::Ok && rounded_height < 0)
        result.status = Status::InsideGround;
    return result;
}

} // namespace m3
