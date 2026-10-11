#pragma once

#include <cfloat>
#include <cstdint>
#include "m3/tracking_parameters.hpp"
#include "m3/tracking_float.hpp"

namespace m3::tracking {

using Vec3=m1::Vec3;

enum class Status : std::uint32_t {
    Ok, InvalidParameters, InvalidRay, NonfiniteGeometry, InsideGround,
    UnresolvedGeometry, InvalidCoefficient, InvalidRandom, TrackingLimit,
    TrackingRoundoff, MajorantViolation, NumericalUnderflow
};
M1_HD inline const char* status_name(Status status) {
    switch(status) {
    case Status::Ok:return "ok";
    case Status::InvalidParameters:return "invalid_parameters";
    case Status::InvalidRay:return "invalid_ray";
    case Status::NonfiniteGeometry:return "nonfinite_geometry";
    case Status::InsideGround:return "inside_ground";
    case Status::UnresolvedGeometry:return "unresolved_geometry";
    case Status::InvalidCoefficient:return "invalid_coefficient";
    case Status::InvalidRandom:return "invalid_random";
    case Status::TrackingLimit:return "tracking_limit";
    case Status::TrackingRoundoff:return "tracking_roundoff";
    case Status::MajorantViolation:return "majorant_violation";
    case Status::NumericalUnderflow:return "numerical_underflow";
    }
    return "unknown_tracking_status";
}

// Raw OptiX parameterization: p(t)=origin+t*direction. The direction is not
// assumed to have exact unit length. Tracking uses rate sigma_majorant*|d|.
struct RaySegment { Vec3 origin{},direction{};float t_max=0; };
struct PairVec3 { FloatPair x{},y{},z{}; };
M1_HD inline bool finite(PairVec3 a) { return finite(a.x)&&finite(a.y)&&finite(a.z); }
M1_HD inline PairVec3 paired(Vec3 a) { return {{a.x,0},{a.y,0},{a.z,0}}; }
M1_HD inline Vec3 value(PairVec3 a) { return {value(a.x),value(a.y),value(a.z)}; }

struct ShellClip {
    Status status=Status::Ok;
    bool has_medium=false,hits_ground=false,ground_occludes=false;
    float begin_t=0,end_t=0,ground_t=0,direction_length=0;
    // These carry the actual integration/sampling bounds. The single-float
    // fields above are rounded diagnostics, not the bounds for a long ray.
    FloatPair begin_parameter{},end_parameter{},ground_parameter{};
};
struct HeightResult {
    Status status=Status::Ok;
    float height_m=0,radius_m=0;
    FloatPair squared_ground_distance{}; // |p-centre|^2-R^2, signed [m^2].
};
struct PositionResult {
    Status status=Status::Ok;
    Vec3 position{};
    bool rounded_outward=false;
};
struct PlanetHit {
    Status status=Status::Ok;
    Vec3 position{},normal{};
    bool rounded_outward=false;
    float radial_correction_m=0;
    // NaN means this diagnostic was not reached. These are recorded on a
    // rejected root as well as a successful hit; they never relax the bound.
    float root_height_m=NAN,root_height_bound_m=NAN;
};

M1_HD inline Status validate_geometry_parameters(const TrackingParameters& p) {
    if (!m1::finite(p.planetRadius)||!(p.planetRadius>0)||
        !m1::finite(p.atmosphereHeight)||!(p.atmosphereHeight>0)||
        !m1::finite(p.groundY)) return Status::InvalidParameters;
    // A bounded FP32 geometry domain. There is no rescaling of the physical
    // scene and no silent overflow/underflow of the squared-radius arithmetic.
    const auto top=add(FloatPair{p.planetRadius,0},p.atmosphereHeight);
    const auto square=mul(top,top);
    if (!finite(top)||!finite(square)||!(square.hi>=FLT_MIN))
        return Status::InvalidParameters;
    return Status::Ok;
}

M1_HD inline PairVec3 point_at(const RaySegment& ray,FloatPair t) {
    return {add(mul(t,ray.direction.x),ray.origin.x),
            add(mul(t,ray.direction.y),ray.origin.y),
            add(mul(t,ray.direction.z),ray.origin.z)};
}
M1_HD inline PairVec3 point_at(const RaySegment& ray,float t) {
    return point_at(ray,FloatPair{t,0});
}

namespace geometry_detail {

M1_HD inline PairVec3 relative_point(const TrackingParameters& p,PairVec3 point) {
    return {point.x,add(sub(point.y,p.groundY),p.planetRadius),point.z};
}
M1_HD inline FloatPair dot(PairVec3 a,PairVec3 b) {
    return add(add(mul(a.x,b.x),mul(a.z,b.z)),mul(a.y,b.y));
}
M1_HD inline FloatPair dot(PairVec3 a,Vec3 b) {
    return add(add(mul(a.x,b.x),mul(a.z,b.z)),mul(a.y,b.y));
}
M1_HD inline FloatPair ground_difference(const TrackingParameters& p,PairVec3 point) {
    const auto y=sub(point.y,p.groundY);
    // Keep R separate from the metre-valued local y. Products and additions
    // retain their FMA low parts before cancellation, including far from the
    // scene's tangent origin. Computing (R+y) as one float first loses height.
    const auto vertical=mul(y,add(FloatPair{2.0f*p.planetRadius,0},y));
    return add(add(mul(point.x,point.x),mul(point.z,point.z)),vertical);
}
M1_HD inline HeightResult height_unchecked(const TrackingParameters& p,PairVec3 point) {
    HeightResult result;
    if(!finite(point)){result.status=Status::NonfiniteGeometry;return result;}
    result.squared_ground_distance=ground_difference(p,point);
    const auto radius_squared=add(product(p.planetRadius,p.planetRadius),
                                  result.squared_ground_distance);
    if(!finite(result.squared_ground_distance)||!finite(radius_squared)||
       compare(radius_squared,0)<0) {
        result.status=Status::NonfiniteGeometry;return result;
    }
    result.radius_m=::sqrtf(value(radius_squared));
    result.height_m=value(result.squared_ground_distance)/(result.radius_m+p.planetRadius);
    if(!m1::finite(result.radius_m)||!m1::finite(result.height_m))
        result.status=Status::NonfiniteGeometry;
    return result;
}

struct PreparedRay {
    FloatPair a{},b{},ground_c{};
    float direction_length=0;
};
M1_HD inline Status prepare(const TrackingParameters& p,const RaySegment& ray,PreparedRay& out) {
    const auto valid=validate_geometry_parameters(p);
    if(valid!=Status::Ok)return valid;
    if(!m1::finite(ray.origin)||!m1::finite(ray.direction)||!(ray.t_max>=0)||
       (!m1::finite(ray.t_max)&&ray.t_max!=HUGE_VALF))return Status::InvalidRay;
    out.a=add(add(product(ray.direction.x,ray.direction.x),
                  product(ray.direction.z,ray.direction.z)),
                  product(ray.direction.y,ray.direction.y));
    if(!finite(out.a)||!(out.a.hi>=FLT_MIN))return Status::InvalidRay;
    out.direction_length=value(square_root(out.a));
    const auto origin=paired(ray.origin);
    out.ground_c=ground_difference(p,origin);
    out.b=dot(relative_point(p,origin),ray.direction);
    if(!finite(out.ground_c)||!finite(out.b)||!m1::finite(out.direction_length))
        return Status::NonfiniteGeometry;
    if(compare(out.ground_c,0)<0)return Status::InsideGround;
    return Status::Ok;
}

struct Roots { Status status=Status::Ok;bool hit=false;FloatPair near_t{},far_t{}; };
M1_HD inline Roots roots(const PreparedRay& r,FloatPair c) {
    Roots result;
    // Retain coefficient low parts as well as the error in b*b-a*c. FMA on a
    // discriminant whose coefficients were already rounded is insufficient.
    const auto discriminant=sub(mul(r.b,r.b),mul(r.a,c));
    if(!finite(discriminant)){result.status=Status::NonfiniteGeometry;return result;}
    if(compare(discriminant,0)<0)return result;
    const auto root=square_root(discriminant);
    const auto signed_root=compare(r.b,0)<0 ? negate(root) : root;
    const auto q=negate(add(r.b,signed_root));
    if(compare(q,0)==0) {
        result.near_t=divide(negate(r.b),r.a);
        result.far_t=result.near_t;
    } else {
        result.near_t=divide(q,r.a);
        result.far_t=divide(c,q);
        if(compare(result.near_t,result.far_t)>0) {
            const auto temp=result.near_t;result.near_t=result.far_t;result.far_t=temp;
        }
    }
    if(!finite(root)||!finite(result.near_t)||!finite(result.far_t)) {
        result.status=Status::NonfiniteGeometry;return result;
    }
    result.hit=true;return result;
}

M1_HD inline PositionResult outward_representation(const TrackingParameters& p,PairVec3 source) {
    PositionResult result;
    result.position=value(source);
    if(!m1::finite(result.position)){result.status=Status::NonfiniteGeometry;return result;}
    auto represented=height_unchecked(p,paired(result.position));
    if(represented.status!=Status::Ok){result.status=represented.status;return result;}
    if(compare(represented.squared_ground_distance,0)>=0)return result;
    const auto relative=relative_point(p,source);
    const FloatPair exact[3]={source.x,source.y,source.z};
    const FloatPair radial[3]={relative.x,relative.y,relative.z};
    float rounded[3]={result.position.x,result.position.y,result.position.z};
    for(int i=0;i<3;++i) {
        // Only components rounded toward the planetary centre move, by one
        // adjacent float. Components already rounded outward stay unchanged.
        if(compare(radial[i],0)>0&&compare(exact[i],rounded[i])>0) {
            rounded[i]=::nextafterf(rounded[i],HUGE_VALF);result.rounded_outward=true;
        } else if(compare(radial[i],0)<0&&compare(exact[i],rounded[i])<0) {
            rounded[i]=::nextafterf(rounded[i],-HUGE_VALF);result.rounded_outward=true;
        }
    }
    result.position={rounded[0],rounded[1],rounded[2]};
    represented=height_unchecked(p,paired(result.position));
    if(represented.status!=Status::Ok)result.status=represented.status;
    else if(compare(represented.squared_ground_distance,0)<0)
        result.status=Status::UnresolvedGeometry;
    return result;
}

} // namespace geometry_detail

M1_HD inline HeightResult height(const TrackingParameters& p,PairVec3 point) {
    HeightResult result;
    result.status=validate_geometry_parameters(p);
    if(result.status!=Status::Ok)return result;
    result=geometry_detail::height_unchecked(p,point);
    if(result.status==Status::Ok&&compare(result.squared_ground_distance,0)<0)
        result.status=Status::InsideGround;
    return result;
}
M1_HD inline HeightResult height(const TrackingParameters& p,Vec3 point) {
    return height(p,paired(point));
}

M1_HD inline ShellClip clip(const TrackingParameters& p,const RaySegment& ray) {
    ShellClip result;
    geometry_detail::PreparedRay r;
    result.status=geometry_detail::prepare(p,ray,r);
    if(result.status!=Status::Ok)return result;
    result.direction_length=r.direction_length;
    if(ray.t_max==0)return result;
    auto ground=geometry_detail::roots(r,r.ground_c);
    if(ground.status!=Status::Ok){result.status=ground.status;return result;}
    const FloatPair limit{ray.t_max,0};
    // Preserve an exact finite first-hit endpoint. Root roundoff must not turn
    // a light/ground endpoint into an occluder just before that endpoint.
    if(ground.hit&&m1::finite(ray.t_max)) {
        const auto endpoint=point_at(ray,limit);
        if(compare(geometry_detail::ground_difference(p,endpoint),0)==0&&
           compare(add(r.b,mul(r.a,limit)),0)<=0)
            ground.near_t=limit;
    }
    if(ground.hit&&compare(ground.near_t,0)>=0&&compare(ground.near_t,limit)<=0&&
       (compare(ground.near_t,0)>0||compare(r.b,0)<0)) {
        result.hits_ground=true;result.ground_parameter=ground.near_t;
        result.ground_t=value(ground.near_t);
        result.ground_occludes=compare(ground.near_t,limit)<0;
    }
    const auto shell_difference=mul(FloatPair{p.atmosphereHeight,0},
        add(FloatPair{2.0f*p.planetRadius,0},p.atmosphereHeight));
    const auto top=geometry_detail::roots(r,sub(r.ground_c,shell_difference));
    if(top.status!=Status::Ok){result.status=top.status;return result;}
    if(!top.hit||compare(top.far_t,0)<=0)return result;
    const auto begin=maximum(FloatPair{},top.near_t);
    auto end=minimum(limit,top.far_t);
    if(result.hits_ground)end=minimum(end,result.ground_parameter);
    if(compare(end,begin)>0) {
        result.has_medium=true;result.begin_parameter=begin;result.end_parameter=end;
        result.begin_t=value(begin);result.end_t=value(end);
    }
    return result;
}

// A valid scatter point is never projected to the surface. If its nearest
// float representation crosses the opaque ground, directed component rounding
// preserves the exterior using at most one adjacent float step per component.
// A genuinely interior PairVec3 is rejected before considering its float image.
M1_HD inline PositionResult represent_medium_point(const TrackingParameters& p,PairVec3 source) {
    PositionResult result;
    const auto checked=height(p,source);
    if(checked.status!=Status::Ok){result.status=checked.status;return result;}
    return geometry_detail::outward_representation(p,source);
}
M1_HD inline PositionResult represent_medium_point(const TrackingParameters& p,Vec3 source) {
    return represent_medium_point(p,paired(source));
}

// Use only with a successful clip() result for this same ray. Unlike a medium
// sample, this endpoint is known to belong to the mathematical planet surface.
// Projecting the tiny root residual is therefore a geometric representation
// correction. Its upper bound scales with the FP32-pair arithmetic, not an
// arbitrary world-space epsilon, and is reported in radial_correction_m.
M1_HD inline PlanetHit planet_hit(const TrackingParameters& p,const RaySegment& ray,const ShellClip& clipped) {
    PlanetHit result;
    result.status=validate_geometry_parameters(p);
    if(result.status!=Status::Ok)return result;
    if(clipped.status!=Status::Ok||!clipped.hits_ground||!finite(clipped.ground_parameter)) {
        result.status=Status::InvalidRay;return result;
    }
    const auto source=point_at(ray,clipped.ground_parameter);
    const auto measured=geometry_detail::height_unchecked(p,source);
    if(measured.status!=Status::Ok){result.status=measured.status;return result;}
    result.root_height_m=measured.height_m;
    const float source_scale=fmaxf(fmaxf(fabsf(ray.origin.x),fabsf(ray.origin.y)),fabsf(ray.origin.z));
    const float travel_scale=fabsf(value(clipped.ground_parameter))*clipped.direction_length;
    const float pair_roundoff_bound=(128.0f*FLT_EPSILON*FLT_EPSILON)*
        (p.planetRadius+source_scale+travel_scale);
    result.root_height_bound_m=pair_roundoff_bound;
    if(!m1::finite(pair_roundoff_bound)||fabsf(measured.height_m)>pair_roundoff_bound) {
        result.status=Status::UnresolvedGeometry;return result;
    }
    const auto relative=geometry_detail::relative_point(p,source);
    const auto radius=square_root(add(product(p.planetRadius,p.planetRadius),
                                      measured.squared_ground_distance));
    if(!finite(radius)||compare(radius,0)<=0){result.status=Status::NonfiniteGeometry;return result;}
    const auto scale=divide(FloatPair{p.planetRadius,0},radius);
    const PairVec3 projected_relative{mul(relative.x,scale),mul(relative.y,scale),mul(relative.z,scale)};
    const PairVec3 projected{projected_relative.x,
        add(sub(projected_relative.y,p.planetRadius),p.groundY),projected_relative.z};
    const Vec3 normal_base{value(divide(relative.x,radius)),value(divide(relative.y,radius)),
                           value(divide(relative.z,radius))};
    const float normal_length=m1::length(normal_base);
    if(!m1::finite(normal_length)||!(normal_length>0)){result.status=Status::NonfiniteGeometry;return result;}
    result.normal=normal_base/normal_length;
    auto represented=geometry_detail::outward_representation(p,projected);
    if(represented.status==Status::UnresolvedGeometry) {
        // The pair-valued projection itself can retain a sub-ULP radius error.
        // One final outward coordinate step is allowed ONLY for this known
        // sphere endpoint. Choose the coordinate with the largest radial gain.
        float coordinates[3]={represented.position.x,represented.position.y,represented.position.z};
        const float normal[3]={result.normal.x,result.normal.y,result.normal.z};
        int best=-1;float gain=0;
        for(int i=0;i<3;++i) {
            if(normal[i]==0)continue;
            const float next=::nextafterf(coordinates[i],normal[i]>0 ? HUGE_VALF : -HUGE_VALF);
            const float candidate=fabsf((next-coordinates[i])*normal[i]);
            if(candidate>gain){gain=candidate;best=i;}
        }
        if(best>=0) {
            coordinates[best]=::nextafterf(coordinates[best],normal[best]>0 ? HUGE_VALF : -HUGE_VALF);
            represented.position={coordinates[0],coordinates[1],coordinates[2]};
            represented.rounded_outward=true;
            represented.status=height(p,represented.position).status;
        }
    }
    result.status=represented.status;result.position=represented.position;
    result.rounded_outward=represented.rounded_outward;
    result.radial_correction_m=-measured.height_m;
    return result;
}

} // namespace m3::tracking
