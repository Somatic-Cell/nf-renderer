#pragma once

#if defined(__CUDACC__)
#error "The FP64 numerical reference is CPU-only. GPU rendering must include m3/tracking.hpp."
#endif

#include "m3/parameters.hpp"
#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define M3_HD __host__ __device__
#else
#define M3_HD
#endif

// Portable CPU/CUDA reference integration of a spherical exponential medium.
// This is numerical quadrature and inverse CDF, not a closed-form Chapman
// solution, a LUT, or a tracking estimator. Fixed work limits return a failure;
// callers must not turn a failed result into transparent medium or a black path.
namespace m3 {

inline constexpr std::uint32_t maximum_quadrature_depth = 24;

enum class Status : std::uint32_t {
    Ok, InvalidParameters, InvalidRay, InsideGround, NonfiniteGeometry,
    InvalidCoefficient, InvalidQuantile, QuadratureBudget, QuadratureDepth,
    QuadratureRoundoff, NonfiniteIntegral, NumericalUnderflow,
    EndpointUnresolved, InverseIterations, InverseRoundoff
};

inline const char* status_name(Status s) {
    switch (s) {
    case Status::Ok: return "ok";
    case Status::InvalidParameters: return "invalid_parameters";
    case Status::InvalidRay: return "invalid_ray";
    case Status::InsideGround: return "inside_ground";
    case Status::NonfiniteGeometry: return "nonfinite_geometry";
    case Status::InvalidCoefficient: return "invalid_coefficient";
    case Status::InvalidQuantile: return "invalid_quantile";
    case Status::QuadratureBudget: return "quadrature_budget";
    case Status::QuadratureDepth: return "quadrature_depth";
    case Status::QuadratureRoundoff: return "quadrature_roundoff";
    case Status::NonfiniteIntegral: return "nonfinite_integral";
    case Status::NumericalUnderflow: return "numerical_underflow";
    case Status::EndpointUnresolved: return "endpoint_unresolved";
    case Status::InverseIterations: return "inverse_iterations";
    case Status::InverseRoundoff: return "inverse_roundoff";
    }
    return "unknown_atmosphere_status";
}

struct DVec3 { double x = 0, y = 0, z = 0; };
M3_HD inline DVec3 operator+(DVec3 a, DVec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
M3_HD inline DVec3 operator-(DVec3 a, DVec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
M3_HD inline DVec3 operator*(DVec3 a, double b) { return {a.x*b,a.y*b,a.z*b}; }
M3_HD inline DVec3 operator/(DVec3 a, double b) { return {a.x/b,a.y/b,a.z/b}; }
M3_HD inline double dot(DVec3 a, DVec3 b) { return ::fma(a.x,b.x,::fma(a.y,b.y,a.z*b.z)); }
M3_HD inline DVec3 cross(DVec3 a, DVec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
M3_HD inline double length(DVec3 a) { return ::hypot(::hypot(a.x,a.y),a.z); }
M3_HD inline bool finite(double x) { return x == x && x <= DBL_MAX && x >= -DBL_MAX; }
M3_HD inline bool finite(DVec3 v) { return finite(v.x)&&finite(v.y)&&finite(v.z); }
M3_HD inline double minimum(double a,double b) { return a < b ? a : b; }
M3_HD inline double maximum(double a,double b) { return a > b ? a : b; }

M3_HD inline Status validate_parameters(const AtmosphereParameters& p) {
    const double top = p.planetRadius + p.atmosphereHeight;
    if (p.enabled > 1 || p.areaNeeTransmittance > 1 ||
        !finite(p.planetRadius) || !(p.planetRadius > 0) ||
        !finite(p.atmosphereHeight) || !(p.atmosphereHeight > 0) ||
        !finite(top*top) || !(top > p.planetRadius) ||
        !finite(p.scaleHeight) || !(p.scaleHeight > 0) ||
        !finite(p.groundY) || !finite(p.groundY-p.planetRadius) ||
        !finite(p.beta550) || p.beta550 < 0 ||
        !finite(p.groundAlbedo) || p.groundAlbedo < 0 || p.groundAlbedo > 1 ||
        !finite(p.tauAbsoluteTolerance) || !(p.tauAbsoluteTolerance > 0) ||
        !finite(p.tauRelativeTolerance) || p.tauRelativeTolerance < 0 ||
        !finite(p.inverseTauTolerance) || !(p.inverseTauTolerance > 0) ||
        !finite(p.distanceAbsoluteTolerance) || !(p.distanceAbsoluteTolerance > 0) ||
        !finite(p.distanceRelativeTolerance) || p.distanceRelativeTolerance < 0 ||
        p.maxQuadratureDepth == 0 || p.maxQuadratureDepth > maximum_quadrature_depth ||
        p.maxQuadratureEvaluations < 15 || p.maxQuadratureEvaluations > 1'048'576 ||
        p.maxInverseIterations == 0 || p.maxInverseIterations > 128)
        return Status::InvalidParameters;
    return Status::Ok;
}

M3_HD inline bool rayleigh_coefficient(const AtmosphereParameters& p,
                                      double wavelength_nm, double& beta_per_m) {
    if (validate_parameters(p) != Status::Ok || !finite(wavelength_nm) || !(wavelength_nm > 0))
        return false;
    if (p.beta550 == 0) { beta_per_m = 0; return true; }
    const double ratio = 550.0/wavelength_nm;
    const double square = ratio*ratio;
    const double result = p.beta550*square*square;
    if (!finite(result) || !(result > 0)) return false;
    beta_per_m = result;
    return true;
}

struct RaySegment {
    DVec3 origin{};                    // Scene coordinates [m].
    DVec3 direction{0,1,0};           // Raw OptiX direction is accepted.
    double t_max = HUGE_VAL;          // Original ray parameter, not assumed metres.
};

struct ShellClip {
    Status status = Status::Ok;
    bool has_medium = false;
    double begin_m = 0, end_m = 0;    // Physical distances from the ray origin.
    bool hits_ground = false;
    bool ground_occludes = false;    // Strictly before t_max, except an inward t=0 hit.
    double ground_distance_m = HUGE_VAL;
    double direction_length = 0;
    double limit_distance_m = 0;
};

struct IntegralResult {
    Status status = Status::Ok;
    double tau = 0;
    double estimated_error = 0;
    std::uint32_t evaluations = 0;
    bool omitted_bounded_tail = false;
};

struct TransmittanceResult {
    Status status = Status::Ok;
    double transmittance = 1; // Medium only. Apply ground_occluded separately.
    double tau = 0, estimated_error = 0;
    bool ground_occluded = false;
    std::uint32_t evaluations = 0;
};

enum class Endpoint : std::uint32_t { SegmentEnd, AtmosphereExit, Ground, None };

struct FreeFlightSample {
    Status status = Status::Ok;
    bool scattered = false;
    Endpoint endpoint = Endpoint::SegmentEnd;
    double distance_m = 0, ray_t = 0;
    double tau = 0, transmittance = 1;
    double pdf_per_m = 0;       // Continuous scattering-event density.
    double endpoint_mass = 1;  // Discrete no-event probability, not a density.
    double sigma_s_per_m = 0;
    double tau_residual = 0, estimated_tau_error = 0;
    double estimated_distance_error_m = 0;
    std::uint32_t iterations = 0, evaluations = 0;
};

namespace detail {

// Preserve metre-scale local coordinates before adding the Earth radius.
// h = (r^2-R^2)/(r+R) avoids the subtraction sqrt(r^2)-R near the ground.
M3_HD inline Status height_at(const AtmosphereParameters& p,DVec3 point,
                              double& height,double& radius,double* ground_c=nullptr) {
    const double y = point.y-p.groundY;
    const double relative_y = p.planetRadius+y;
    radius = ::hypot(::hypot(point.x,point.z),relative_y);
    const double difference = ::fma(y,2*p.planetRadius+y,
                                   ::fma(point.x,point.x,point.z*point.z));
    if (ground_c) *ground_c=difference;
    height = difference/(radius+p.planetRadius);
    return finite(height)&&finite(radius*radius) ? Status::Ok : Status::NonfiniteGeometry;
}

struct PreparedRay {
    DVec3 origin{}, unit_direction{}, relative_origin{};
    double norm = 0, limit = 0, axial = 0, impact = 0, radius = 0, height = 0;
    double direction_squared = 1,ground_c = 0,ground_discriminant = 0,closest_height = 0;
};

M3_HD inline double difference_of_products(double a,double b,double c,double d) {
    const double cd=c*d;
    const double correction=::fma(-c,d,cd);
    return ::fma(a,b,-cd)+correction;
}

M3_HD inline Status prepare(const AtmosphereParameters& p,const RaySegment& r,PreparedRay& out) {
    const Status valid = validate_parameters(p);
    if (valid != Status::Ok) return valid;
    if (!finite(r.origin)||!finite(r.direction)||!(r.t_max >= 0)||
        (!finite(r.t_max)&&r.t_max != HUGE_VAL)) return Status::InvalidRay;
    out.norm = length(r.direction);
    if (!finite(out.norm)||!(out.norm > 0)) return Status::InvalidRay;
    out.origin = r.origin;
    out.unit_direction = r.direction/out.norm;
    out.limit = r.t_max == HUGE_VAL ? HUGE_VAL : r.t_max*out.norm;
    if (r.t_max != HUGE_VAL && !finite(out.limit)) return Status::InvalidRay;
    out.relative_origin = {r.origin.x,p.planetRadius+(r.origin.y-p.groundY),r.origin.z};
    Status status = height_at(p,r.origin,out.height,out.radius,&out.ground_c);
    if (status != Status::Ok) return status;
    if (out.height < 0) return Status::InsideGround;
    out.axial = dot(out.relative_origin,out.unit_direction);
    out.direction_squared=dot(out.unit_direction,out.unit_direction);
    out.impact = length(cross(out.relative_origin,out.unit_direction))/::sqrt(out.direction_squared);
    out.ground_discriminant=difference_of_products(out.axial,out.axial,out.direction_squared,out.ground_c);
    // The closest altitude must keep the sign of the same discriminant used
    // for clipping. Subtracting impact-R can round a real near-tangent miss to
    // a ground hit (or a small positive altitude to a negative one).
    out.closest_height=(-out.ground_discriminant/out.direction_squared)/(out.impact+p.planetRadius);
    if (!finite(out.axial)||!finite(out.impact)||!finite(out.ground_discriminant)||
        !finite(out.closest_height)) return Status::NonfiniteGeometry;
    return Status::Ok;
}

struct Roots { bool hit = false; double near_t = 0, far_t = 0; Status status = Status::Ok; };
M3_HD inline Roots roots(const AtmosphereParameters& p,const PreparedRay& r,double height) {
    Roots result;
    const double radius = p.planetRadius+height;
    // The ground/top height differences retain small distances at a boundary.
    const double c = height == 0 ? r.ground_c : (r.height-height)*(r.radius+radius);
    const double discriminant=height == 0 ? r.ground_discriminant :
        difference_of_products(r.axial,r.axial,r.direction_squared,c);
    if (discriminant < 0) return result;
    if (!finite(discriminant)) { result.status=Status::NonfiniteGeometry; return result; }
    const double span=::sqrt(discriminant);
    const double q=-r.axial-::copysign(span,r.axial);
    double a=(-r.axial-span)/r.direction_squared,b=(-r.axial+span)/r.direction_squared;
    if (q != 0) { a=q/r.direction_squared; b=c/q; }
    if (a > b) { const double temp = a; a = b; b = temp; }
    if (!finite(a)||!finite(b)) { result.status=Status::NonfiniteGeometry; return result; }
    result.hit=true; result.near_t=a; result.far_t=b;
    return result;
}

M3_HD inline ShellClip clip(const AtmosphereParameters& p,const PreparedRay& r) {
    ShellClip c;
    c.direction_length=r.norm; c.limit_distance_m=r.limit;
    if (r.limit == 0) return c;
    const Roots ground=roots(p,r,0);
    if (ground.status != Status::Ok) { c.status=ground.status; return c; }
    if (ground.hit && ground.near_t >= 0 && ground.near_t <= r.limit &&
        (ground.near_t > 0 || r.axial < 0)) {
        c.hits_ground=true; c.ground_distance_m=ground.near_t;
        c.ground_occludes=ground.near_t < r.limit;
    }
    const Roots top=roots(p,r,p.atmosphereHeight);
    if (top.status != Status::Ok) { c.status=top.status; return c; }
    if (!top.hit || top.far_t <= 0) return c;
    const double begin=maximum(0,top.near_t);
    double end=minimum(r.limit,top.far_t);
    if (c.hits_ground) end=minimum(end,c.ground_distance_m);
    if (end > begin) { c.has_medium=true; c.begin_m=begin; c.end_m=end; }
    return c;
}

M3_HD inline DVec3 point_at(const PreparedRay& r,double distance) {
    return {::fma(r.unit_direction.x,distance,r.origin.x),
            ::fma(r.unit_direction.y,distance,r.origin.y),
            ::fma(r.unit_direction.z,distance,r.origin.z)};
}

M3_HD inline double radius_at(const PreparedRay& r,double distance) {
    const double norm=::sqrt(r.direction_squared);
    return ::hypot(r.impact,::fma(norm,distance,r.axial/norm));
}

M3_HD inline double sigma_at_height(double height,double scale_height,double beta) {
    return beta == 0 ? 0 : ::exp(::log(beta)-height/scale_height);
}

struct Piece {
    double radius=0,height=0,impact=0,radial_dot=0,length=0;
    double direction_squared=1;
    M3_HD double radial_change(double t) const {
        const double change_squared=t*::fma(direction_squared,t,2*radial_dot);
        return change_squared/(::sqrt(::fma(radius,radius,change_squared))+radius);
    }
    M3_HD double distance_at_change(double delta) const {
        if (delta == 0) return 0;
        const double n=delta*(2*radius+delta);
        return n/(::sqrt(::fma(direction_squared,n,radial_dot*radial_dot))+radial_dot);
    }
};

M3_HD inline Status make_piece(const AtmosphereParameters& p,const PreparedRay& r,
                               const ShellClip& c,double anchor,double other,bool tangent,Piece& piece) {
    const double sign=other > anchor ? 1.0 : -1.0;
    piece.direction_squared=r.direction_squared;
    piece.radial_dot=tangent ? 0 : sign*::fma(r.direction_squared,anchor,r.axial);
    piece.impact=r.impact; piece.length=::fabs(other-anchor);
    if (piece.radial_dot < 0 || !finite(piece.radial_dot)) return Status::NonfiniteGeometry;
    if (c.hits_ground && anchor == c.ground_distance_m) {
        piece.radius=p.planetRadius; piece.height=0;
    } else if (tangent) {
        piece.radius=r.impact; piece.height=r.closest_height;
    } else {
        const Status status=height_at(p,point_at(r,anchor),piece.height,piece.radius);
        if (status != Status::Ok) return status;
    }
    if (piece.height < 0) return Status::InsideGround;
    return Status::Ok;
}

struct Numerics { double absolute=0,relative=0; };
struct Budget { std::uint32_t used=0,maximum=0; };
struct Quad { Status status=Status::Ok; double value=0,error=0; };

M3_HD inline Quad kronrod15(const Piece& p,double scale_height,double log_sigma,
                            double a,double b,Budget& budget) {
    Quad result;
    if (budget.maximum-budget.used < 15) { result.status=Status::QuadratureBudget; return result; }
    budget.used+=15;
    // Embedded Gauss 7 / Kronrod 15, with QUADPACK-style resasc/resabs error
    // rescaling. This is an error estimate, not a rigorous floating-point bound.
    constexpr double x[8]={0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,0.586087235467691130294144838258730,
        0.405845151377397166906606412076961,0.207784955007898467600689403773245,0};
    constexpr double wk[8]={0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,0.204432940075298892414161999234649,
        0.209482141084727828012999174891714};
    constexpr double wg[4]={0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,0.381830050505118944950369775488975,
        0.417959183673469387755102040816327};
    const double half=0.5*(b-a),mid=a+half;
    const double center=::exp(log_sigma-p.radial_change(mid)/scale_height);
    double k=wk[7]*center,g=wg[3]*center,abs_sum=wk[7]*::fabs(center);
    double left[7],right[7];
    for (unsigned j=0;j<7;++j) {
        left[j]=::exp(log_sigma-p.radial_change(::fma(-half,x[j],mid))/scale_height);
        right[j]=::exp(log_sigma-p.radial_change(::fma(half,x[j],mid))/scale_height);
        const double pair=left[j]+right[j];
        k+=wk[j]*pair; abs_sum+=wk[j]*(::fabs(left[j])+::fabs(right[j]));
        if (j%2 == 1) g+=wg[j/2]*pair;
    }
    const double mean=0.5*k;
    double asc=wk[7]*::fabs(center-mean);
    for (unsigned j=0;j<7;++j)
        asc+=wk[j]*(::fabs(left[j]-mean)+::fabs(right[j]-mean));
    result.value=k*half; abs_sum*=half; asc*=half;
    result.error=::fabs((k-g)*half);
    if (asc > 0 && result.error > 0)
        result.error=asc*minimum(1,::pow(200*result.error/asc,1.5));
    result.error=maximum(result.error,50*DBL_EPSILON*abs_sum);
    if (!finite(result.value)||result.value < 0||!finite(result.error))
        result.status=Status::NonfiniteIntegral;
    return result;
}

M3_HD inline Quad adaptive(const AtmosphereParameters& settings,const Piece& p,
                           double log_sigma,double a,double b,Numerics n,Budget& budget) {
    struct Work { double a,b,absolute; std::uint32_t depth; };
    Work stack[maximum_quadrature_depth+2];
    unsigned size=1;
    stack[0]={a,b,n.absolute,0};
    Quad result;
    while (size != 0) {
        const Work current=stack[--size];
        const Quad q=kronrod15(p,settings.scaleHeight,log_sigma,current.a,current.b,budget);
        if (q.status != Status::Ok) { result.status=q.status; return result; }
        if (q.error <= current.absolute+n.relative*q.value) {
            result.value+=q.value; result.error+=q.error;
            continue;
        }
        if (current.depth >= settings.maxQuadratureDepth) {
            result.status=Status::QuadratureDepth; return result;
        }
        const double mid=current.a+0.5*(current.b-current.a);
        if (mid == current.a || mid == current.b) { result.status=Status::QuadratureRoundoff; return result; }
        const unsigned depth=current.depth+1;
        const double absolute=0.5*current.absolute;
        if (!(absolute > 0)) { result.status=Status::NumericalUnderflow; return result; }
        stack[size++]={mid,current.b,absolute,depth};
        stack[size++]={current.a,mid,absolute,depth};
    }
    return result;
}

M3_HD inline IntegralResult integrate_piece(const AtmosphereParameters& settings,
                                            const Piece& p,double beta,Numerics n,Budget& budget) {
    IntegralResult result;
    if (p.length == 0 || beta == 0) return result;
    const double log_sigma=::log(beta)-p.height/settings.scaleHeight;
    const double log_upper=::log(p.length)+log_sigma;
    const double tail_allowance=0.125*n.absolute;
    if (!(tail_allowance > 0)) { result.status=Status::NumericalUnderflow; return result; }
    const double log_allowance=::log(tail_allowance);
    // Outward density is monotone. Bound any omitted tail by its length times
    // density at the tail start; never rely on an embedded rule noticing a
    // narrow high-density region in a long ray.
    if (log_upper <= log_allowance) {
        result.estimated_error=::exp(log_upper);
        if (result.estimated_error == 0) result.estimated_error=DBL_MIN;
        result.omitted_bounded_tail=true;
        return result;
    }
    const double total_q=p.radial_change(p.length)/settings.scaleHeight;
    const double core_q=minimum(total_q,log_upper-log_allowance);
    double core_length=p.length;
    if (core_q < total_q) {
        core_length=minimum(p.length,p.distance_at_change(core_q*settings.scaleHeight));
        if (core_length < p.length) {
            result.estimated_error=::exp(::log(p.length-core_length)+log_sigma-
                p.radial_change(core_length)/settings.scaleHeight);
            if (result.estimated_error == 0) result.estimated_error=DBL_MIN;
            result.omitted_bounded_tail=true;
        }
    }
    const double panel_count=maximum(1,::ceil(core_q/4));
    if (!finite(panel_count)||panel_count > double(settings.maxQuadratureEvaluations/15)) {
        result.status=Status::QuadratureBudget; return result;
    }
    const auto panels=static_cast<std::uint32_t>(panel_count);
    const double absolute=(n.absolute-result.estimated_error)/double(panels);
    if (!(absolute > 0)) { result.status=Status::NumericalUnderflow; return result; }
    double previous=0;
    for (std::uint32_t i=1;i<=panels;++i) {
        const double current=i == panels ? core_length :
            p.distance_at_change((core_q*double(i)/double(panels))*settings.scaleHeight);
        if (!(current > previous)) { result.status=Status::QuadratureRoundoff; return result; }
        const Quad q=adaptive(settings,p,log_sigma,previous,current,{absolute,n.relative},budget);
        if (q.status != Status::Ok) { result.status=q.status; return result; }
        result.tau+=q.value; result.estimated_error+=q.error;
        previous=current;
    }
    return result;
}

M3_HD inline IntegralResult integrate_range(const AtmosphereParameters& settings,const PreparedRay& r,
                                            const ShellClip& c,double end,double beta,Numerics n) {
    IntegralResult result;
    if (!c.has_medium || end <= c.begin_m || beta == 0) return result;
    Piece pieces[2]; unsigned count=1;
    const double closest=-r.axial/r.direction_squared;
    if (closest > c.begin_m && closest < end) {
        count=2;
        result.status=make_piece(settings,r,c,closest,c.begin_m,true,pieces[0]);
        if (result.status == Status::Ok)
            result.status=make_piece(settings,r,c,closest,end,true,pieces[1]);
    } else if (closest >= end) {
        result.status=make_piece(settings,r,c,end,c.begin_m,closest == end,pieces[0]);
    } else {
        result.status=make_piece(settings,r,c,c.begin_m,end,closest == c.begin_m,pieces[0]);
    }
    if (result.status != Status::Ok) return result;
    Budget budget{0,settings.maxQuadratureEvaluations};
    for (unsigned i=0;i<count;++i) {
        const IntegralResult part=integrate_piece(settings,pieces[i],beta,{n.absolute/count,n.relative},budget);
        result.evaluations=budget.used;
        if (part.status != Status::Ok) { result.status=part.status; return result; }
        result.tau+=part.tau; result.estimated_error+=part.estimated_error;
        result.omitted_bounded_tail=result.omitted_bounded_tail||part.omitted_bounded_tail;
    }
    if (!finite(result.tau)||!finite(result.estimated_error)) result.status=Status::NonfiniteIntegral;
    else if (result.estimated_error > n.absolute+n.relative*result.tau)
        result.status=Status::QuadratureRoundoff;
    return result;
}

} // namespace detail

M3_HD inline ShellClip clip_atmosphere(const AtmosphereParameters& p,const RaySegment& ray) {
    detail::PreparedRay r;
    ShellClip result;
    result.status=detail::prepare(p,ray,r);
    return result.status == Status::Ok ? detail::clip(p,r) : result;
}

M3_HD inline Status local_scattering_coefficient(const AtmosphereParameters& p,DVec3 point,
                                                 double beta_per_m,double& sigma_s_per_m) {
    const Status valid=validate_parameters(p);
    if (valid != Status::Ok) return valid;
    if (!finite(beta_per_m)||beta_per_m < 0) return Status::InvalidCoefficient;
    if (!finite(point)) return Status::NonfiniteGeometry;
    double height=0,radius=0;
    const Status geometry=detail::height_at(p,point,height,radius);
    if (geometry != Status::Ok) return geometry;
    if (height < 0) return Status::InsideGround;
    sigma_s_per_m=height > p.atmosphereHeight ? 0 :
        detail::sigma_at_height(height,p.scaleHeight,beta_per_m);
    return finite(sigma_s_per_m) ? Status::Ok : Status::NonfiniteIntegral;
}

M3_HD inline IntegralResult optical_depth(const AtmosphereParameters& p,const RaySegment& ray,
                                          double beta_per_m) {
    detail::PreparedRay r;
    IntegralResult result;
    result.status=detail::prepare(p,ray,r);
    if (result.status != Status::Ok) return result;
    if (!finite(beta_per_m)||beta_per_m < 0) { result.status=Status::InvalidCoefficient; return result; }
    const ShellClip c=detail::clip(p,r);
    if (c.status != Status::Ok) { result.status=c.status; return result; }
    return detail::integrate_range(p,r,c,c.end_m,beta_per_m,{p.tauAbsoluteTolerance,p.tauRelativeTolerance});
}

M3_HD inline TransmittanceResult transmittance(const AtmosphereParameters& p,const RaySegment& ray,
                                              double beta_per_m) {
    TransmittanceResult result;
    const ShellClip c=clip_atmosphere(p,ray);
    result.status=c.status;
    if (c.status != Status::Ok) return result;
    const IntegralResult integral=optical_depth(p,ray,beta_per_m);
    result.status=integral.status; result.tau=integral.tau;
    result.estimated_error=integral.estimated_error; result.evaluations=integral.evaluations;
    result.ground_occluded=c.ground_occludes;
    if (result.status == Status::Ok) result.transmittance=::exp(-result.tau);
    return result;
}

// The sampler uses the same physical density and numerical tau as transmittance.
// xi in [0,1) gives tau*=-log1p(-xi). There is a discrete endpoint mass T(L)
// and a continuous collision density sigma_s(s)*T(s) [1/m]. In this pure
// scattering model, the distance-sampling throughput factor is ONE on BOTH
// branches. Do not multiply throughput by transmittance or endpoint_mass again.
// The remaining approximation is explicit numerical integration/inversion error;
// the returned error is estimated, so this is not a claim of exact unbiasedness.
M3_HD inline FreeFlightSample sample_free_flight(const AtmosphereParameters& p,const RaySegment& ray,
                                                double beta_per_m,double xi) {
    FreeFlightSample result;
    detail::PreparedRay r;
    result.status=detail::prepare(p,ray,r);
    if (result.status != Status::Ok) return result;
    if (!finite(beta_per_m)||beta_per_m < 0) { result.status=Status::InvalidCoefficient; return result; }
    if (!finite(xi)||xi < 0||!(xi < 1)) { result.status=Status::InvalidQuantile; return result; }
    const ShellClip c=detail::clip(p,r);
    if (c.status != Status::Ok) { result.status=c.status; return result; }
    if (c.ground_occludes) { result.endpoint=Endpoint::Ground; result.distance_m=c.ground_distance_m; }
    else if (finite(r.limit)) { result.endpoint=Endpoint::SegmentEnd; result.distance_m=r.limit; }
    else { result.endpoint=Endpoint::AtmosphereExit; result.distance_m=c.has_medium ? c.end_m : 0; }
    result.ray_t=result.distance_m/r.norm;
    if (!c.has_medium || beta_per_m == 0) return result;
    const double target=-::log1p(-xi);
    if (target == 0) {
        double height=0,radius=0;
        result.status=detail::height_at(p,detail::point_at(r,c.begin_m),height,radius);
        if (result.status != Status::Ok) return result;
        const double sigma=detail::sigma_at_height(height,p.scaleHeight,beta_per_m);
        if (!finite(sigma)||!(sigma > 0)) { result.status=Status::NumericalUnderflow; return result; }
        result.scattered=true; result.endpoint=Endpoint::None;
        result.distance_m=c.begin_m; result.ray_t=c.begin_m/r.norm;
        result.pdf_per_m=sigma; result.endpoint_mass=0; result.sigma_s_per_m=sigma;
        return result;
    }
    detail::Numerics n{p.tauAbsoluteTolerance,p.tauRelativeTolerance};
    IntegralResult total;
    bool classified=false,collision=false;
    for (unsigned refinement=0;refinement<9;++refinement) {
        total=detail::integrate_range(p,r,c,c.end_m,beta_per_m,n);
        result.evaluations+=total.evaluations;
        if (total.status != Status::Ok) { result.status=total.status; return result; }
        if (target >= total.tau+total.estimated_error) { classified=true; break; }
        if (target < total.tau-total.estimated_error) { classified=true; collision=true; break; }
        n.absolute*=0.0625; n.relative*=0.0625;
    }
    if (!classified) { result.status=Status::EndpointUnresolved; return result; }
    if (!collision) {
        result.tau=total.tau; result.transmittance=::exp(-total.tau);
        result.endpoint_mass=result.transmittance; result.estimated_tau_error=total.estimated_error;
        return result;
    }
    double lo=c.begin_m,hi=c.end_m;
    double t=lo+(hi-lo)*(target/total.tau);
    const double distance_tolerance=p.distanceAbsoluteTolerance+p.distanceRelativeTolerance*(hi-lo);
    if (!finite(distance_tolerance)) { result.status=Status::InvalidParameters; return result; }
    for (std::uint32_t iteration=1;iteration<=p.maxInverseIterations;++iteration) {
        result.iterations=iteration;
        // Also retain the current candidate in a failed result for diagnostics;
        // only status==Ok makes it an accepted sampled distance.
        result.distance_m=t; result.ray_t=t/r.norm;
        double height=0,radius=0;
        result.status=detail::height_at(p,detail::point_at(r,t),height,radius);
        if (result.status != Status::Ok) return result;
        if (height < 0) { result.status=Status::InsideGround; return result; }
        const double sigma=detail::sigma_at_height(height,p.scaleHeight,beta_per_m);
        if (!finite(sigma)||!(sigma > 0)) { result.status=Status::NumericalUnderflow; return result; }
        const double wanted_error=0.0625*minimum(p.inverseTauTolerance,sigma*distance_tolerance);
        if (!(wanted_error > 0)) { result.status=Status::NumericalUnderflow; return result; }
        n.absolute=minimum(n.absolute,wanted_error);
        // Each positive panel has the GK roundoff estimate 50*eps*panel_tau.
        // An equal absolute-error allocation cannot pay that floor at a dense
        // endpoint of every subdivision. Do not demand sub-roundoff INTERNAL
        // relative accuracy; the final user tau/distance criteria below remain
        // mandatory, including the full returned quadrature error estimate.
        n.relative=maximum(64*DBL_EPSILON,minimum(n.relative,wanted_error/maximum(1,target)));
        const IntegralResult integral=detail::integrate_range(p,r,c,t,beta_per_m,n);
        result.evaluations+=integral.evaluations;
        if (integral.status != Status::Ok) { result.status=integral.status; return result; }
        const double residual=integral.tau-target;
        const double combined_error=::fabs(residual)+integral.estimated_error;
        // Convex radius along a straight line reaches its maximum at a local
        // bracket endpoint. The corresponding density is a lower bound in that
        // neighborhood, allowing a residual-to-distance error estimate.
        const double near_t=maximum(c.begin_m,t-distance_tolerance);
        const double far_t=minimum(c.end_m,t+distance_tolerance);
        const double max_radius=maximum(detail::radius_at(r,near_t),detail::radius_at(r,far_t));
        const double min_sigma=detail::sigma_at_height(max_radius-p.planetRadius,p.scaleHeight,beta_per_m);
        const double distance_error=min_sigma > 0 ? combined_error/min_sigma : HUGE_VAL;
        result.tau=integral.tau; result.tau_residual=residual;
        result.estimated_tau_error=integral.estimated_error;
        result.estimated_distance_error_m=distance_error;
        if (combined_error <= p.inverseTauTolerance && distance_error <= distance_tolerance) {
            result.scattered=true; result.endpoint=Endpoint::None;
            result.distance_m=t; result.ray_t=t/r.norm;
            result.transmittance=::exp(-integral.tau);
            result.pdf_per_m=sigma*result.transmittance; result.endpoint_mass=0;
            result.sigma_s_per_m=sigma;
            if (!finite(result.pdf_per_m)||!(result.pdf_per_m > 0)) result.status=Status::NumericalUnderflow;
            return result;
        }
        if (::fabs(residual) <= integral.estimated_error) {
            n.absolute*=0.0625; n.relative*=0.0625;
            continue; // An unresolved sign must not move the root bracket.
        }
        if (residual > 0) hi=t; else lo=t;
        const double next_newton=t-residual/sigma;
        const double margin=0.05*(hi-lo);
        const double next=finite(next_newton)&&next_newton > lo+margin&&next_newton < hi-margin ?
            next_newton : lo+0.5*(hi-lo);
        if (next == t || !(next > lo&&next < hi)) { result.status=Status::InverseRoundoff; return result; }
        t=next;
    }
    result.status=Status::InverseIterations;
    return result;
}

} // namespace m3

#undef M3_HD
