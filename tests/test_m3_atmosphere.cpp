#include "m3/atmosphere.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void ok(m3::Status status,const char* operation) {
    if (status != m3::Status::Ok)
        throw std::runtime_error(std::string(operation)+": "+m3::status_name(status));
}

void near(double actual,long double expected,double relative,double absolute,const char* message) {
    const long double error=std::abs(static_cast<long double>(actual)-expected);
    if (!(error <= absolute+relative*std::abs(expected))) {
        std::cerr << std::setprecision(20) << message << ": actual=" << actual
                  << " expected=" << expected << " error=" << error << '\n';
        throw std::runtime_error(message);
    }
}

m3::AtmosphereParameters precise() {
    m3::AtmosphereParameters p;
    p.tauAbsoluteTolerance=1e-12;
    p.tauRelativeTolerance=1e-11;
    p.distanceAbsoluteTolerance=1e-5;
    return p;
}

long double vertical_tau(const m3::AtmosphereParameters& p,long double height,
                         long double distance,long double beta) {
    const long double h=p.scaleHeight;
    return beta*std::exp(-height/h)*h*(-std::expm1(-distance/h));
}

// Independent long-double composite Simpson integration in Cartesian distance.
// This does not use production clipping, outward pieces, radial-change formula,
// adaptive Gauss/Kronrod rule, or bounded-tail omission. Cases provide explicit
// finite intervals wholly in the atmosphere.
long double independent_tau(const m3::AtmosphereParameters& p,const m3::RaySegment& ray,
                            long double length_m,long double beta,unsigned panels=262144) {
    const long double norm=std::sqrt(static_cast<long double>(ray.direction.x)*ray.direction.x+
        static_cast<long double>(ray.direction.y)*ray.direction.y+
        static_cast<long double>(ray.direction.z)*ray.direction.z);
    const long double dx=ray.direction.x/norm,dy=ray.direction.y/norm,dz=ray.direction.z/norm;
    const long double radius=p.planetRadius;
    const long double step=length_m/panels;
    auto f=[&](long double s) {
        const long double x=ray.origin.x+dx*s;
        const long double y=static_cast<long double>(ray.origin.y)-p.groundY+radius+dy*s;
        const long double z=ray.origin.z+dz*s;
        const long double r=std::sqrt(x*x+y*y+z*z);
        return beta*std::exp(-(r-radius)/static_cast<long double>(p.scaleHeight));
    };
    long double sum=f(0)+f(length_m);
    for (unsigned i=1;i<panels;++i) sum+=(i%2 == 0 ? 2 : 4)*f(step*i);
    return sum*step/3;
}

void shell_and_metric() {
    const auto p=precise();
    const double infinity=std::numeric_limits<double>::infinity();
    auto c=m3::clip_atmosphere(p,{{0,p.groundY+1000,0},{0,-2,0},infinity});
    ok(c.status,"ground clip");
    check(c.has_medium&&c.hits_ground&&c.ground_occludes,"Ground is an opaque endpoint");
    near(c.ground_distance_m,1000,0,1e-9,"Ground distance in metres");
    near(c.direction_length,2,0,0,"Raw direction metric norm");
    near(c.end_m,1000,0,1e-9,"Clip stops at ground");
    c=m3::clip_atmosphere(p,{{0,p.groundY+1000,0},{0,-2,0},500});
    check(c.hits_ground&&!c.ground_occludes,"Ground at endpoint does not occlude that endpoint");
    c=m3::clip_atmosphere(p,{{0,p.groundY,0},{0,-1,0},infinity});
    check(c.hits_ground&&c.ground_distance_m == 0&&c.ground_occludes&&!c.has_medium,
          "Inward ray on ground is blocked at zero");
    c=m3::clip_atmosphere(p,{{0,p.groundY,0},{0,1,0},infinity});
    check(!c.hits_ground&&c.has_medium,"Outward ground ray is not self-occluded");
    near(c.end_m,p.atmosphereHeight,0,1e-8,"Outward atmosphere exit");
    c=m3::clip_atmosphere(p,{{0,p.groundY+p.atmosphereHeight+2000,0},{0,-1,0},infinity});
    near(c.begin_m,2000,0,1e-8,"Exterior atmosphere entry");
    near(c.end_m,p.atmosphereHeight+2000,0,1e-8,"Exterior ray stops at first ground");
    c=m3::clip_atmosphere(p,{{-100000,p.groundY,0},{1,0,0},200000});
    check(c.hits_ground&&c.ground_occludes,"Future tangent to opaque sphere is blocked");
    near(c.ground_distance_m,100000,0,1e-7,"Future tangent distance");
    const auto failed=m3::clip_atmosphere(p,{{0,p.groundY-1e-6,0},{0,1,0},10});
    check(failed.status == m3::Status::InsideGround,"Genuine inside-ground origin fails");

    // A float-normalized direction is not exactly unit when promoted to double.
    const float fx=0.6f,fy=0.8f;
    const m3::RaySegment raw{{0,p.groundY+1000,0},{fx,fy,0},10000};
    const double n=m3::length(raw.direction);
    const m3::RaySegment metric{raw.origin,raw.direction/n,raw.t_max*n};
    const auto a=m3::optical_depth(p,raw,p.beta550);
    const auto b=m3::optical_depth(p,metric,p.beta550);
    ok(a.status,"raw float direction");ok(b.status,"metric unit direction");
    near(a.tau,b.tau,3e-13,1e-14,"OptiX parameter to physical distance agreement");
}

void vertical_and_local_density() {
    const auto p=precise();
    for (double height : {0.0,1e-6,1234.0,10000.0}) {
        double sigma=-1;
        ok(m3::local_scattering_coefficient(p,{0,p.groundY+height,0},p.beta550,sigma),"local density");
        near(sigma,p.beta550*std::exp(-height/p.scaleHeight),3e-13,1e-18,"Exponential local density");
        for (double distance : {1e-6,0.25,100.0,20000.0}) {
            const auto integral=m3::optical_depth(p,{{0,p.groundY+height,0},{0,1,0},distance},p.beta550);
            ok(integral.status,"vertical integral");
            near(integral.tau,vertical_tau(p,height,distance,p.beta550),4e-11,2e-14,"Exact vertical optical depth");
            check(integral.estimated_error <= p.tauAbsoluteTolerance+p.tauRelativeTolerance*integral.tau,
                  "Successful integral satisfies its advertised estimated error");
        }
    }
    const auto down=m3::optical_depth(p,{{0,p.groundY+10000,0},{0,-1,0},8000},p.beta550);
    ok(down.status,"downward radial integral");
    near(down.tau,vertical_tau(p,2000,8000,p.beta550),4e-11,2e-14,"Downward radial reversal");
    const auto ground=m3::transmittance(p,{{0,p.groundY+1000,0},{0,-1,0},2000},p.beta550);
    ok(ground.status,"ground transmittance");
    check(ground.ground_occluded&&ground.transmittance > 0,
          "Opaque visibility is separate from nonzero transmittance to ground");
}

void tangent_short_and_additivity() {
    const auto p=precise();
    const m3::RaySegment cross{{-250000,p.groundY+800,0},{1,0,0},500000};
    const auto a=m3::optical_depth(p,cross,p.beta550);
    ok(a.status,"cross-tangent integral");
    const long double ref=independent_tau(p,cross,500000,p.beta550);
    near(a.tau,ref,3e-11,2e-12,"Cross-tangent independent Cartesian Simpson");
    std::cout << "METRIC tangent_tau=" << a.tau << " independent_tau=" << ref
              << " evaluations=" << a.evaluations << '\n';
    const m3::RaySegment short_ray{{123,p.groundY+10,17},{0.8,0.6,0},1e-4};
    const auto short_value=m3::optical_depth(p,short_ray,p.beta550);
    ok(short_value.status,"short interval");
    near(short_value.tau,independent_tau(p,short_ray,1e-4,p.beta550,1024),2e-10,1e-18,
         "Short interval avoids primitive subtraction");
    const double split=177123;
    const m3::RaySegment first{cross.origin,cross.direction,split};
    const m3::RaySegment second{cross.origin+cross.direction*split,cross.direction,cross.t_max-split};
    const m3::RaySegment reversed{cross.origin+cross.direction*cross.t_max,{-1,0,0},cross.t_max};
    const auto b=m3::optical_depth(p,first,p.beta550),c=m3::optical_depth(p,second,p.beta550);
    const auto d=m3::optical_depth(p,reversed,p.beta550);
    ok(b.status,"additivity first");ok(c.status,"additivity second");ok(d.status,"reverse");
    near(a.tau,b.tau+c.tau,4e-11,2e-12,"Optical-depth segment additivity");
    near(a.tau,d.tau,4e-11,2e-12,"Endpoint symmetry");
    near(std::exp(-a.tau),std::exp(-b.tau)*std::exp(-c.tau),4e-11,2e-14,
         "Transmittance multiplicativity");
    const m3::RaySegment horizon{{0,p.groundY,0},{1,0,0},500000};
    const auto h=m3::optical_depth(p,horizon,p.beta550);
    ok(h.status,"ground horizon");
    near(h.tau,independent_tau(p,horizon,500000,p.beta550),3e-11,2e-12,"Grazing ground integral");
}

void sampling_and_refinement() {
    auto p=precise();
    const m3::RaySegment ray{{0,p.groundY+1000,0},{0,2,0},15000};
    const long double total=vertical_tau(p,1000,30000,p.beta550);
    for (double fraction : {1e-5,0.03,0.4,0.95,0.999999}) {
        const double target=static_cast<double>(total)*fraction;
        const double xi=-std::expm1(-target);
        const auto sample=m3::sample_free_flight(p,ray,p.beta550,xi);
        ok(sample.status,"free-flight inverse");
        check(sample.scattered&&sample.endpoint == m3::Endpoint::None,"Sample is a collision");
        const long double prefactor=static_cast<long double>(p.beta550)*p.scaleHeight*
            std::exp(-1000.0L/p.scaleHeight);
        const long double expected=-p.scaleHeight*std::log1p(-static_cast<long double>(target)/prefactor);
        near(sample.distance_m,expected,2e-10,2e-5,"Exact vertical inverse distance");
        near(sample.ray_t,sample.distance_m/2,0,0,"Sample parameter/metric distance conversion");
        near(static_cast<double>(vertical_tau(p,1000,sample.distance_m,p.beta550)),target,1e-10,2e-10,
             "Independent inverse optical-depth residual");
        near(sample.pdf_per_m,sample.sigma_s_per_m*sample.transmittance,1e-14,0,"Continuous event PDF");
        check(sample.endpoint_mass == 0,"Collision does not carry a discrete endpoint mass");
    }
    const auto no_event=m3::sample_free_flight(p,ray,p.beta550,0.99);
    ok(no_event.status,"free-flight endpoint");
    check(!no_event.scattered&&no_event.endpoint == m3::Endpoint::SegmentEnd,"No-event endpoint");
    near(no_event.endpoint_mass,std::exp(-total),3e-11,1e-13,"Survival endpoint mass");
    check(no_event.pdf_per_m == 0,"Endpoint is not assigned a continuous PDF");
    const auto zero=m3::sample_free_flight(p,ray,p.beta550,0);
    ok(zero.status,"zero quantile");
    check(zero.scattered&&zero.distance_m == 0&&zero.transmittance == 1,"Zero quantile is interval entry");
    const auto vacuum=m3::sample_free_flight(p,ray,0,0);
    ok(vacuum.status,"vacuum zero quantile");
    check(!vacuum.scattered&&vacuum.endpoint_mass == 1,"True vacuum stays distinguishable");

    // Both targets start within the deliberately loose initial error budget.
    // Correct endpoint classification requires tightening that same integral.
    p.tauAbsoluteTolerance=1e-5;p.tauRelativeTolerance=1e-4;
    for (double sign : {-1.0,1.0}) {
        const double target=static_cast<double>(total)+sign*2e-11;
        const auto sample=m3::sample_free_flight(p,ray,p.beta550,-std::expm1(-target));
        ok(sample.status,"near-endpoint refinement");
        check(sample.scattered == (sign < 0),"Near-endpoint classification agrees with analytic depth");
    }
    const double thin_beta=1e-18;
    const double thin_target=static_cast<double>(vertical_tau(p,1000,10000,thin_beta));
    const auto thin=m3::sample_free_flight(p,ray,thin_beta,-std::expm1(-thin_target));
    ok(thin.status,"thin positive-medium refinement");
    check(thin.scattered,"A small positive medium is not silently treated as vacuum");
    near(thin.distance_m,10000,1e-8,2e-5,"Thin-medium exact inverse");
}

void explicit_failures() {
    auto p=precise();
    const m3::RaySegment ray{{0,p.groundY,0},{0,1,0},100000};
    const double nan=std::numeric_limits<double>::quiet_NaN();
    check(m3::optical_depth(p,ray,nan).status == m3::Status::InvalidCoefficient,"NaN coefficient fails");
    check(m3::sample_free_flight(p,ray,p.beta550,1).status == m3::Status::InvalidQuantile,"xi=1 fails");
    check(m3::sample_free_flight(p,ray,p.beta550,nan).status == m3::Status::InvalidQuantile,"NaN xi fails");
    check(m3::clip_atmosphere(p,{ray.origin,{0,0,0},1}).status == m3::Status::InvalidRay,"Zero direction fails");
    p.maxQuadratureEvaluations=15;
    check(m3::optical_depth(p,ray,p.beta550).status == m3::Status::QuadratureBudget,"Work-budget exhaustion fails");
    p=precise();p.maxInverseIterations=1;
    check(m3::sample_free_flight(p,ray,p.beta550,0.05).status == m3::Status::InverseIterations,
          "Unconverged inverse is not returned as a successful sample");
    p=precise();p.maxQuadratureDepth=25;
    check(m3::validate_parameters(p) == m3::Status::InvalidParameters,"Fixed stack capacity is validated");
}

void high_altitude_precision() {
    m3::AtmosphereParameters p;
    const m3::RaySegment ray{{0,p.groundY,0},{0,1,0},HUGE_VAL};
    double largest_distance_error=0,largest_estimate=0;
    for (double wavelength : {360.0,550.0,830.0}) {
        double beta=0;
        check(m3::rayleigh_coefficient(p,wavelength,beta),"Wavelength coefficient");
        for (double height : {80000.0,90000.0,95000.0,99000.0,99900.0,99990.0}) {
            const double target=static_cast<double>(vertical_tau(p,0,height,beta));
            const auto sample=m3::sample_free_flight(p,ray,beta,-std::expm1(-target));
            ok(sample.status,"rare high-altitude free flight");
            check(sample.scattered,"High-altitude quantile must collide before atmosphere top");
            const double tolerance=p.distanceAbsoluteTolerance+p.distanceRelativeTolerance*p.atmosphereHeight;
            near(sample.distance_m,height,0,tolerance,"High-altitude analytic inverse distance");
            check(std::abs(sample.tau_residual)+sample.estimated_tau_error <= p.inverseTauTolerance,
                  "High-altitude sample satisfies final optical-depth criterion");
            check(sample.estimated_distance_error_m <= tolerance,
                  "High-altitude sample satisfies final distance criterion");
            largest_distance_error=std::max(largest_distance_error,std::abs(sample.distance_m-height));
            largest_estimate=std::max(largest_estimate,sample.estimated_distance_error_m);
        }
    }
    // A user can ask for 10 micrometres, but the estimated quadrature roundoff
    // at this near-top event exceeds that distance budget. Internal precision
    // safeguards must NOT turn this genuinely unresolved request into success.
    p.distanceAbsoluteTolerance=1e-5;
    const double target=static_cast<double>(vertical_tau(p,0,99900,p.beta550));
    const auto strict=m3::sample_free_flight(p,ray,p.beta550,-std::expm1(-target));
    check(strict.status == m3::Status::InverseIterations || strict.status == m3::Status::InverseRoundoff,
          "Unachievable user distance accuracy fails explicitly");
    check(strict.estimated_distance_error_m >
          p.distanceAbsoluteTolerance+p.distanceRelativeTolerance*p.atmosphereHeight,
          "Strict failure retains the unresolved distance estimate");
    std::cout << "METRIC high_altitude_max_distance_error_m=" << largest_distance_error
              << " high_altitude_max_estimate_m=" << largest_estimate << '\n';
}

void grazing_discriminant() {
    const auto p=precise();
    unsigned checked=0;
    for (double requested_height : {1e-5,1e-4,1e-3,0.1,1.0,1000.0}) {
        const float origin_y=static_cast<float>(p.groundY+requested_height);
        const long double height=static_cast<long double>(origin_y)-p.groundY;
        const long double radius=p.planetRadius;
        const long double c=height*(2*radius+height);
        const float critical=static_cast<float>(-std::sqrt(c)/radius);
        for (int offset=-10;offset<=10;++offset) {
            float dy=critical;
            for (int i=0;i<std::abs(offset);++i)
                dy=std::nextafter(dy,offset < 0 ? -INFINITY : INFINITY);
            const long double a=1+static_cast<long double>(dy)*dy;
            const long double b=(radius+height)*dy;
            const long double discriminant=b*b-a*c;
            const bool reference_hit=discriminant >= 0;
            const m3::RaySegment ray{{0,origin_y,0},{1,dy,0},HUGE_VAL};
            const auto clip=m3::clip_atmosphere(p,ray);
            ok(clip.status,"near-tangent clipping");
            check(clip.hits_ground == reference_hit,
                  "Ground support agrees with independent long-double discriminant");
            if (reference_hit) {
                const long double first_t=(-b-std::sqrt(discriminant))/a;
                near(clip.ground_distance_m,first_t*std::sqrt(a),1e-11,1e-6,
                     "Near-tangent ground distance");
            } else {
                // The closest altitude can be far below one double ULP of R.
                // Both the clipping and the two integration pieces must retain
                // its positive sign; a rounded impact-R would cut this path in
                // half or incorrectly report an inside-ground error.
                const double physical_length=static_cast<double>(-2*b/std::sqrt(a)+1);
                const m3::RaySegment finite_ray{ray.origin,ray.direction,
                    physical_length/m3::length(ray.direction)};
                const auto integral=m3::optical_depth(p,finite_ray,p.beta550);
                ok(integral.status,"positive sub-ULP closest altitude");
                near(integral.tau,independent_tau(p,finite_ray,physical_length,p.beta550,2048),
                     8e-11,2e-13,"Near-tangent miss integrates the full interval");
            }
            ++checked;
        }
    }
    std::cout << "METRIC near_tangent_float_direction_cases=" << checked << " disagreements=0\n";
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(17);
        shell_and_metric();
        vertical_and_local_density();
        tangent_short_and_additivity();
        sampling_and_refinement();
        high_altitude_precision();
        grazing_discriminant();
        explicit_failures();
        std::cout << "M3 atmosphere: shell/ground, metric conversion, analytic radial, independent tangent/short, "
                     "additivity, inverse-CDF and explicit-failure checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
