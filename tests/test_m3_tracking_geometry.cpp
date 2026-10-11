#include "m3/tracking_geometry.hpp"
#include "m3/atmosphere.hpp" // Independent CPU numerical reference only.

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
namespace g=m3::tracking;

void check(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
void ok(g::Status status,const char* message) {
    if(status!=g::Status::Ok)throw std::runtime_error(std::string(message)+": "+g::status_name(status));
}
long double exact(g::FloatPair p) { return static_cast<long double>(p.hi)+p.lo; }
g::FloatPair split(long double x) {
    const float hi=static_cast<float>(x);
    return {hi,static_cast<float>(x-hi)};
}
void near(long double actual,long double expected,long double absolute,long double relative,const char* message) {
    const long double error=std::abs(actual-expected);
    if(!(error<=absolute+relative*std::abs(expected))) {
        std::cerr<<std::setprecision(22)<<message<<": actual="<<actual<<" expected="<<expected
                 <<" error="<<error<<'\n';
        throw std::runtime_error(message);
    }
}

long double independent_height(const m3::TrackingParameters& p,g::PairVec3 point) {
    const long double x=exact(point.x),y=exact(point.y)-p.groundY,z=exact(point.z),r=p.planetRadius;
    // Independent long-double Cartesian evaluation. No production pair
    // arithmetic or compensated discriminant is used in the oracle.
    const long double radial=std::sqrt(x*x+(r+y)*(r+y)+z*z);
    return radial-r;
}
struct RefRoots { bool hit;long double near_t,far_t; };
RefRoots independent_roots(const m3::TrackingParameters& p,const g::RaySegment& ray,long double shell_height) {
    const long double x=ray.origin.x,y=static_cast<long double>(ray.origin.y)-p.groundY;
    const long double z=ray.origin.z,r=p.planetRadius;
    const long double dx=ray.direction.x,dy=ray.direction.y,dz=ray.direction.z;
    const long double a=dx*dx+dy*dy+dz*dz;
    const long double b=x*dx+(r+y)*dy+z*dz;
    const long double c=x*x+z*z+y*(2*r+y)-shell_height*(2*r+shell_height);
    const long double d=b*b-a*c;
    if(d<0)return {false,0,0};
    const long double root=std::sqrt(d);
    const long double q=-b-std::copysign(root,b);
    long double first=q==0 ? -b/a : q/a;
    long double second=q==0 ? first : c/q;
    if(first>second)std::swap(first,second);
    return {true,first,second};
}

void pair_progression() {
    g::FloatPair t{1000000.0f,0};
    const float step=1e-4f;
    check(t.hi+step==t.hi,"test must exercise a float addition that cannot advance");
    for(unsigned i=0;i<10000;++i) {
        const auto next=g::add(t,step);
        check(g::compare(next,t)>0,"a positive sub-ULP flight increment advances the pair");
        t=next;
    }
    // The two-float representation is finite precision, not an exact summation
    // of arbitrarily many draws. After 10000 steps it must preserve the metre
    // of travel to much less than one step (one float would lose all of it).
    near(exact(t),1000000.0L+10000*static_cast<long double>(step),1e-5L,0,
         "compensated progression keeps sub-ULP travel to below one tenth of a step");
    const auto shortened=g::sub(t,g::FloatPair{1000000.0f,0});
    near(exact(shortened),10000*static_cast<long double>(step),1e-5L,0,"local pair subtraction");
    const auto quotient=g::divide(g::mul(t,0.6f),0.6f);
    near(exact(quotient),exact(t),2e-8L,0,"product/division retain a long ray's low component");
    std::cout<<"METRIC pair_subulp_progress_m="<<exact(shortened)<<'\n';
}

void shell_metric_and_short() {
    m3::TrackingParameters p;
    const float inf=std::numeric_limits<float>::infinity();
    auto c=g::clip(p,{{0,p.groundY+1000,0},{0,-2,0},inf});
    ok(c.status,"raw-direction ground clip");
    check(c.has_medium&&c.hits_ground&&c.ground_occludes,"ground is opaque");
    near(exact(c.ground_parameter),500,1e-9L,0,"ground endpoint remains a raw ray parameter");
    near(c.direction_length,2,0,0,"non-unit direction length");
    c=g::clip(p,{{0,p.groundY+1000,0},{0,-2,0},500});
    ok(c.status,"finite ground endpoint");
    check(c.hits_ground&&!c.ground_occludes&&g::compare(c.end_parameter,500)==0,
          "the ground at the connection endpoint does not occlude it");
    c=g::clip(p,{{0,p.groundY,0},{0,-1,0},inf});
    ok(c.status,"inward ground ray");
    check(c.hits_ground&&c.ground_occludes&&!c.has_medium&&g::compare(c.ground_parameter,0)==0,
          "an inward ray on the ground is blocked at zero");
    c=g::clip(p,{{0,p.groundY,0},{0,1,0},inf});
    ok(c.status,"outward ground ray");
    check(c.has_medium&&!c.hits_ground,"outward boundary ray is not self occluded");
    near(exact(c.end_parameter),p.atmosphereHeight,1e-7L,0,"finite top exit");

    const g::RaySegment outside{{0,p.groundY+120000,0},{0,-2,0},55000};
    c=g::clip(p,outside);ok(c.status,"exterior camera");
    check(c.has_medium&&!c.hits_ground,"finite outside-to-atmosphere segment");
    near(exact(c.begin_parameter),10000,1e-7L,0,"exterior entry raw t");
    near(exact(c.end_parameter),55000,0,0,"finite endpoint retained exactly");
    c=g::clip(p,{{0,p.groundY+p.atmosphereHeight,0},{0,1,0},inf});
    ok(c.status,"top outward boundary");
    check(!c.has_medium,"top outward ray is vacuum");
    c=g::clip(p,{{0,p.groundY+120000,0},{1,0,0},inf});
    ok(c.status,"outside miss");check(!c.has_medium&&!c.hits_ground,"outside tangent misses atmosphere");

    const g::RaySegment raw{{123,p.groundY+10,17},{0.6f,0.8f,0},1e-4f};
    c=g::clip(p,raw);ok(c.status,"short nonunit-float ray");
    check(c.has_medium&&g::compare(c.end_parameter,raw.t_max)==0,"short raw endpoint is not displaced");
    const long double norm=std::sqrt(static_cast<long double>(raw.direction.x)*raw.direction.x+
                                    static_cast<long double>(raw.direction.y)*raw.direction.y);
    near(c.direction_length,norm,8e-8L,0,"float normalized direction retains its metric conversion");
    const auto endpoint=g::point_at(raw,c.end_parameter);
    near(exact(endpoint.x),static_cast<long double>(raw.origin.x)+
         static_cast<long double>(raw.direction.x)*raw.t_max,1e-12L,0,"short local x increment");
    check(exact(endpoint.x)!=raw.origin.x,"short increments survive before OptiX narrowing");

    // R+height is indistinguishable from R as one float in this case.
    p.atmosphereHeight=0.001f;
    check(p.planetRadius+p.atmosphereHeight==p.planetRadius,"thin-shell test requires lost float radius sum");
    c=g::clip(p,{{0,p.groundY,0},{0,1,0},inf});
    ok(c.status,"sub-ULP-radius shell");check(c.has_medium,"local-height shell is retained");
    near(exact(c.end_parameter),p.atmosphereHeight,1e-11L,0,"thin shell thickness remains in metres");
}

void altitude_values() {
    m3::TrackingParameters p;
    long double largest_error=0;
    for(long double requested:{0.0L,1e-6L,0.0001L,1.0L,1234.0L,80000.0L,100000.0L,120000.0L}) {
        const g::PairVec3 point{{},split(static_cast<long double>(p.groundY)+requested),{}};
        const auto h=g::height(p,point);ok(h.status,"radial local height");
        const auto reference=independent_height(p,point);
        near(h.height_m,reference,2e-7L,2.0e-7L,"stable radial height across scales");
        largest_error=std::max(largest_error,std::abs(static_cast<long double>(h.height_m)-reference));
    }
    for(long double x:{1000.0L,100000.0L,250000.0L,1000000.0L,4000000.0L}) {
        const long double r=p.planetRadius;
        const long double sag=x*x/(r+std::sqrt(r*r-x*x));
        const g::PairVec3 point{split(x),split(p.groundY-sag+0.0001L),{}};
        const auto h=g::height(p,point);ok(h.status,"far local near-ground height");
        near(h.height_m,independent_height(p,point),2e-7L,0,"far local cancellation preserves ground altitude");
    }
    std::cout<<"METRIC fp32_height_max_radial_error_m="<<largest_error<<'\n';
}

void grazing_regression() {
    m3::TrackingParameters p;
    unsigned count=0;long double largest_error=0;
    for(long double requested:{1e-5L,1e-4L,1e-3L,0.1L,1.0L,1000.0L}) {
        const float oy=static_cast<float>(p.groundY+requested);
        const long double h=static_cast<long double>(oy)-p.groundY,r=p.planetRadius;
        const long double c=h*(2*r+h);
        const float critical=static_cast<float>(-std::sqrt(c)/r);
        for(int offset=-10;offset<=10;++offset) {
            float dy=critical;
            for(int i=0;i<std::abs(offset);++i)
                dy=std::nextafter(dy,offset<0 ? -INFINITY : INFINITY);
            const g::RaySegment ray{{0,oy,0},{1,dy,0},INFINITY};
            const auto ref=independent_roots(p,ray,0);
            const auto clipped=g::clip(p,ray);ok(clipped.status,"near-tangent ray");
            check(clipped.hits_ground==ref.hit,"near-tangent support matches independent long double");
            if(ref.hit) {
                near(exact(clipped.ground_parameter),ref.near_t,2e-5L,2e-10L,"near-tangent root accuracy");
                largest_error=std::max(largest_error,std::abs(exact(clipped.ground_parameter)-ref.near_t));
                const auto hit=g::planet_hit(p,ray,clipped);ok(hit.status,"near-tangent planet representation");
                check(independent_height(p,g::paired(hit.position))>=0,"tangent ground point stays exterior");
            } else {
                check(clipped.has_medium,"near-tangent miss keeps the complete atmospheric interval");
                const auto top=independent_roots(p,ray,p.atmosphereHeight);
                near(exact(clipped.end_parameter),top.far_t,1e-5L,2e-10L,"grazing atmosphere exit");
            }
            ++count;
        }
    }
    const g::RaySegment contact{{-100000,p.groundY,0},{1,0,0},200000};
    const auto c=g::clip(p,contact);ok(c.status,"exact future tangent");
    check(c.hits_ground&&c.ground_occludes,"exact tangent remains a geometric contact");
    near(exact(c.ground_parameter),100000,0,0,"exact tangent location");
    std::cout<<"METRIC fp32_grazing_cases="<<count<<" disagreements=0 max_root_error_t="<<largest_error<<'\n';
}

void reported_planet_regression() {
    m3::TrackingParameters p;
    // Verbatim binary32-round-trip values from the user's PlanetIntersection
    // report: frame=0, pixel=(113,60), sample=0, bounce=2, seed=17.
    const g::RaySegment reported{{3374.39648f,14831.293f,-8108.75244f},
        {-0.702025175f,-0.673400521f,-0.231716275f},INFINITY};
    unsigned count=0;
    long double maximum_root_error=0;
    for(int changed=-1;changed<6;++changed) {
        for(int side=-1;side<=1;side+=2) {
            if(changed==-1&&side==1)continue;
            auto ray=reported;
            if(changed>=0) {
                float& component=changed==0 ? ray.origin.x : changed==1 ? ray.origin.y :
                    changed==2 ? ray.origin.z : changed==3 ? ray.direction.x :
                    changed==4 ? ray.direction.y : ray.direction.z;
                component=std::nextafter(component,side<0 ? -INFINITY : INFINITY);
            }
            const auto reference=independent_roots(p,ray,0);
            check(reference.hit&&reference.near_t>0,"reported planet ray has an independent forward root");
            const auto clipped=g::clip(p,ray);ok(clipped.status,"reported planet clip");
            check(clipped.hits_ground&&clipped.ground_occludes,"reported ray reaches the planet");
            near(exact(clipped.ground_parameter),reference.near_t,2e-5L,2e-10L,
                 "reported planet root matches independent long double");
            maximum_root_error=std::max(maximum_root_error,
                std::abs(exact(clipped.ground_parameter)-reference.near_t));
            const auto planet=g::planet_hit(p,ray,clipped);ok(planet.status,"reported planet representation");
            check(m1::finite(planet.position)&&m1::finite(planet.normal),"reported planet point and normal are finite");
            near(m1::length(planet.normal),1,2e-7L,0,"reported planet normal is unit length");
            check(independent_height(p,g::paired(planet.position))>=0,
                  "reported planet representation is independently exterior");
            const g::RaySegment outward{planet.position,planet.normal,INFINITY};
            const auto escape=g::clip(p,outward);ok(escape.status,"reported planet outward continuation");
            check(escape.has_medium&&!escape.hits_ground,"outward continuation escapes the planet");
            const g::RaySegment inward{planet.position,-planet.normal,INFINITY};
            const auto inward_reference=independent_roots(p,inward,0);
            const auto blocked=g::clip(p,inward);ok(blocked.status,"reported planet inward continuation");
            check(inward_reference.hit&&blocked.hits_ground&&blocked.ground_occludes,
                  "inward continuation is blocked by the planet");
            near(exact(blocked.ground_parameter),inward_reference.near_t,2e-5L,2e-10L,
                 "inward continuation endpoint matches independent long double");
            ++count;
        }
    }
    check(count==13,"reported ray and every one-ULP coordinate/direction neighbor were checked");
    std::cout<<"METRIC reported_planet_cases="<<count<<" max_root_error_t="<<maximum_root_error<<'\n';
}

void check_position(const m3::TrackingParameters& p,g::PairVec3 source,const g::PositionResult& result) {
    ok(result.status,"medium representation");
    check(independent_height(p,g::paired(result.position))>=0,"represented point is physically outside");
    const g::FloatPair s[3]={source.x,source.y,source.z};
    const long double radial[3]={exact(source.x),p.planetRadius+(exact(source.y)-p.groundY),exact(source.z)};
    const float before[3]={g::value(source.x),g::value(source.y),g::value(source.z)};
    const float after[3]={result.position.x,result.position.y,result.position.z};
    bool changed=false;
    for(unsigned i=0;i<3;++i)if(before[i]!=after[i]) {
        changed=true;
        const float expected=std::nextafter(before[i],radial[i]>0 ? INFINITY : -INFINITY);
        check(after[i]==expected,"medium representation moves at most one float step per coordinate");
        check(radial[i]*(static_cast<long double>(after[i])-before[i])>0,"changed coordinate moves outward");
        check((radial[i]>0&&before[i]<exact(s[i]))||(radial[i]<0&&before[i]>exact(s[i])),
              "already outward-rounded coordinates do not move");
    }
    check(changed==result.rounded_outward,"representation displacement is reported");
    for(const auto direction:std::array<g::Vec3,6>{{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}})
        ok(g::clip(p,{result.position,direction,1}).status,"represented point starts another ray");
}

void representation_regressions() {
    m3::TrackingParameters p;
    // Same actual M3 24-bit free-flight case that once failed on narrowing.
    m3::AtmosphereParameters reference;
    const m3::RaySegment original{{250000,-4902.92822265625,0},{0,-1,0},HUGE_VAL};
    double beta=0;
    check(m3::rayleigh_coefficient(reference,360,beta),"reference wavelength");
    const auto event=m3::sample_free_flight(reference,original,beta,(11870.0+0.5)/16777216.0);
    check(event.status==m3::Status::Ok&&event.scattered,"original real free-flight event");
    const g::RaySegment ray{{250000,-4902.92822265625f,0},{0,-1,0},INFINITY};
    const auto point=g::point_at(ray,split(event.ray_t));
    check(independent_height(p,point)>0&&independent_height(p,g::paired(g::value(point)))<0,
          "the original float boundary counterexample is reproduced");
    const auto represented=g::represent_medium_point(p,point);
    check_position(p,point,represented);check(represented.rounded_outward,"counterexample requires directed rounding");

    unsigned checked=0,corrected=0,planet_count=0;
    for(long double theta:{0.0L,0.25L,0.8L,1.4L,1.7L,2.6L,3.0L})
        for(long double phi:{0.0L,0.7L,1.6L,3.2L,4.8L}) {
            const long double r=static_cast<long double>(p.planetRadius)+0.0001L;
            const long double nx=std::sin(theta)*std::cos(phi),ny=std::cos(theta),nz=std::sin(theta)*std::sin(phi);
            const g::PairVec3 source{split(r*nx),split(p.groundY-p.planetRadius+r*ny),split(r*nz)};
            check(independent_height(p,source)>0,"constructed pair is outside");
            const auto result=g::represent_medium_point(p,source);check_position(p,source,result);
            corrected+=result.rounded_outward;++checked;
            const long double outer=static_cast<long double>(p.planetRadius)+1000;
            const g::RaySegment ground_ray{{float(outer*nx),float(p.groundY-p.planetRadius+outer*ny),float(outer*nz)},
                {float(-nx),float(-ny),float(-nz)},INFINITY};
            const auto c=g::clip(p,ground_ray);ok(c.status,"planet-direction clip");
            check(c.hits_ground,"inward radial direction hits the planet");
            const auto planet=g::planet_hit(p,ground_ray,c);ok(planet.status,"planet-direction representation");
            check(independent_height(p,g::paired(planet.position))>=0,"projected planet representation is outside");
            near(m1::length(planet.normal),1,2e-7L,0,"planet normal normalization");
            ++planet_count;
        }
    check(checked==35&&corrected>0&&planet_count==35,"all spherical quadrants were tested");
    std::cout<<"METRIC fp32_medium_position_directions="<<checked<<" corrected="<<corrected
             <<" planet_hits="<<planet_count<<'\n';
}

void invalid_inputs() {
    m3::TrackingParameters p;
    const auto inside=g::PairVec3{{},split(static_cast<long double>(p.groundY)-1e-6L),{}};
    check(g::represent_medium_point(p,inside).status==g::Status::InsideGround,"truly inside point is not repaired");
    bool inside_rounded_outside=false;
    for(long double x:{1000.0L,10000.0L,50000.0L,100000.0L,150000.0L,500000.0L,1000000.0L}) {
        const long double r=p.planetRadius,sag=x*x/(r+std::sqrt(r*r-x*x));
        const g::PairVec3 source{split(x),split(p.groundY-sag-1e-6L),{}};
        check(independent_height(p,source)<0,"invalid source is independently inside");
        inside_rounded_outside|=independent_height(p,g::paired(g::value(source)))>=0;
        check(g::represent_medium_point(p,source).status==g::Status::InsideGround,
              "an inside point with a plausible outside float image is still rejected");
    }
    check(inside_rounded_outside,"invalid tests include an outside-rounded interior point");
    const g::RaySegment ground_ray{{0,p.groundY+100,0},{0,-1,0},INFINITY};
    auto forged=g::clip(p,ground_ray);
    ok(forged.status,"valid ground point before malformed-endpoint test");
    forged.ground_parameter=g::add(forged.ground_parameter,1.0f);
    check(g::planet_hit(p,ground_ray,forged).status==g::Status::UnresolvedGeometry,
          "known-hit projection does not repair a point a metre inside the planet");
    const float nan=std::numeric_limits<float>::quiet_NaN();
    check(g::clip(p,{{0,0,0},{0,0,0},1}).status==g::Status::InvalidRay,"zero direction is invalid");
    check(g::clip(p,{{0,0,0},{0,1,0},nan}).status==g::Status::InvalidRay,"NaN endpoint is invalid");
    check(g::clip(p,{{nan,0,0},{0,1,0},1}).status==g::Status::InvalidRay,"NaN origin is invalid");
    check(g::height(p,g::Vec3{FLT_MAX,0,0}).status==g::Status::NonfiniteGeometry,"coordinate overflow fails");
    check(g::represent_medium_point(p,g::PairVec3{{nan,0},{},{}}).status==g::Status::NonfiniteGeometry,
          "nonfinite pair is rejected");
    p.planetRadius=0;
    check(g::clip(p,{{0,0,0},{0,1,0},1}).status==g::Status::InvalidParameters,"zero planet radius fails");
}
}

int main() {
    try {
        std::cout<<std::setprecision(15);
        pair_progression();shell_metric_and_short();altitude_values();grazing_regression();
        reported_planet_regression();representation_regressions();invalid_inputs();
        std::cout<<"FP32 tracking geometry checks passed: sub-ULP progress, spherical shell/raw ray metric, "
            "stable altitude, 126 grazing cases, 13 reported-ray cases, 35 spherical representations, "
            "and invalid-input rejection\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FP32 tracking geometry failed: "<<e.what()<<'\n';return 1;
    }
}
