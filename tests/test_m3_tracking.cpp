#include "m3/tracking.hpp"
#include "m3/transport.hpp"
#include "m3_tracking_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace tracking = m3::tracking;
using tracking::Status;

constexpr std::uint32_t sample_count = 65536;
constexpr unsigned cdf_bin_count = 8;
// Allow for FP32 geometry/rates/elementary functions independently of sampling
// error. This is an explicit test tolerance, not a claim of a certified bound.
constexpr double probability_allowance = 3e-5;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_ok(Status status, const std::string& operation) {
    if (status != Status::Ok)
        throw std::runtime_error(operation+": status="+
                                 std::to_string(static_cast<unsigned>(status)));
}

void near(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected
                  << " tolerance=" << tolerance << '\n';
        throw std::runtime_error(message);
    }
}

double six_sigma(double variance, std::uint32_t count, double rounding_allowance) {
    return 6*std::sqrt(std::max(0.0,variance)/count)+rounding_allowance;
}

struct OpenRng {
    std::mt19937 engine;
    std::uint64_t calls = 0;
    explicit OpenRng(std::uint32_t seed) : engine(seed) {}
    float operator()() {
        ++calls;
        // Each midpoint is exactly representable in float; neither 0 nor 1
        // can occur. The integer generator is independent of production PCG.
        return (float(engine()>>9u)+0.5f)*(1.0f/8388608.0f);
    }
};

struct ConstantRng {
    float value;
    std::uint64_t calls = 0;
    float operator()() { ++calls; return value; }
};

struct SequenceRng {
    std::vector<float> values;
    std::size_t next = 0;
    float operator()() {
        if (next == values.size()) throw std::runtime_error("deterministic RNG sequence exhausted");
        return values[next++];
    }
};

struct Moments {
    std::uint64_t count = 0;
    double mean = 0, m2 = 0, square_sum = 0;
    void add(double value) {
        ++count;
        const double delta=value-mean;
        mean+=delta/double(count);
        m2+=delta*(value-mean);
        square_sum+=value*value;
    }
    double variance() const { return count > 1 ? m2/double(count-1) : 0; }
    double second_moment() const { return square_sum/double(count); }
};

m3::TrackingParameters parameters() {
    m3::TrackingParameters p;
    p.enabled=1;
    p.ratioRrThreshold=0;
    return p;
}

tracking_reference::Atmosphere reference_atmosphere(const m3::TrackingParameters& p) {
    return {p.planetRadius,p.scaleHeight,p.groundY};
}

tracking_reference::Ray reference_ray(const tracking::RaySegment& ray,
    long double begin_m, long double end_m) {
    return {{ray.origin.x,ray.origin.y,ray.origin.z},
            {ray.direction.x,ray.direction.y,ray.direction.z},begin_m,end_m};
}

double metric_distance(tracking::FloatPair parameter, long double direction_length) {
    return static_cast<double>((static_cast<long double>(parameter.hi)+parameter.lo)*direction_length);
}

struct Case {
    const char* name;
    tracking::RaySegment ray;
    long double begin_m, end_m;
    bool vertical;
};

void test_coefficient_and_random_contract() {
    auto p=parameters();
    for (const float wavelength : {360.0f,550.0f,830.0f}) {
        float actual=0;
        require(tracking::rayleigh_coefficient(p,wavelength,actual),"valid Rayleigh coefficient");
        const double expected=double(p.beta550)*std::pow(550.0/double(wavelength),4);
        near(actual,expected,2e-6*expected,"FP32 wavelength-dependent sea-level coefficient");
    }
    float beta=1;
    p.beta550=0;
    require(tracking::rayleigh_coefficient(p,550,beta)&&beta==0,"zero sea-level coefficient remains vacuum");
    p=parameters();
    for (const float invalid : {0.0f,-1.0f,std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()})
        require(!tracking::rayleigh_coefficient(p,invalid,beta),"invalid wavelength is observable");
    OpenRng random(0x44a73ac9u);
    for (unsigned i=0;i<100000;++i) {
        const float value=random();
        require(value>0&&value<1,"test RNG is strictly open in float");
    }
}

void test_distribution_case(const m3::TrackingParameters& p, const Case& item,
    float wavelength, std::uint32_t seed) {
    float beta=0;
    require(tracking::rayleigh_coefficient(p,wavelength,beta),"statistical case coefficient");
    const auto rr=reference_ray(item.ray,item.begin_m,item.end_m);
    const tracking_reference::ExponentialRay reference(reference_atmosphere(p),rr,beta);
    const auto moments=reference.ratio_moments();
    const double target=double(moments[1]);
    const double collision_mass=1-target;
    const std::string label=std::string(item.name)+"/"+std::to_string(int(wavelength))+"nm";

    if (item.vertical) {
        const long double start_height=reference.height(item.begin_m);
        const auto exact=tracking_reference::vertical_tau(start_height,
            item.end_m-item.begin_m,p.scaleHeight,beta);
        near(double(reference.tau()),double(exact),2e-11,
             label+" independent Cartesian integral agrees with vertical analytic tau");
    }

    std::array<double,cdf_bin_count-1> boundaries{}, cdf{};
    for (unsigned i=0;i<boundaries.size();++i) {
        const long double distance=reference.collision_quantile(
            static_cast<long double>(i+1)/cdf_bin_count);
        boundaries[i]=double(distance);
        cdf[i]=double(-std::expm1(-reference.tau(distance)));
    }
    std::array<std::uint32_t,cdf_bin_count-1> cumulative{};
    std::array<std::uint32_t,cdf_bin_count> bins{};
    std::uint32_t escaped=0;
    std::uint64_t delta_candidates=0, nulls=0, ratio_candidates=0;
    OpenRng delta_random(seed), ratio_random(seed^0xa742bf51u);
    Moments attenuation;
    const long double direction_length=rr.direction_length();
    const double distance_allowance=2e-4+2e-7*double(item.end_m);
    for (std::uint32_t i=0;i<sample_count;++i) {
        const auto flight=tracking::delta_free_flight(p,item.ray,beta,delta_random);
        require_ok(flight.status,label+" delta");
        delta_candidates+=flight.candidates;
        nulls+=flight.null_collisions;
        require(flight.candidates==flight.null_collisions+(flight.scattered?1u:0u),
                label+" null collisions are not real scattering vertices");
        if (flight.scattered) {
            require(flight.endpoint==tracking::Endpoint::None,label+" collision is not an endpoint");
            const double distance=metric_distance(flight.parameter,direction_length);
            require(distance>=double(item.begin_m)-distance_allowance&&
                    distance<=double(item.end_m)+distance_allowance,
                    label+" event lies inside the independently specified medium interval");
            unsigned bin=0;
            while (bin<boundaries.size()&&distance>boundaries[bin]) ++bin;
            ++bins[bin];
            for (unsigned j=0;j<boundaries.size();++j)
                if (distance<=boundaries[j]) ++cumulative[j];
        } else {
            require(flight.endpoint==tracking::Endpoint::SegmentEnd,
                    label+" finite unobstructed ray reaches its geometric endpoint");
            near(metric_distance(flight.parameter,direction_length),double(item.end_m),
                 distance_allowance,label+" endpoint keeps raw OptiX parameter units");
            ++escaped;
        }
        const auto ratio=tracking::ratio_transmittance(p,item.ray,beta,ratio_random);
        require_ok(ratio.status,label+" ratio");
        require(!ratio.ground_occluded,label+" explicit reference interval is not planet-occluded");
        require(ratio.transmittance>=0&&ratio.transmittance<=1&&
                std::isfinite(ratio.transmittance),label+" ratio sample is finite and bounded");
        require(ratio.roulette_tests==0&&!ratio.roulette_terminated,label+" moments use roulette disabled");
        ratio_candidates+=ratio.candidates;
        attenuation.add(ratio.transmittance);
    }

    const double escape_fraction=double(escaped)/sample_count;
    near(escape_fraction,target,six_sigma(target*(1-target),sample_count,probability_allowance),
         label+" endpoint survival probability");
    double maximum_cdf_sigma=0;
    for (unsigned i=0;i<boundaries.size();++i) {
        const double actual=double(cumulative[i])/sample_count;
        const double standard_error=std::sqrt(cdf[i]*(1-cdf[i])/sample_count);
        near(actual,cdf[i],6*standard_error+probability_allowance,label+" unconditional free-flight CDF");
        maximum_cdf_sigma=std::max(maximum_cdf_sigma,
            std::abs(actual-cdf[i])/(standard_error+probability_allowance/6));
    }
    for (unsigned i=0;i<cdf_bin_count;++i) {
        const double lo=i==0?0:cdf[i-1];
        const double hi=i+1==cdf_bin_count?collision_mass:cdf[i];
        const double probability=hi-lo;
        near(double(bins[i])/sample_count,probability,
             six_sigma(probability*(1-probability),sample_count,probability_allowance),
             label+" collision-bin mass including the separate endpoint mass");
    }
    const double expected_second=double(moments[2]);
    const double expected_variance=std::max(0.0,expected_second-target*target);
    near(attenuation.mean,target,six_sigma(expected_variance,sample_count,probability_allowance),
         label+" ratio first moment");
    const double second_variance=std::max(0.0,double(moments[4])-expected_second*expected_second);
    near(attenuation.second_moment(),expected_second,
         six_sigma(second_variance,sample_count,probability_allowance),label+" ratio second moment");
    const double candidate_mean=double(reference.full_candidate_mean());
    near(double(ratio_candidates)/sample_count,candidate_mean,
         six_sigma(candidate_mean,sample_count,2e-4),label+" Poisson candidate count to the endpoint");

    std::cout << "case=" << label << " N=" << sample_count << " tau=" << double(reference.tau())
              << " expected_T=" << target << " escaped=" << escape_fraction
              << " ratio_mean=" << attenuation.mean << " ratio_variance=" << attenuation.variance()
              << " expected_ratio_variance=" << expected_variance
              << " max_CDF_sigma=" << maximum_cdf_sigma
              << " delta_candidates=" << double(delta_candidates)/sample_count
              << " delta_nulls=" << double(nulls)/sample_count
              << " ratio_candidates=" << double(ratio_candidates)/sample_count << '\n';
}

void test_distributions() {
    const auto p=parameters();
    const std::array<Case,4> cases{{
        {"vertical_6km",{{0,p.groundY+250,0},{0,1,0},6000},0,6000,true},
        {"grazing_80km",{{0,p.groundY+2,0},{1,0,0},80000},0,80000,false},
        {"high_20km_grazing_300km",{{0,p.groundY+20000,0},{1,0,0},300000},0,300000,false},
        {"outside_down_to_10km",{{0,p.groundY+120000,0},{0,-1,0},110000},20000,110000,false}
    }};
    std::uint32_t seed=0x3d287b10u;
    for (const auto& item : cases)
        for (const float wavelength : {360.0f,550.0f,830.0f}) {
            test_distribution_case(p,item,wavelength,seed);
            seed+=0x6d2b79f5u;
        }
}

void test_vacuum_ground_and_rejections() {
    auto p=parameters();
    const float infinity=std::numeric_limits<float>::infinity();
    const float nan=std::numeric_limits<float>::quiet_NaN();
    const tracking::RaySegment upward{{0,p.groundY+100,0},{0,1,0},500};
    for (const auto ray : std::array<tracking::RaySegment,3>{{upward,
            {{0,p.groundY+200000,0},{0,1,0},500},
            {{0,p.groundY+100,0},{0,1,0},0}}}) {
        ConstantRng unused{nan};
        const auto flight=tracking::delta_free_flight(p,ray,0,unused);
        const auto ratio=tracking::ratio_transmittance(p,ray,0,unused);
        require_ok(flight.status,"vacuum delta"); require_ok(ratio.status,"vacuum ratio");
        require(!flight.scattered&&flight.endpoint==tracking::Endpoint::SegmentEnd&&
                flight.candidates==0&&ratio.transmittance==1&&ratio.candidates==0&&unused.calls==0,
                "vacuum/empty interval consumes no RNG and has exact unit transmittance");
    }
    // A genuinely absent atmosphere interval must skip RNG even with beta>0.
    ConstantRng outside_random{nan};
    const tracking::RaySegment outside{{0,p.groundY+200000,0},{0,1,0},500};
    const auto outside_flight=tracking::delta_free_flight(p,outside,p.beta550,outside_random);
    const auto outside_ratio=tracking::ratio_transmittance(p,outside,p.beta550,outside_random);
    require_ok(outside_flight.status,"exterior delta");require_ok(outside_ratio.status,"exterior ratio");
    require(!outside_flight.scattered&&outside_ratio.transmittance==1&&outside_random.calls==0,
            "outward exterior ray has no fictitious density");

    for (const auto ray : std::array<tracking::RaySegment,2>{{
            {{0,p.groundY+100,0},{0,-1,0},200},{{0,p.groundY,0},{0,-1,0},1}}}) {
        ConstantRng unused{nan};
        const auto flight=tracking::delta_free_flight(p,ray,0,unused);
        const auto ratio=tracking::ratio_transmittance(p,ray,0,unused);
        require_ok(flight.status,"opaque ground delta");require_ok(ratio.status,"opaque ground ratio");
        require(!flight.scattered&&flight.endpoint==tracking::Endpoint::Ground&&
                ratio.ground_occluded&&ratio.transmittance==0&&unused.calls==0,
                "planet occlusion remains geometric even with zero scattering");
    }
    ConstantRng boundary_random{nan};
    const tracking::RaySegment exact_ground{{0,p.groundY+100,0},{0,-1,0},100};
    const auto at_ground=tracking::ratio_transmittance(p,exact_ground,0,boundary_random);
    require_ok(at_ground.status,"ground at segment endpoint");
    require(!at_ground.ground_occluded&&at_ground.transmittance==1,
            "ground exactly at endpoint does not occlude an earlier connection");
    const tracking::RaySegment solar{{0,p.groundY+1,0},{0,1,0},infinity};
    const auto solar_exit=tracking::delta_free_flight(p,solar,0,boundary_random);
    require_ok(solar_exit.status,"vacuum solar exit");
    require(solar_exit.endpoint==tracking::Endpoint::AtmosphereExit,
            "unbounded solar ray has an atmosphere exit endpoint");
    near(metric_distance(solar_exit.parameter,1),double(p.atmosphereHeight)-1,0.02,
         "solar exit distance in metres");
    require(boundary_random.calls==0,"boundary-only handling does not sample the medium");

    for (const float invalid : {0.0f,1.0f,-0.1f,1.1f,nan,infinity}) {
        ConstantRng a{invalid},b{invalid};
        require(tracking::delta_free_flight(p,upward,p.beta550,a).status==Status::InvalidRandom,
                "delta rejects invalid distance RNG rather than clamping");
        require(tracking::ratio_transmittance(p,upward,p.beta550,b).status==Status::InvalidRandom,
                "ratio rejects invalid distance RNG rather than clamping");
    }
    SequenceRng invalid_acceptance{{0.001f,1.0f}};
    require(tracking::delta_free_flight(p,upward,0.01f,invalid_acceptance).status==Status::InvalidRandom,
            "delta validates the acceptance RNG as well as the distance RNG");
    auto roulette_parameters=p;
    roulette_parameters.ratioRrThreshold=1;
    SequenceRng invalid_roulette{{0.001f,0.0f}};
    require(tracking::ratio_transmittance(roulette_parameters,upward,0.01f,invalid_roulette).status==
            Status::InvalidRandom,"ratio validates its roulette RNG");

    for (const float invalid : {-1.0f,nan,infinity}) {
        ConstantRng a{0.5f},b{0.5f};
        require(tracking::delta_free_flight(p,upward,invalid,a).status==Status::InvalidCoefficient,
                "invalid extinction is rejected by delta");
        require(tracking::ratio_transmittance(p,upward,invalid,b).status==Status::InvalidCoefficient,
                "invalid extinction is rejected by ratio");
    }
    for (const auto ray : std::array<tracking::RaySegment,4>{{
            {{0,p.groundY+100,0},{0,0,0},1},{{nan,0,0},{0,1,0},1},
            {{0,p.groundY+100,0},{0,1,0},-1},{{0,p.groundY+100,0},{0,1,0},nan}}}) {
        ConstantRng a{0.5f},b{0.5f};
        require(tracking::delta_free_flight(p,ray,p.beta550,a).status==Status::InvalidRay,
                "delta rejects invalid ray geometry");
        require(tracking::ratio_transmittance(p,ray,p.beta550,b).status==Status::InvalidRay,
                "ratio rejects invalid ray geometry");
    }
    const tracking::RaySegment inside{{0,p.groundY-1,0},{0,1,0},10};
    ConstantRng inside_random{0.5f};
    require(tracking::delta_free_flight(p,inside,p.beta550,inside_random).status==Status::InsideGround,
            "true inside-ground origins are not repaired as if valid");

    auto limited=p;
    limited.maxTrackingEvents=1;
    const tracking::RaySegment dense_candidates{{0,p.groundY+80000,0},{1,0,0},1000};
    ConstantRng nulls{0.5f},weights{0.5f};
    const auto exhausted_delta=tracking::delta_free_flight(limited,dense_candidates,10,nulls);
    const auto exhausted_ratio=tracking::ratio_transmittance(limited,dense_candidates,10,weights);
    require(exhausted_delta.status==Status::TrackingLimit&&exhausted_delta.candidates==1&&
            !exhausted_delta.scattered,"delta event limit rejects an unfinished flight");
    require(exhausted_ratio.status==Status::TrackingLimit&&exhausted_ratio.candidates==1,
            "ratio event limit does not accept a truncated product");
    limited.maxTrackingEvents=0;
    ConstantRng invalid_budget{0.5f};
    require(tracking::delta_free_flight(limited,upward,p.beta550,invalid_budget).status==
            Status::InvalidParameters,"zero tracking budget is invalid");
    std::cout << "deterministic: vacuum, exterior, planet, endpoint, raw RNG, invalid coefficient/ray, budget passed\n";
}

void test_raw_ray_parameter() {
    const auto p=parameters();
    const tracking::RaySegment a{{0,p.groundY+250,0},{0,1,0},6000};
    const tracking::RaySegment b{a.origin,{0,2,0},3000};
    OpenRng da(0x37e14061u),db(0x37e14061u),ra(0xdda82943u),rb(0xdda82943u);
    for (unsigned i=0;i<2048;++i) {
        const auto fa=tracking::delta_free_flight(p,a,0.0001f,da);
        const auto fb=tracking::delta_free_flight(p,b,0.0001f,db);
        require_ok(fa.status,"raw direction delta a");require_ok(fb.status,"raw direction delta b");
        require(fa.scattered==fb.scattered&&fa.endpoint==fb.endpoint,
                "rescaling a raw ray does not change its physical event");
        near(metric_distance(fa.parameter,1),metric_distance(fb.parameter,2),1e-5,
             "raw t is converted through the direction norm");
        const auto ta=tracking::ratio_transmittance(p,a,0.0001f,ra);
        const auto tb=tracking::ratio_transmittance(p,b,0.0001f,rb);
        require_ok(ta.status,"raw direction ratio a");require_ok(tb.status,"raw direction ratio b");
        near(ta.transmittance,tb.transmittance,1e-7,"ratio samples are invariant to power-of-two raw ray scaling");
    }
}

void test_roulette_mean() {
    auto p=parameters();
    p.ratioRrThreshold=m3::TrackingParameters{}.ratioRrThreshold;
    require(p.ratioRrThreshold==0.01f,"regression exercises the documented default roulette threshold");
    // A nearly homogeneous, dilute layer with a loose global majorant makes
    // low products common without making the target mean practically invisible.
    // beta is deliberately larger than the Earth preset to test the estimator.
    const tracking::RaySegment ray{{0,p.groundY+20000,0},{1,0,0},1000};
    constexpr float beta=0.06f;
    const auto rr=reference_ray(ray,0,1000);
    const tracking_reference::ExponentialRay reference(reference_atmosphere(p),rr,beta);
    const double expected=double(reference.transmittance());
    OpenRng random(0x43852b17u);
    Moments samples,hard_cutoff;
    std::uint64_t roulette_tests=0,terminated=0,candidates=0;
    for (std::uint32_t i=0;i<sample_count;++i) {
        const auto sample=tracking::ratio_transmittance(p,ray,beta,random);
        require_ok(sample.status,"default roulette transmittance");
        require(sample.transmittance>=0&&sample.transmittance<=1,
                "roulette transmittance remains bounded");
        roulette_tests+=sample.roulette_tests;
        terminated+=sample.roulette_terminated?1u:0u;
        candidates+=sample.candidates;
        samples.add(sample.transmittance);
        // Before its first roulette decision, this is the same process as a
        // wrong implementation that discards every sub-threshold product.
        // Its lost contribution quantifies this regression's detection power.
        hard_cutoff.add(sample.roulette_tests==0?sample.transmittance:0);
    }
    require(roulette_tests>sample_count/4&&terminated>sample_count/8,
            "roulette test actually reaches and terminates low-product paths");
    const double tolerance=six_sigma(samples.variance(),sample_count,probability_allowance);
    near(samples.mean,expected,tolerance,"default roulette preserves expected transmittance");
    require(std::abs(hard_cutoff.mean-expected)>12*tolerance,
            "roulette test has power to reject a hard sub-threshold cutoff");
    std::cout << "roulette: N=" << sample_count << " threshold=" << p.ratioRrThreshold
              << " expected_T=" << expected << " mean=" << samples.mean
              << " estimated_standard_error=" << std::sqrt(samples.variance()/sample_count)
              << " tolerance=" << tolerance << " tests=" << roulette_tests
              << " terminated=" << terminated << " candidates=" << double(candidates)/sample_count
              << " erroneous_hard_cutoff_mean=" << hard_cutoff.mean
              << " hard_cutoff_error_over_tolerance=" << std::abs(hard_cutoff.mean-expected)/tolerance << '\n';
}

void test_ballistic_and_mis_expectations() {
    const auto p=parameters();
    constexpr long double height=250,distance=6000,target_tau=0.8L;
    const long double column=tracking_reference::vertical_tau(height,distance,p.scaleHeight,1);
    const float beta=float(target_tau/column);
    const tracking::RaySegment ray{{0,p.groundY+float(height),0},{0,1,0},float(distance)};
    const double target=double(std::exp(-tracking_reference::vertical_tau(
        height,distance,p.scaleHeight,beta)));
    OpenRng delta_random(0x152e8c61u),ratio_random(0xbfe68157u);
    Moments ballistic,attenuation;
    for (std::uint32_t i=0;i<sample_count;++i) {
        const auto flight=tracking::delta_free_flight(p,ray,beta,delta_random);
        const auto ratio=tracking::ratio_transmittance(p,ray,beta,ratio_random);
        require_ok(flight.status,"ballistic free flight");require_ok(ratio.status,"ballistic ratio");
        ballistic.add(flight.scattered?0:1);
        attenuation.add(ratio.transmittance);
    }
    const double tolerance=six_sigma(target*(1-target),sample_count,probability_allowance);
    near(ballistic.mean,target,tolerance,"surviving endpoint has weight one, giving expectation T");
    const double wrong=ballistic.mean*target;
    require(std::abs(wrong-target)>15*tolerance,
            "test distinguishes an erroneous second attenuation factor at the endpoint");

    // A constant source and fixed angular factors isolate the bookkeeping:
    // continuation supplies a Bernoulli survival; NEE supplies an independent
    // ratio estimate. Both target the SAME attenuated term. This CPU check
    // does not by itself claim to have executed OptiX's endpoint shaders.
    for (unsigned mode=0;mode<3;++mode) {
        const float bsdf_pdf=0.37f,light_pdf=0.81f;
        const float a=m3::sun_emission_weight(1,false,mode,bsdf_pdf,light_pdf);
        const float b=m3::nee_weight(mode,light_pdf,bsdf_pdf);
        near(double(a)+b,1,1e-7,"sun MIS strategies partition unity");
        const double mean=a*ballistic.mean+b*attenuation.mean;
        const double variance=a*a*ballistic.variance()+b*b*attenuation.variance();
        near(mean,target,six_sigma(variance,sample_count,probability_allowance),
             "continuation/NEE/MIS retain the same transport expectation");
        near(m3::area_emission_weight(true,1,false,mode,bsdf_pdf,0),1,0,
             "medium-to-area endpoint has weight one because medium has no area NEE");
    }
    std::cout << "transport: expected_T=" << target << " ballistic=" << ballistic.mean
              << " ratio=" << attenuation.mean << " erroneous_T_twice=" << wrong
              << " double_T_error_over_tolerance=" << std::abs(wrong-target)/tolerance << '\n';
}
}

int main() {
    try {
        std::cout << std::setprecision(12);
        test_coefficient_and_random_contract();
        test_vacuum_ground_and_rejections();
        test_raw_ray_parameter();
        test_distributions();
        test_roulette_mean();
        test_ballistic_and_mis_expectations();
        std::cout << "M3 FP32 delta/ratio tracking regressions passed; CPU statistical validation only.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M3 tracking regression FAILED: " << error.what() << '\n';
        return 1;
    }
}
