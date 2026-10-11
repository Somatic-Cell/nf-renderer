#pragma once

// CPU-only independent reference for the tracking regression. This header does
// not use production clipping, the compensated FP32 geometry, tracking code,
// Gauss--Kronrod quadrature, or the numerical inverse CDF.
#include <array>
#include <cmath>
#include <stdexcept>

namespace tracking_reference {

struct Atmosphere {
    long double radius = 6371000;
    long double scale_height = 8000;
    long double ground_y = -6;
};

struct Ray {
    std::array<long double, 3> origin{};
    std::array<long double, 3> direction{0, 1, 0};
    // These bounds are supplied independently by each test case, in metres
    // from the original origin, rather than copied from production clipping.
    long double begin_m = 0;
    long double end_m = 0;

    long double direction_length() const {
        return std::sqrt(direction[0]*direction[0] + direction[1]*direction[1] +
                         direction[2]*direction[2]);
    }
};

template<class Function>
long double refine_simpson(const Function& f, long double a, long double b,
    long double fa, long double fm, long double fb, long double coarse,
    long double tolerance, unsigned depth) {
    const long double middle = (a+b)/2;
    const long double left_middle = (a+middle)/2;
    const long double right_middle = (middle+b)/2;
    const long double fl = f(left_middle), fr = f(right_middle);
    const long double left = (middle-a)*(fa+4*fl+fm)/6;
    const long double right = (b-middle)*(fm+4*fr+fb)/6;
    const long double difference = left+right-coarse;
    if (std::abs(difference) <= 15*tolerance)
        return left+right+difference/15;
    if (depth == 0)
        throw std::runtime_error("independent Simpson reference exhausted its depth");
    return refine_simpson(f,a,middle,fa,fl,fm,left,tolerance/2,depth-1) +
           refine_simpson(f,middle,b,fm,fr,fb,right,tolerance/2,depth-1);
}

template<class Function>
long double integrate_unit(const Function& function) {
    const long double first=function(0), middle=function(0.5L), last=function(1);
    return refine_simpson(function,0,1,first,middle,last,
                          (first+4*middle+last)/6,2e-14L,24);
}

class ExponentialRay {
public:
    ExponentialRay(Atmosphere atmosphere, Ray ray, long double beta)
        : atmosphere_(atmosphere), ray_(ray), beta_(beta), norm_(ray.direction_length()) {
        if (!(norm_ > 0) || !std::isfinite(norm_) || !(beta_ >= 0) ||
            !(ray_.end_m >= ray_.begin_m) || !(ray_.begin_m >= 0))
            throw std::runtime_error("invalid independent reference ray");
    }

    long double height(long double distance_m) const {
        const long double x=ray_.origin[0]+ray_.direction[0]*distance_m/norm_;
        const long double y=ray_.origin[1]-atmosphere_.ground_y+atmosphere_.radius+
                            ray_.direction[1]*distance_m/norm_;
        const long double z=ray_.origin[2]+ray_.direction[2]*distance_m/norm_;
        // Deliberately Cartesian long-double arithmetic, independent of the
        // production compensated products / rationalized-height expression.
        return std::sqrt(x*x+y*y+z*z)-atmosphere_.radius;
    }

    long double column(unsigned power, long double end_m) const {
        if (end_m == ray_.begin_m) return 0;
        if (power == 0 || end_m < ray_.begin_m || end_m > ray_.end_m)
            throw std::runtime_error("independent column interval outside its test case");
        const long double length=end_m-ray_.begin_m;
        return length*integrate_unit([&](long double u) {
            const long double h=height(ray_.begin_m+length*u);
            if (h < -1e-7L)
                throw std::runtime_error("independent reference segment enters the planet");
            return std::exp(-static_cast<long double>(power)*h/atmosphere_.scale_height);
        });
    }

    long double tau(long double end_m) const { return beta_*column(1,end_m); }
    long double tau() const { return tau(ray_.end_m); }
    long double transmittance() const { return std::exp(-tau()); }

    // CDF targets are fractions of the total continuous collision mass. The
    // endpoint mass is retained separately; it is never renormalized away.
    long double collision_quantile(long double fraction) const {
        if (!(fraction > 0 && fraction < 1))
            throw std::runtime_error("invalid conditional reference quantile");
        const long double target=-std::log1p(fraction*std::expm1(-tau()));
        long double lo=ray_.begin_m, hi=ray_.end_m;
        for (unsigned iteration=0;iteration<40;++iteration) {
            const long double midpoint=(lo+hi)/2;
            if (tau(midpoint) < target) lo=midpoint; else hi=midpoint;
        }
        return (lo+hi)/2;
    }

    // For global M=beta and no roulette, the Poisson product formula gives
    // E[T^k] = exp(integral M*((1-sigma/M)^k - 1) ds). These moments are
    // independent of the production random sequence and sampled products.
    std::array<long double, 5> ratio_moments() const {
        std::array<long double,5> integral{};
        for (unsigned power=1;power<=4;++power)
            integral[power]=beta_*column(power,ray_.end_m);
        return {{1,
            std::exp(-integral[1]),
            std::exp(-2*integral[1]+integral[2]),
            std::exp(-3*integral[1]+3*integral[2]-integral[3]),
            std::exp(-4*integral[1]+6*integral[2]-4*integral[3]+integral[4])}};
    }

    long double full_candidate_mean() const {
        return beta_*(ray_.end_m-ray_.begin_m);
    }

private:
    Atmosphere atmosphere_;
    Ray ray_;
    long double beta_;
    long double norm_;
};

inline long double vertical_tau(long double height, long double distance,
                               long double scale_height, long double beta) {
    return beta*std::exp(-height/scale_height)*scale_height*
           (-std::expm1(-distance/scale_height));
}

} // namespace tracking_reference
