#pragma once
#include "parameters.hpp"
#include "m3/tracking.hpp"
#include "m4_nf/flow.hpp"
namespace m4rain {
namespace mt=m3::tracking;
using mt::FloatPair;using mt::Status;using mt::RaySegment;
NF_HD Status validate(const Parameters& p){
    if(p.enabled>1u||(p.backend!=Backend::Fp32&&p.backend!=Backend::CoopVec)||
       (p.phaseModel!=PhaseModel::NfAnisotropic&&p.phaseModel!=PhaseModel::SphereMie1mm)||
       (p.crossSectionMode!=m4mie::CrossSection::GeometricMatched&&
        p.crossSectionMode!=m4mie::CrossSection::PhysicalMie)||
       (p.densityModel!=DensityModel::Constant&&p.densityModel!=DensityModel::ExponentialY)||
       (p.endcapPolicy!=m4nf::EndcapPolicy::ConstantPhysical&&p.endcapPolicy!=m4nf::EndcapPolicy::Reject)||
       !m1::finite(p.boundsMin)||!m1::finite(p.boundsMax)||
       !(p.boundsMin.x<p.boundsMax.x)||!(p.boundsMin.y<p.boundsMax.y)||!(p.boundsMin.z<p.boundsMax.z)||
       !m4nf::unit(p.particleAxis)||!m1::finite(p.numberDensityAtMinY)||p.numberDensityAtMinY<0||
       !m1::finite(p.densityScaleHeight)||!(p.densityScaleHeight>0))return Status::InvalidParameters;
    return Status::Ok;
}
struct BoxInterval {Status status=Status::Ok;bool nonempty=false;FloatPair begin{},end{};};
NF_HD BoxInterval clip_box(const Parameters& p,const RaySegment& r,FloatPair begin,FloatPair end){
    BoxInterval out;out.begin=begin;out.end=end;
    const float lo[3]{p.boundsMin.x,p.boundsMin.y,p.boundsMin.z};
    const float hi[3]{p.boundsMax.x,p.boundsMax.y,p.boundsMax.z};
    const float o[3]{r.origin.x,r.origin.y,r.origin.z},d[3]{r.direction.x,r.direction.y,r.direction.z};
    for(unsigned k=0;k<3;++k){
        if(d[k]==0.0f){if(o[k]<lo[k]||o[k]>hi[k])return out;continue;}
        const auto da=mt::sub(FloatPair{lo[k],0},o[k]);
        const auto db=mt::sub(FloatPair{hi[k],0},o[k]);
        // Near-parallel directions may have slab endpoints beyond FP32 range.
        // Compare the coordinate interval at the already finite ray endpoints
        // before division; inactive slabs need no unrepresentable quotient.
        const auto pa=mt::add(mt::mul(begin,d[k]),o[k]);
        const auto pb=mt::add(mt::mul(end,d[k]),o[k]);
        const auto minp=mt::minimum(pa,pb),maxp=mt::maximum(pa,pb);
        if(mt::compare(maxp,lo[k])<0||mt::compare(minp,hi[k])>0)return out;
        if(mt::compare(minp,lo[k])>=0&&mt::compare(maxp,hi[k])<=0)continue;
        // At least one boundary is active inside the finite segment. Clamp
        // inactive endpoints geometrically before dividing, not the density.
        FloatPair a=begin,b=end;
        if(d[k]>0){
            if(mt::compare(pa,lo[k])<0)a=mt::divide(da,d[k]);
            if(mt::compare(pb,hi[k])>0)b=mt::divide(db,d[k]);
        }else{
            if(mt::compare(pa,hi[k])>0)a=mt::divide(db,d[k]);
            if(mt::compare(pb,lo[k])<0)b=mt::divide(da,d[k]);
        }
        if(!mt::finite(a)||!mt::finite(b)){out.status=Status::NonfiniteGeometry;return out;}
        out.begin=mt::maximum(out.begin,a);out.end=mt::minimum(out.end,b);
        if(mt::compare(out.end,out.begin)<=0)return out;
    }
    out.nonempty=mt::compare(out.end,out.begin)>0;return out;
}
NF_HD Status density_exponent(const Parameters& p,const RaySegment& r,FloatPair t,float& e){
    e=0.0f;if(p.densityModel==DensityModel::Constant)return Status::Ok;
    const auto y=mt::sub(mt::point_at(r,t).y,p.boundsMin.y);
    if(!mt::finite(y)||mt::compare(y,0.0f)<0)return Status::MajorantViolation;
    e=-mt::value(y)/p.densityScaleHeight;
    return e==e&&e<=0.0f?Status::Ok:Status::MajorantViolation;
}
struct RainFlight {
    Status status=Status::Ok;bool scattered=false;FloatPair parameter{};
    std::uint32_t candidates=0,nulls=0;
};
// A straight ray has constant Cext because all drops in this rain shaft have
// one fixed orientation. Null events evaluate only the vertical density.
template<class RNG>
NF_HD RainFlight rain_delta(const Parameters& p,const RaySegment& r,const BoxInterval& box,
                            float cext,float length,std::uint32_t budget,RNG& rng){
    RainFlight out;out.status=box.status;
    if(out.status!=Status::Ok||!box.nonempty||p.numberDensityAtMinY==0.0f)return out;
    const float rate=(p.numberDensityAtMinY*cext)*length;
    if(!m1::finite(rate)||!(rate>0)){out.status=Status::InvalidCoefficient;return out;}
    auto t=box.begin;
    for(;;){
        FloatPair next{};bool beyond=false;
        out.status=mt::tracking_detail::next_candidate(rng,rate,t,box.end,next,beyond);
        if(out.status!=Status::Ok||beyond)return out;
        if(out.candidates==budget){out.status=Status::TrackingLimit;return out;}
        ++out.candidates;float e=0,u=0;
        out.status=density_exponent(p,r,next,e);if(out.status!=Status::Ok)return out;
        const float accept=::expf(e);
        if(!m1::finite(accept)||accept<0||accept>1){out.status=Status::MajorantViolation;return out;}
        out.status=mt::tracking_detail::uniform_open(rng,u);if(out.status!=Status::Ok)return out;
        if(u<accept){out.scattered=true;out.parameter=next;return out;}
        ++out.nulls;t=next;
    }
}
struct RainRatio {
    Status status=Status::Ok;float value=1.0f;std::uint32_t candidates=0,roulette=0;
};
template<class RNG>
NF_HD RainRatio rain_ratio(const Parameters& p,const RaySegment& r,const BoxInterval& box,
                           float cext,float length,std::uint32_t budget,float threshold,RNG& rng){
    RainRatio out;out.status=box.status;
    if(out.status!=Status::Ok||!box.nonempty||p.numberDensityAtMinY==0.0f)return out;
    const float rate=(p.numberDensityAtMinY*cext)*length;
    if(!m1::finite(rate)||!(rate>0)){out.status=Status::InvalidCoefficient;return out;}
    auto t=box.begin;
    for(;;){
        FloatPair next{};bool beyond=false;
        out.status=mt::tracking_detail::next_candidate(rng,rate,t,box.end,next,beyond);
        if(out.status!=Status::Ok||beyond)return out;
        if(out.candidates==budget){out.status=Status::TrackingLimit;return out;}
        ++out.candidates;float e=0;
        out.status=density_exponent(p,r,next,e);if(out.status!=Status::Ok)return out;
        const float factor=-::expm1f(e);
        if(!m1::finite(factor)||factor<0||factor>1){out.status=Status::MajorantViolation;return out;}
        if(factor==0){out.value=0;return out;}
        if(threshold>0&&factor<threshold/out.value){
            const float survival=(out.value/threshold)*factor;
            if(!m1::finite(survival)||!(survival>0)||survival>1){out.status=Status::NumericalUnderflow;return out;}
            ++out.roulette;float u=0;
            out.status=mt::tracking_detail::uniform_open(rng,u);if(out.status!=Status::Ok)return out;
            if(u>=survival){out.value=0;return out;}out.value=threshold;
        }else{
            out.value*=factor;
            if(!(out.value>0)){out.status=Status::NumericalUnderflow;return out;}
        }
        t=next;
    }
}
NF_HD m4nf::CrossSections camera_cross_sections(const Parameters& p,const m4nf::ModelView& m,
                                               const RaySegment& r,float lambda,
                                               const m4mie::View& mie={}){
    if(p.phaseModel==PhaseModel::SphereMie1mm){
        m4nf::CrossSections out{};
        std::uint32_t a=0,b=0;float wa=0,wb=0;
        if(!m4mie::spectral_weights(mie,lambda,p.crossSectionMode,a,b,wa,wb,
                                    out.scattering,out.extinction)){
            out.status=m4nf::Status::invalid_asset;return out;
        }
        out.status=m4nf::Status::success;return out;
    }
    const auto d=m4nf::normalize(m4nf::Vec3{-r.direction.x,-r.direction.y,-r.direction.z});
    return m4nf::cross_sections(m,lambda,m4nf::direction_cosine(p.particleAxis,d),p.endcapPolicy);
}
enum class Component : std::uint32_t { Rayleigh=0,Rain=1 };
struct MixedFlight {
    mt::FreeFlightResult flight{};Component component=Component::Rayleigh;
    RainFlight rain{};m4nf::Status nfStatus=m4nf::Status::success;
};
// Independent competing free flights give p_j(t)=T_R(t)*T_r(t)*sigma_t,j(-d).
// The component label is part of the sampled vertex. Scatter/NEE must then use
// K_j/sigma_t,j and q_j, NOT total K divided by a component-only PDF.
template<class RNG>
NF_HD MixedFlight free_flight(const m3::TrackingParameters& a,const Parameters& p,
                              const m4nf::ModelView& m,const RaySegment& r,
                              float beta,float lambda,RNG& rng,const m4mie::View& mie={}){
    MixedFlight out;out.flight=mt::delta_free_flight(a,r,beta,rng);
    if(out.flight.status!=Status::Ok||!p.enabled||p.numberDensityAtMinY==0||!out.flight.clip.has_medium)return out;
    out.flight.status=validate(p);if(out.flight.status!=Status::Ok)return out;
    const auto end=out.flight.scattered?out.flight.parameter:out.flight.clip.end_parameter;
    const auto box=clip_box(p,r,out.flight.clip.begin_parameter,end);
    if(box.status!=Status::Ok){out.flight.status=box.status;return out;}
    if(!box.nonempty)return out;
    const auto c=camera_cross_sections(p,m,r,lambda,mie);out.nfStatus=c.status;
    if(c.status!=m4nf::Status::success){out.flight.status=Status::InvalidCoefficient;return out;}
    out.rain=rain_delta(p,r,box,c.extinction,out.flight.clip.direction_length,a.maxTrackingEvents,rng);
    if(out.rain.status!=Status::Ok){out.flight.status=out.rain.status;return out;}
    if(out.rain.scattered){
        out.component=Component::Rain;out.flight.scattered=true;out.flight.endpoint=mt::Endpoint::None;
        out.flight.parameter=out.rain.parameter;out.flight.ray_t=mt::value(out.rain.parameter);
    }
    return out;
}
struct MixedRatio {
    mt::RatioResult rayleigh{};RainRatio rain{};
    Status status=Status::Ok;m4nf::Status nfStatus=m4nf::Status::success;float value=1.0f;
};
template<class RNG>
NF_HD MixedRatio transmittance(const m3::TrackingParameters& a,const Parameters& p,
                               const m4nf::ModelView& m,const RaySegment& r,
                               float beta,float lambda,RNG& rng,const m4mie::View& mie={}){
    MixedRatio out;out.rayleigh=mt::ratio_transmittance(a,r,beta,rng);
    out.status=out.rayleigh.status;out.value=out.rayleigh.transmittance;
    if(out.status!=Status::Ok||out.value==0||!p.enabled||p.numberDensityAtMinY==0||!out.rayleigh.clip.has_medium)return out;
    out.status=validate(p);if(out.status!=Status::Ok)return out;
    const auto box=clip_box(p,r,out.rayleigh.clip.begin_parameter,out.rayleigh.clip.end_parameter);
    out.status=box.status;if(out.status!=Status::Ok||!box.nonempty)return out;
    const auto c=camera_cross_sections(p,m,r,lambda,mie);out.nfStatus=c.status;
    if(c.status!=m4nf::Status::success){out.status=Status::InvalidCoefficient;return out;}
    out.rain=rain_ratio(p,r,box,c.extinction,out.rayleigh.clip.direction_length,
                        a.maxTrackingEvents,a.ratioRrThreshold,rng);
    out.status=out.rain.status;if(out.status!=Status::Ok)return out;
    out.value*=out.rain.value;
    if(!m1::finite(out.value)||(out.value==0&&out.rain.value>0))out.status=Status::NumericalUnderflow;
    return out;
}
}
