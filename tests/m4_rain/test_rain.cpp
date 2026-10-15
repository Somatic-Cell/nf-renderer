#include "m4_rain/tracking.hpp"
#include "m4_rain/scattering.hpp"
#include "m4_rain/sha256.hpp"
#include "m4_nf/asset.hpp"
#include "m4_nf/anisotropic.hpp"
#include "m3/transport.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <cstring>
#include <iomanip>

static void require(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
struct RNG {
    std::mt19937 engine;
    explicit RNG(unsigned s):engine(s){}
    float operator()(){return (float(engine()>>9)+0.5f)*0x1p-23f;}
};
struct FixedRNG {float operator()(){return 0.5f;}};
struct Stats {
    double sum=0,sum2=0;unsigned n=0;
    void add(float v){sum+=v;sum2+=double(v)*v;++n;}
    double mean()const{return sum/n;}
    double se()const{return std::sqrt(std::max(0.0,(sum2-sum*sum/n)/(n-1))/n);}
    void compare(double target,const char* name)const{
        const double err=std::abs(mean()-target),bound=6*se()+4.0/n;
        std::cout<<name<<" n="<<n<<" measured="<<mean()<<" reference="<<target<<" stderr="<<se()<<" error="<<err<<" bound="<<bound<<'\n';
        require(err<=bound,"statistical expectation mismatch");
    }
};
int main(int argc,char** argv){
try{
    require(argc==2,"Pass model.bin");
    const auto asset=m4nf::load_asset(argv[1]);const auto m=asset.view();
    const auto policy=m4nf::EndcapPolicy::ConstantPhysical;
    require(m4rain::file_sha256(argv[1])=="b1dca30aa8fee6ee1fb45b25b47a92cf871682943d81f07112e9a09bc7d4208e","model hash mismatch");
    for(float l:{380.0f,442.710419f,550.0f,830.0f})for(float c:{-1.0f,-0.999999f,0.999999f,1.0f}){
        const auto ctx=m4nf::prepare(m,l,c,policy);
        const float edge=c<0?asset.cosines.front():asset.cosines.back();
        const auto base=m4nf::prepare(m,l,edge);
        require(ctx.status==m4nf::Status::success&&ctx.cosine==c&&ctx.external[1]==c,"neural condition was clipped");
        require(ctx.a==base.a&&ctx.b==base.b&&ctx.csca==base.csca&&ctx.cext==base.cext,"physical endcap not constant");
        require(ctx.strength==std::sqrt((1.0f-c)*(1.0f+c)),"azimuth strength changed");
        const auto cs=m4nf::cross_sections(m,l,c,policy);
        require(cs.status==m4nf::Status::success&&cs.extinction==base.cext,"endcap coefficient mismatch");
        require(m4nf::prepare(m,l,c).status==m4nf::Status::condition_out_of_domain,"strict policy changed");
    }
    for(float c:{-.9f,0.0f,.5f,.9f}){
        const auto a=m4nf::prepare(m,550,c),b=m4nf::prepare(m,550,c,policy);
        require(a.a==b.a&&a.b==b.b&&a.external[1]==b.external[1]&&std::memcmp(a.base_logits,b.base_logits,sizeof a.base_logits)==0,"interior changed");
    }
    require(m4nf::prepare(m,379,0,policy).status==m4nf::Status::condition_out_of_domain,"wavelength extrapolation");
    require(m4nf::prepare(m,550,1.00001f,policy).status==m4nf::Status::condition_out_of_domain,"invalid NN cosine accepted");
    std::cout<<"endcap: 16 physical/NN cases + interior parity + strict rejection PASS\n";
    m4rain::Parameters p=m4rain::default_parameters();p.enabled=1;
    p.boundsMin={-1000,0,-1000};p.boundsMax={1000,2000,1000};
    const m3::tracking::RaySegment ray{{0,2,-2000},{0,0,1},5000};
    const auto box=m4rain::clip_box(p,ray,{0,0},{5000,0});
    require(box.nonempty&&m3::tracking::value(box.begin)==1000&&m3::tracking::value(box.end)==3000,"rain box clip");
    auto parallel=ray;parallel.origin.x=1100;
    require(!m4rain::clip_box(p,parallel,{0,0},{5000,0}).nonempty,"parallel miss");
    parallel=ray;parallel.direction.x=1e-30f;
    require(m4rain::clip_box(p,parallel,{0,0},{5000,0}).nonempty,"near parallel clip");
    std::cout<<"compensated AABB/parallel boundaries PASS\n";
    const float cs=m4rain::camera_cross_sections(p,m,ray,550).extinction;
    constexpr unsigned N=65536;
    for(unsigned profile=0;profile<2;++profile){
        p.densityModel=static_cast<m4rain::DensityModel>(profile);p.densityScaleHeight=300.0f;
        m3::tracking::RaySegment r=profile?m3::tracking::RaySegment{{0,0,0},{0,1,0},2000}:ray;
        const auto b=m4rain::clip_box(p,r,{0,0},{r.t_max,0});
        const float c=m4rain::camera_cross_sections(p,m,r,550).extinction;
        const double column=profile?300.0*(-std::expm1(-2000.0/300.0)):2000.0;
        const double expected=std::exp(-double(c)*p.numberDensityAtMinY*column);
        RNG rng(411+profile);Stats delta,ratio,rr;
        for(unsigned i=0;i<N;++i){
            const auto d=m4rain::rain_delta(p,r,b,c,1,100000,rng);
            const auto t=m4rain::rain_ratio(p,r,b,c,1,100000,0,rng);
            const auto u=m4rain::rain_ratio(p,r,b,c,1,100000,.1f,rng);
            require(d.status==m4rain::Status::Ok&&t.status==m4rain::Status::Ok&&u.status==m4rain::Status::Ok,"tracking failed");
            delta.add(d.scattered?0.0f:1.0f);ratio.add(t.value);rr.add(u.value);
        }
        delta.compare(expected,profile?"exponential delta":"constant delta");
        ratio.compare(expected,profile?"exponential ratio":"constant ratio");
        rr.compare(expected,profile?"exponential ratio RR":"constant ratio RR");
    }
    // Check propagation sign against deliberately non-even artificial Cext.
    const float wavelengths[2]{380,830},cosines[2]{-1,1},fields[4]{1e-6f,3e-6f,1e-6f,3e-6f};
    auto uneven=m;uneven.wavelengths=wavelengths;uneven.cosines=cosines;uneven.nlambda=uneven.ncos=2;
    uneven.scattering=uneven.extinction=fields;
    const m3::tracking::RaySegment up{{0,2,0},{0,1,0},10};
    require(m4rain::camera_cross_sections(p,uneven,up,550).extinction==3e-6f,"used +d instead of physical -d");
    // Equal hazard limit of the real spherical R density: very large scale H.
    m3::TrackingParameters a;a.enabled=1;a.beta550=.001f;a.scaleHeight=1e10f;a.ratioRrThreshold=.01f;
    p=m4rain::default_parameters();p.enabled=1;p.boundsMin={-1000,0,-1000};p.boundsMax={1000,100,1000};p.numberDensityAtMinY=4000;
    const m3::tracking::RaySegment r{{0,2,0},{0,0,1},100};
    const float c=m4rain::camera_cross_sections(p,m,r,550).extinction;
    const double sr=double(c)*p.numberDensityAtMinY,total=sr+a.beta550,T=std::exp(-total*100.0);
    Stats end,R,rain,totalTr;RNG rng(772);
    for(unsigned i=0;i<N;++i){
        const auto s=m4rain::free_flight(a,p,m,r,a.beta550,550,rng);
        const auto t=m4rain::transmittance(a,p,m,r,a.beta550,550,rng);
        require(s.flight.status==m4rain::Status::Ok&&t.status==m4rain::Status::Ok,"mixture failure");
        end.add(s.flight.scattered?0.0f:1.0f);
        R.add(s.flight.scattered&&s.component==m4rain::Component::Rayleigh?1.0f:0.0f);
        rain.add(s.flight.scattered&&s.component==m4rain::Component::Rain?1.0f:0.0f);totalTr.add(t.value);
    }
    end.compare(T,"mixture escape");R.compare(a.beta550/total*(1-T),"mixture Rayleigh mark");rain.compare(sr/total*(1-T),"mixture rain mark");totalTr.compare(T,"mixture shadow");
    p.numberDensityAtMinY=0;RNG oldrng(711),newrng(711);
    for(unsigned i=0;i<128;++i){
        const auto x=m3::tracking::delta_free_flight(a,r,a.beta550,oldrng),y=m4rain::free_flight(a,p,m,r,a.beta550,550,newrng).flight;
        require(x.scattered==y.scattered&&x.parameter.hi==y.parameter.hi&&x.parameter.lo==y.parameter.lo,"zero-density M3 changed");
    }
    require(oldrng.engine==newrng.engine,"zero-density consumed extra random numbers");
    const m3::tracking::RaySegment ground{{0,2,0},{0,-1,0},100};
    require(m4rain::transmittance(a,p,m,ground,a.beta550,550,newrng).value==0,"opaque planet shadow lost");
    p.numberDensityAtMinY=400000;p.densityModel=m4rain::DensityModel::ExponentialY;p.densityScaleHeight=.2f;
    FixedRNG fixed;auto shortbox=m4rain::clip_box(p,r,{0,0},{100,0});
    require(m4rain::rain_delta(p,r,shortbox,c,1,1,fixed).status==m4rain::Status::TrackingLimit,"tracking budget accepted truncated path");
    std::cout<<"component law / propagation signs / zero-density bitwise / planet / budget PASS\n";
    const m4nf::NativeMlp mlp{m.weights};RNG rays(539);float maxPdf=0,maxWeight=0;unsigned nonunitWeights=0;
    for(unsigned i=0;i<128;++i){
        const float mu=i<2?(i?1.0f:-1.0f):2*rays()-1;
        const float phi=6.28318530718f*rays();
        const m4nf::Vec3 d{std::sqrt((1-mu)*(1+mu))*std::cos(phi),mu,std::sqrt((1-mu)*(1+mu))*std::sin(phi)};
        m4rain::ScatterRequest q{m4rain::ScatterOp::Sample,380+450*rays(),rays(),rays(),d,{}};
        const auto s=m4rain::scatter(m,mlp,{0,-1,0},policy,q);
        require(s.status==m4nf::Status::success,"rain angular sample");
        q.op=m4rain::ScatterOp::Evaluate;q.nextDirection=s.direction;
        const auto e=m4rain::scatter(m,mlp,{0,-1,0},policy,q);
        require(e.status==m4nf::Status::success,"rain angular eval");
        maxPdf=std::max(maxPdf,std::abs(std::log(s.pdf/e.pdf)));
        const auto k=m4nf::forward_kernel(m,mlp,{0,-1,0},-s.direction,-d,q.wavelength,100.0f,policy);
        const auto co=m4nf::cross_sections(m,q.wavelength,m4nf::direction_cosine({0,-1,0},-d),policy);
        require(k.status==m4nf::Status::success,"kernel input");
        const float expected=k.kernel/(100.0f*co.extinction*s.pdf);
        maxWeight=std::max(maxWeight,std::abs(std::log(s.weight/expected)));
        if(std::abs(s.weight-1.0f)>.01f)++nonunitWeights;
    }
    require(maxPdf<.003f&&maxWeight<.003f&&nonunitWeights>0,"kernel/proposal or reciprocity incorrectly collapsed");
    std::cout<<"128 camera angular sample/kernel cases PASS max_log_pdf_diff="<<maxPdf<<" max_log_weight_diff="<<maxWeight<<" nonunit_weights="<<nonunitWeights<<'\n';
    // All MIS weights use the SAME component-conditioned q on both sides.
    for(float q:{1e-6f,.1f,100.0f})require(std::abs(m3::nee_weight(0,1000,q)+m3::sun_emission_weight(1,false,0,q,1000)-1)<1e-6f,"MIS complement");
    require(m3::area_emission_weight(true,1,false,0,.1f,.7f)==1,"medium area NEE must be absent");
    std::cout<<"MIS complement and medium-emitter policy PASS\n";
    std::cout<<"All rain integration core tests passed; GPU renderer not exercised.\n";
    (void)cs;return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
