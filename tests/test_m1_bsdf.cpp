#include "m1/bsdf.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace m1;
constexpr double dpi=3.141592653589793238462643383279502884;

void require(bool pass,const std::string& label) {
    if(!pass)throw std::runtime_error(label);
}
void close(double actual,double expected,double absolute,const std::string& label) {
    if(!std::isfinite(actual) || std::abs(actual-expected)>absolute)
        throw std::runtime_error(label+": actual="+std::to_string(actual)+
                                 " expected="+std::to_string(expected));
}
float radical_inverse(std::uint32_t bits) {
    bits=(bits<<16u)|(bits>>16u);
    bits=((bits&0x55555555u)<<1u)|((bits&0xaaaaaaaau)>>1u);
    bits=((bits&0x33333333u)<<2u)|((bits&0xccccccccu)>>2u);
    bits=((bits&0x0f0f0f0fu)<<4u)|((bits&0xf0f0f0f0u)>>4u);
    bits=((bits&0x00ff00ffu)<<8u)|((bits&0xff00ff00u)>>8u);
    return float(double(bits)*0x1p-32);
}

struct Integrals {double albedo=0,pdf_mass=0;};
Integrals integrate_hemisphere(const BsdfParameters& p,Vec3 wo,
                               unsigned ncos=768,unsigned nphi=1536) {
    Integrals total;
    for(unsigned i=0;i<ncos;++i) {
        const double cosine=(double(i)+0.5)/ncos;
        const double sine=std::sqrt(1.0-cosine*cosine);
        for(unsigned j=0;j<nphi;++j) {
            const double phi=2.0*dpi*(double(j)+0.5)/nphi;
            const Vec3 wi{float(sine*std::cos(phi)),float(sine*std::sin(phi)),float(cosine)};
            const auto v=evaluate_bsdf(p,wo,wi);
            require(v.valid,"hemispherical quadrature has valid evaluation");
            total.albedo+=v.f*cosine;
            total.pdf_mass+=v.pdf;
        }
    }
    const double domega=2.0*dpi/(double(ncos)*nphi);
    total.albedo*=domega;total.pdf_mass*=domega;
    return total;
}

Integrals sample_integrals(const BsdfParameters& p,Vec3 wo,unsigned count=262144) {
    Integrals total;
    for(unsigned i=0;i<count;++i) {
        const auto s=sample_bsdf(p,wo,0.5f,(float(i)+0.5f)/float(count),radical_inverse(i));
        require(s.status!=BsdfSampleStatus::Invalid,"BSDF sampler has no invalid numerical event");
        if(s.status==BsdfSampleStatus::Null)continue;
        require(s.weight<=1.00002f,"reflection sample weight is energy bounded");
        require(!s.delta&&s.branch_pmf==0,"continuous/delta measures stay separate");
        const auto v=evaluate_bsdf(p,wo,s.wi);
        require(v.valid,"sample direction evaluates");
        close(s.pdf,v.pdf,2.0e-6*std::max(1.0,double(v.pdf)),"sample/eval PDF agreement");
        close(s.weight,double(v.f)*s.wi.z/v.pdf,
              2.0e-5*std::max(1.0,double(s.weight)),"weight equals f*cos/pdf");
        total.albedo+=s.weight;total.pdf_mass+=1.0;
    }
    total.albedo/=count;total.pdf_mass/=count;
    return total;
}

void test_lambert() {
    const BsdfParameters p{MaterialKind::Diffuse,0.63f,0.2f,1};
    const Vec3 wo=normalize(Vec3{0.7f,0.2f,1});
    const auto q=integrate_hemisphere(p,wo,64,128);
    close(q.albedo,0.63,1.0e-7,"Lambert directional albedo");
    close(q.pdf_mass,1.0,1.0e-7,"cosine PDF normalization");
    const auto s=sample_integrals(p,wo,16384);
    close(s.albedo,0.63,1.0e-7,"cosine-weighted Lambert estimator");
    close(s.pdf_mass,1,0,"Lambert has no null events");
    require(evaluate_bsdf(p,wo,{0,0,-1}).f==0,"Lambert has no transmission");
}

void test_ggx_analytic() {
    const BsdfParameters p{MaterialKind::GgxReflection,1,1,1};
    const auto q=integrate_hemisphere(p,{0,0,1},4096,4);
    const auto s=sample_integrals(p,{0,0,1});
    const double exact=1.0-std::log(2.0);
    close(q.albedo,exact,1.0e-7,"GGX alpha=1 normal-incidence exact albedo 1-ln(2)");
    close(q.pdf_mass,0.5,1.0e-7,"VNDF reflection PDF has one-half continuous mass");
    close(s.albedo,exact,1.0e-6,"GGX estimator retains null-event mass");
    close(s.pdf_mass,0.5,1.0e-6,"GGX reflected hemisphere acceptance one-half");
    std::cout<<"GGX alpha=1: albedo="<<s.albedo<<" exact="<<exact
             <<" acceptance="<<s.pdf_mass<<'\n';
}

void test_ggx_general() {
    for(const auto [alpha,cosine]:std::array<std::array<float,2>,4>{
            {{{0.2f,0.9f}},{{0.2f,0.2f}},{{0.7f,0.9f}},{{0.7f,0.2f}}}}) {
        const Vec3 wo{sqrtf(1-cosine*cosine),0,cosine};
        const BsdfParameters p{MaterialKind::GgxReflection,0.74f,alpha,1};
        const auto q=integrate_hemisphere(p,wo);
        const auto s=sample_integrals(p,wo);
        close(s.albedo,q.albedo,1.4e-4,"VNDF albedo vs independent direction quadrature");
        close(s.pdf_mass,q.pdf_mass,2.5e-4,"VNDF acceptance vs PDF hemisphere integral");
        require(q.albedo<=1.0 && q.pdf_mass<=1.00001,"GGX energy and density bounds");
        std::cout<<"GGX alpha="<<alpha<<" cos_o="<<cosine
                 <<" albedo quadrature="<<q.albedo<<" sampled="<<s.albedo
                 <<" PDF mass="<<q.pdf_mass<<" accepted="<<s.pdf_mass<<'\n';
    }
}

void test_grazing_and_reciprocity() {
    Rng rng{0x3540a713u};
    for(float alpha:std::array<float,4>{1.0e-4f,0.001f,0.25f,1.0f}) {
        const BsdfParameters p{MaterialKind::GgxReflection,0.04f,alpha,1};
        for(float cosine:std::array<float,5>{1.0f,0.8f,0.01f,1.0e-4f,1.0e-6f}) {
            const Vec3 wo{sqrtf(1-cosine*cosine),0,cosine};
            for(unsigned i=0;i<8192;++i) {
                const auto s=sample_bsdf(p,wo,rng.uniform(),rng.uniform(),rng.uniform());
                require(s.status!=BsdfSampleStatus::Invalid,"small-alpha/grazing sample is finite");
                if(s.status==BsdfSampleStatus::Null)continue;
                const auto a=evaluate_bsdf(p,wo,s.wi),b=evaluate_bsdf(p,s.wi,wo);
                require(a.valid&&b.valid,"reciprocal GGX evaluations are finite");
                close(a.f,b.f,2.0e-6*std::max(1.0,double(a.f)),"GGX reciprocity");
                require(s.weight<=1.00004f,"grazing VNDF weight stays bounded");
            }
        }
    }
}

void test_dielectric() {
    const auto normal=dielectric_fresnel(1,1.0f/1.5f);
    require(normal.valid&&!normal.total_internal_reflection,"normal dielectric Fresnel valid");
    close(normal.reflectance,0.04,2e-8,"exact dielectric normal Fresnel");
    const BsdfParameters entering{MaterialKind::Dielectric,0,0.2f,1.0f/1.5f};
    const auto reflected=sample_bsdf(entering,{0,0,1},0.01f,0.3f,0.6f);
    const auto transmitted=sample_bsdf(entering,{0,0,1},0.5f,0.3f,0.6f);
    require(reflected.status==BsdfSampleStatus::Valid&&reflected.delta&&!reflected.transmission,
            "ideal reflection is a discrete branch");
    require(transmitted.status==BsdfSampleStatus::Valid&&transmitted.delta&&transmitted.transmission,
            "ideal transmission is a discrete branch");
    require(reflected.pdf==0&&transmitted.pdf==0,"Dirac branches have no continuous PDF");
    close(reflected.branch_pmf+transmitted.branch_pmf,1,1e-7,"Fresnel branch normalization");
    close(transmitted.weight,4.0/9.0,1e-7,"radiance transport eta_i/eta_t squared");
    close(transmitted.weight*transmitted.eta_scale,1,1e-7,"RR eta compensation");
    close(evaluate_bsdf(entering,{0,0,1},{0,0,1}).f,0,0,"delta is not continuous f");

    const Vec3 wo=normalize(Vec3{0.6f,0,0.8f});
    const auto oblique=sample_bsdf(entering,wo,0.5f,0.3f,0.6f);
    close(oblique.wi.x,-0.4,1e-7,"Snell tangential direction");
    const auto reverse=dielectric_fresnel(-oblique.wi.z,1.5f);
    const auto forward=dielectric_fresnel(wo.z,1.0f/1.5f);
    close(reverse.reflectance,forward.reflectance,2e-7,"dielectric Fresnel path reversal");
    const auto exiting=sample_bsdf({MaterialKind::Dielectric,0,0.2f,1.5f},
                                  {sqrtf(0.75f),0,0.5f},0.99f,0.3f,0.6f);
    require(exiting.status==BsdfSampleStatus::Valid&&!exiting.transmission,
            "total internal reflection cannot choose transmission");
    close(exiting.branch_pmf,1,0,"TIR mass");
    close(exiting.weight,1,0,"TIR weight");
    const auto equal=dielectric_fresnel(0,1);
    require(equal.valid&&equal.reflectance==0,"equal IOR remains transparent even at grazing");
}

void test_mis_and_roulette() {
    for(const auto [a,b]:std::array<std::array<float,2>,5>{
            {{{0,1}},{{1,0}},{{0.1f,0.8f}},{{1e-20f,1e20f}},{{3e30f,2e30f}}}})
        close(power_heuristic(a,b)+power_heuristic(b,a),1,1e-7,"MIS complementary weights");

    // Independent techniques for a Lambert surface in a constant hemispherical
    // emitter. RR kills only the BSDF continuation, and its PDF is deliberately
    // excluded from BOTH MIS weights. beta/q supplies the survival correction.
    constexpr unsigned count=524288;
    constexpr float rho=0.7f,survival=0.37f,light_pdf=1.0f/(2.0f*pi);
    Rng rng{0x32786321u};
    double nee=0,bsdf=0;
    const BsdfParameters p{MaterialKind::Diffuse,rho,0.2f,1};
    for(unsigned i=0;i<count;++i) {
        const float c=rng.uniform();
        const Vec3 wi{sqrtf(1-c*c),0,c};
        const auto v=evaluate_bsdf(p,{0,0,1},wi);
        nee+=v.f*c/light_pdf*power_heuristic(light_pdf,v.pdf);
        const auto s=sample_bsdf(p,{0,0,1},rng.uniform(),rng.uniform(),rng.uniform());
        if(rng.uniform()<survival)
            bsdf+=s.weight/survival*power_heuristic(s.pdf,light_pdf);
    }
    const double estimate=(nee+bsdf)/count;
    close(estimate,rho,0.003,"NEE+BSDF MIS with independent roulette remains unbiased");
    std::cout<<"MIS + RR constant illumination estimate="<<estimate<<" expected="<<rho<<'\n';
}

void test_rejection() {
    const Vec3 wo{0,0,1};
    auto p=BsdfParameters{MaterialKind::GgxReflection,0.5f,1e-5f,1};
    require(!evaluate_bsdf(p,wo,wo).valid,"unsupported alpha must not be silently clamped");
    require(sample_bsdf(p,wo,0.5f,0.5f,0.5f).status==BsdfSampleStatus::Invalid,
            "invalid sampling is distinct from zero contribution");
    p.alpha=0.2f;p.color=1.01f;
    require(!evaluate_bsdf(p,wo,wo).valid,"nonphysical reflectance must not be clamped");
    p.color=std::numeric_limits<float>::quiet_NaN();
    require(!evaluate_bsdf(p,wo,wo).valid,"NaN material is rejected");
    p.color=0.5f;
    require(!evaluate_bsdf(p,{0,0,2},wo).valid,"non-unit direction rejected");
    require(sample_bsdf(p,wo,0.5f,1.0f,0.5f).status==BsdfSampleStatus::Invalid,
            "uniform variate endpoint contract is explicit");
}

void test_existing_metallic_mixture() {
    const MetallicParameters p{0.61f,0.4f,0.5f};
    const Vec3 wo=normalize(Vec3{0.8f,0,0.4f});
    constexpr unsigned nc=768,np=1536,n=262144;
    double integrated=0,sampled=0,pdf_mass=0,accepted=0;
    for(unsigned i=0;i<nc;++i) {
        const double c=(double(i)+0.5)/nc,s=std::sqrt(1-c*c);
        for(unsigned j=0;j<np;++j) {
            const double phi=2*dpi*(double(j)+0.5)/np;
            const Vec3 wi{float(s*std::cos(phi)),float(s*std::sin(phi)),float(c)};
            const auto v=evaluate_metallic(p,wo,wi);
            require(v.valid,"legacy metallic blend finite evaluation");
            integrated+=v.f*c;pdf_mass+=v.pdf;
        }
    }
    integrated*=2*dpi/(double(nc)*np);pdf_mass*=2*dpi/(double(nc)*np);
    Rng rng{0x24cf4871u};
    for(unsigned i=0;i<n;++i) {
        const auto s=sample_metallic(p,wo,rng.uniform(),rng.uniform(),rng.uniform());
        require(s.status!=BsdfSampleStatus::Invalid,"metallic mixture sampler is valid");
        if(s.status==BsdfSampleStatus::Null)continue;
        const auto v=evaluate_metallic(p,wo,s.wi);
        close(s.pdf,v.pdf,1e-6*std::max(1.0,double(v.pdf)),"metallic sample uses full mixture PDF");
        close(s.weight,double(v.f)*s.wi.z/v.pdf,2e-6,"metallic sample uses full mixture f");
        sampled+=s.weight;accepted+=1;
    }
    sampled/=n;accepted/=n;
    close(sampled,integrated,0.0025,"full metallic mixture sampling vs independent quadrature");
    close(accepted,pdf_mass,0.0025,"metallic mixture retains GGX null mass");
    close(legacy_principled_alpha(0),0.001,1e-10,"legacy alpha regularization is explicit");
    require(!valid_metallic_parameters({0.5f,1.1f,0.5f}),"invalid metalness is rejected");
    std::cout<<"Existing metallic model: albedo quadrature="<<integrated<<" sampled="<<sampled
             <<" PDF mass="<<pdf_mass<<" accepted="<<accepted<<'\n';
}

void test_shading_normal_support() {
    // A cosine distribution around Ns, restricted to the Ng hemisphere, has
    // mass (1+dot(Ns,Ng))/2. No renormalization after clipping is permitted.
    const Frame shading=make_frame({sqrtf(0.75f),0,0.5f});
    const BsdfParameters p{MaterialKind::Diffuse,0.8f,1,1};
    constexpr unsigned n=262144;
    double accepted=0,energy=0;
    for(unsigned i=0;i<n;++i) {
        const auto s=sample_bsdf(p,{0,0,1},0.5f,(float(i)+0.5f)/n,radical_inverse(i));
        require(s.status==BsdfSampleStatus::Valid,"Lambert shading sample valid");
        const Vec3 wi=shading.to_world(s.wi);
        if(wi.z<=0)continue;
        accepted+=1;energy+=s.weight;
    }
    accepted/=n;energy/=n;
    close(accepted,0.75,1.5e-4,"geometric hemisphere clipping retains null event mass");
    close(energy,0.6,1.5e-4,"radiance convention uses shading cosine consistently");
    std::cout<<"Shading-normal support: acceptance="<<accepted<<" exact=0.75\n";
}

void test_reported_bsdf_grazing() {
    // Exact decimal-to-float inputs from the frame 111 Grass_blades.DoubleSided
    // diagnostic. They reproduce the sampler's precondition failure; the
    // original normal/tangent frame was not logged and is not reconstructed.
    const MetallicParameters material{0.0516714454f,0.0f,0.741176486f};
    const Vec3 reported_wo{0.0777869374f,0.996969998f,-1.47486912e-08f};
    constexpr float branch=0.233046412f,u1=0.572265685f,u2=0.70618248f;
    require(valid_metallic_parameters(material)&&bsdf_unit_direction(reported_wo),
            "reported material and unit-vector contract are valid");
    require(sample_metallic(material,reported_wo,branch,u1,u2).status==BsdfSampleStatus::Invalid,
            "reported negative local cosine remains a strict BSDF rejection");

    // Use that grazing direction in an axis-aligned WORLD frame to exercise
    // the production caller contract. Orienting the frame must preserve the
    // world ray while presenting a positive, nonzero local cosine to the BSDF.
    const auto oriented=make_bsdf_frame({0,0,1},reported_wo);
    require(oriented.wo.z>0.0f&&oriented.wo.z<2.0e-8f,
            "grazing orientation preserves the small cosine without a floor");
    const Vec3 reconstructed=oriented.frame.to_world(oriented.wo);
    close(reconstructed.x,reported_wo.x,2.0e-7,"reported grazing world x preserved");
    close(reconstructed.y,reported_wo.y,2.0e-7,"reported grazing world y preserved");
    close(reconstructed.z,reported_wo.z,1.0e-14,"reported grazing world z preserved");
    const auto s=sample_metallic(material,oriented.wo,branch,u1,u2);
    require(s.status==BsdfSampleStatus::Valid,
            "reported variates produce a valid diffuse-branch sample in the oriented frame");
    const auto e=evaluate_metallic(material,oriented.wo,s.wi);
    require(e.valid&&e.pdf>0.0f,"oriented grazing sample has finite evaluation");
    close(s.pdf,e.pdf,1.0e-7,"grazing continuation and NEE use the same mixture PDF");
    close(s.weight,double(e.f)*s.wi.z/e.pdf,2.0e-7,
            "grazing continuation retains full-mixture f*cos/pdf");
    std::cout<<"Reported grazing BSDF: strict rejection retained; oriented sample PDF="
             <<s.pdf<<" weight="<<s.weight<<'\n';
}

void test_bsdf_frame_invariants() {
    // Independent double-precision dot products are a CPU test oracle only.
    const auto dot_reference=[](Vec3 a,Vec3 b) {
        return double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z;
    };
    unsigned count=0;
    const auto check=[&](Vec3 normal,Vec3 wo_world) {
        const auto oriented=make_bsdf_frame(normal,wo_world);
        const auto& f=oriented.frame;
        require(bsdf_unit_direction(oriented.wo)&&oriented.wo.z>=0.0f,
                "one shared frame/local-direction pair faces the incoming ray");
        // The synthetic normal below differs from exact unit length by four
        // float epsilons. Allow that input representation error in this oracle.
        close(dot_reference(f.x,f.x),1,2.0e-6,"oriented frame x is unit length");
        close(dot_reference(f.y,f.y),1,2.0e-6,"oriented frame y is unit length");
        close(dot_reference(f.z,f.z),1,2.0e-6,"oriented frame z is unit length");
        close(dot_reference(f.x,f.y),0,2.0e-7,"oriented frame x and y are orthogonal");
        close(dot_reference(f.x,f.z),0,2.0e-7,"oriented frame x and z are orthogonal");
        close(dot_reference(f.y,f.z),0,2.0e-7,"oriented frame y and z are orthogonal");
        close(dot_reference(cross(f.x,f.y),f.z),1,2.0e-6,
                "orientation preserves a right-handed frame");
        close(std::abs(dot_reference(f.z,normal)),1,2.0e-6,
                "orientation preserves the original shading-normal axis");
        const double length=std::sqrt(dot_reference(wo_world,wo_world));
        const Vec3 reconstructed=f.to_world(oriented.wo);
        close(reconstructed.x,wo_world.x/length,1.0e-6,"frame roundtrip preserves world x");
        close(reconstructed.y,wo_world.y/length,1.0e-6,"frame roundtrip preserves world y");
        close(reconstructed.z,wo_world.z/length,1.0e-6,"frame roundtrip preserves world z");
        close(oriented.wo.z,std::abs(dot_reference(normal,wo_world))/length,2.0e-7,
                "local cosine magnitude agrees with the world-space projection");
        ++count;
    };
    const std::array<Vec3,5> normals{{{0,0,1},{0,0,-1},{1,0,0},
        normalize({0.2f,0.7f,0.3f}),normalize({-0.7f,0.2f,-0.5f})}};
    for(Vec3 normal:normals) {
        const Frame source=make_frame(normal);
        for(float cosine:std::array<float,9>{-1.0f,-0.2f,-1.0e-5f,-1.0e-8f,
                                               0.0f,1.0e-8f,1.0e-5f,0.2f,1.0f}) {
            const float sine=sqrtf(1.0f-cosine*cosine);
            check(normal,source.to_world({0.6f*sine,0.8f*sine,cosine}));
        }
    }

    // A synthetic cancellation case illustrates why separately evaluated dot
    // products cannot certify the sign of the final local wo.z. Volatile stores
    // fix one legal rounding sequence, while explicit fma fixes another. This
    // is not a claim to reproduce the user's OptiX instruction sequence.
    const Vec3 normal{0.90521770715713501f,0.050454605370759964f,-0.42194125056266785f};
    const Vec3 wo_world{0.41568207740783691f,0.10107676684856415f,0.90387600660324097f};
    volatile float product_x=normal.x*wo_world.x;
    volatile float product_y=normal.y*wo_world.y;
    volatile float product_z=normal.z*wo_world.z;
    volatile float partial=product_x+product_y;
    volatile float unfused=partial+product_z;
    const float fused=std::fma(normal.z,wo_world.z,std::fma(normal.x,wo_world.x,product_y));
    require(unfused==0.0f&&fused<0.0f,
            "synthetic separately rounded projections disagree on the grazing sign");
    check(normal,wo_world);check(-normal,wo_world);
    std::cout<<"BSDF frame invariants: "<<count<<" world-direction cases passed\n";
}

void test_bsdf_frame_boundary_contracts() {
    const MetallicParameters material{0.3f,0.0f,0.7f};
    for(float zero:std::array<float,2>{0.0f,-0.0f}) {
        const auto tangent=make_bsdf_frame({0,0,1},{1,0,zero});
        require(tangent.wo.z==0.0f,"exact tangent is not moved into the BSDF hemisphere");
        require(sample_metallic(material,tangent.wo,0.25f,0.3f,0.6f).status==BsdfSampleStatus::Null,
                "exact tangent retains the metallic null-event contract");
        for(MaterialKind kind:std::array<MaterialKind,3>{MaterialKind::Diffuse,
                MaterialKind::GgxReflection,MaterialKind::Dielectric})
            require(sample_bsdf({kind,0.3f,0.5f,1.0f/1.5f},tangent.wo,0.25f,0.3f,0.6f).status==
                    BsdfSampleStatus::Null,"exact tangent retains every BSDF's null-event contract");
    }
    for(Vec3 local:std::array<Vec3,2>{{{0,0,-1},{1,0,-1.0e-8f}}}) {
        require(sample_metallic(material,local,0.25f,0.3f,0.6f).status==BsdfSampleStatus::Invalid,
                "negative local directions must not become null or valid samples");
        require(!evaluate_metallic(material,local,{0,0,1}).valid,
                "negative local directions remain invalid for evaluation");
    }
    const float nan=std::numeric_limits<float>::quiet_NaN();
    for(const auto input:std::array<std::array<Vec3,2>,4>{{
            {{{nan,0,1},{0,0,1}}},{{{0,0,1},{nan,0,1}}},
            {{{0,0,0},{0,0,1}}},{{{0,0,1},{0,0,0}}}}}) {
        const auto invalid=make_bsdf_frame(input[0],input[1]);
        require(sample_metallic(material,invalid.wo,0.25f,0.3f,0.6f).status==BsdfSampleStatus::Invalid,
                "invalid frame input remains a detectable BSDF failure");
    }
    const auto oriented=make_bsdf_frame({0,0,1},{0,0,-1});
    require(sample_metallic({-0.1f,0,0.7f},oriented.wo,0.25f,0.3f,0.6f).status==BsdfSampleStatus::Invalid,
            "frame orientation does not conceal invalid material parameters");
}
}

int main() {
    try {
        std::cout<<std::setprecision(10);
        test_lambert();test_ggx_analytic();test_ggx_general();
        test_grazing_and_reciprocity();test_dielectric();
        test_mis_and_roulette();test_rejection();
        test_existing_metallic_mixture();test_shading_normal_support();
        test_reported_bsdf_grazing();test_bsdf_frame_invariants();test_bsdf_frame_boundary_contracts();
        std::cout<<"M1 BSDF tests: 12 groups passed\n";
    } catch(const std::exception& e) {
        std::cerr<<"M1 BSDF test failed: "<<e.what()<<'\n';
        return 1;
    }
}
