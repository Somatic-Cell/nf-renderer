#include "m4_nf/asset.hpp"
#include "m4_nf/anisotropic.hpp"
#include "m4_nf/binary16.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

void require(bool p,const char* message){if(!p)throw std::runtime_error(message);}
int main(int argc,char**argv){
 try{
    using namespace m4nf;
    require(argc==2,"model.bin argument required");
    const auto asset=load_asset(argv[1]);const auto view=asset.view();
    for(unsigned b=0;b<65536;++b){
        const auto h=static_cast<std::uint16_t>(b);if((h&0x7c00u)==0x7c00u)continue;
        require(to_half_bits(from_half_bits(h))==h,"binary16 finite round trip");
    }
    require(to_half_bits(1.00048828125f)==0x3c00,"half ties to even lower");
    require(to_half_bits(1.00146484375f)==0x3c02,"half ties to even upper");
    std::cout<<"binary16 finite patterns and rounding ties: PASS\n";
    require(prepare(view,379,0).status==Status::condition_out_of_domain,"wavelength extrapolation accepted");
    require(prepare(view,550,1).status==Status::condition_out_of_domain,"polar endcap extrapolation accepted");
    require(prepare(view,550,-1).status==Status::condition_out_of_domain,"negative polar extrapolation accepted");
    std::cout<<"explicit domain rejection: PASS\n";
    for(int j=0;j<1000;++j){
        const float z=-0.999f+1.998f*float(j)/999;
        Frame f;const Vec3 ki{::sqrtf(1-z*z),z,0},axis{0,-1,0};
        require(nf_frame(ki,axis,f),"NF frame invalid");
        require(::fabsf(dot(f.x,f.y))<1e-6f&&::fabsf(dot(f.y,f.z))<1e-6f,"NF frame orthogonality");
        require(dot(cross(f.x,f.y),f.z)>.99999f,"NF frame handedness");
        require(dot(f.x,axis)>0,"NF projected axis sign");
    }
    std::cout<<"1000 oriented NF frames: PASS\n";
    // Scalar Jacobian of a deliberately nonuniform RQS, independent of NN.
    Spline s;float w[64],h[64];
    for(int j=0;j<64;++j){w[j]=1+float(j);h[j]=65-float(j);s.d[j]=.5f+.01f*float(j);}s.d[64]=2;
    require(knots(w,s.x)&&knots(h,s.y),"scalar knot setup");
    float inverse_error=0,logdet_error=0;
    unsigned conditioning_violations=0;
    for(int j=0;j<=4096;++j){
        const float x=float(j)/4096;float y,ld,z,ild;
        require(spline_map(s,x,false,y,ld)&&spline_map(s,y,true,z,ild),"RQS nonfinite");
        // Inverting a rounded y amplifies error by 1/f'(x). Use that
        // condition number, not a fixed absolute coordinate tolerance.
        const float coordinate_bound=8*std::numeric_limits<float>::epsilon()*std::max(1.0f,::expf(-ld));
        conditioning_violations+=::fabsf(z-x)>coordinate_bound;
        inverse_error=std::max(inverse_error,::fabsf(z-x));logdet_error=std::max(logdet_error,::fabsf(ld+ild));
    }
    require(conditioning_violations==0&&logdet_error<.01f,"RQS roundtrip conditioning bound");
    std::cout<<"4097 RQS round trips: PASS; x_error="<<inverse_error<<" logdet_error="<<logdet_error<<'\n';
    const auto ctx=prepare(view,550,.25f);require(ctx.status==Status::success,"physical preparation");
    const auto cs=cross_sections(view,550,.25f);
    require(cs.status==Status::success&&cs.scattering==ctx.csca&&cs.extinction==ctx.cext,"coefficient-only lookup");
    NativeMlp mlp{view.weights};std::mt19937 rng(8031);
    float max_symmetry=0,max_consistency=0;
    for(int j=0;j<2048;++j){
        const float u=(float(rng()>>9)+.5f)*0x1p-23f,v=(float(rng()>>9)+.5f)*0x1p-23f;
        const auto a=sample(u,v,ctx,mlp);require(a.status==Status::success,"NF sample failure");
        const auto b=log_prob(a.direction,ctx,mlp);require(b.status==Status::success,"NF sample evaluation failure");
        max_consistency=std::max(max_consistency,::fabsf(a.log_pdf-b.log_pdf));
        Vec3 reflected=a.direction;reflected.y=-reflected.y;
        const auto q=log_prob(reflected,ctx,mlp);require(q.status==Status::success,"reflection evaluation failure");
        max_symmetry=std::max(max_symmetry,::fabsf(q.log_pdf-b.log_pdf));
    }
    require(max_consistency<.002f&&max_symmetry<.002f,"NF consistency tolerance");
    std::cout<<"2048 learned NF sample/evaluate and mirror tests: PASS; log_consistency="<<max_consistency<<" symmetry="<<max_symmetry<<'\n';
    // Reference-direction coefficient and output-factor cancellation.
    const float rho=230.0f,ci=3.1e-6f,co=3.3e-6f,p=.02f;
    const float kernel=rho*ci*p,fp=rte_phase_factor(kernel,rho*co);
    require(::fabsf(fp-(ci/co)*p)<1e-8f,"incoming/outgoing coefficient placement");
    require(::fabsf(directional_coefficient(rho*co,ci,co)-rho*ci)<1e-9f,"reference coefficient scaling");
    require(::fabsf(camera_angular_weight(kernel,rho*co,.01f)-(ci/co)*(p/.01f))<1e-6f,"camera angular weight");
    std::cout<<"anisotropic coefficient placement: PASS\n";
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
