#include "m4_mie/asset.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(int argc,char** argv){
 if(argc!=2)return 99;
 const auto a=m4mie::load_asset(argv[1]);const auto t=a.view();
 require(t.nw==91 && t.nbins==15450,"actual uploaded LUT shape");
 require(std::abs(t.radius_m-.001f)<1e-9f,"radius");
 std::mt19937 rng(1234);std::uniform_real_distribution<float> u(0,1);
 std::uint32_t samples=0, bin_mismatches=0, bin_mismatches_small=0;float max_relative=0;
 const m4mie::Vec3 d{0.1f,0.2f,0.974679434f};
 for(float wl: {380.0f,450.25f,550.0f,650.76f,830.0f})for(auto mode:{m4mie::CrossSection::GeometricMatched,m4mie::CrossSection::PhysicalMie}){
   for(int i=0;i<8192;++i){
     auto s=m4mie::sample(t,d,wl,mode,std::fmax(u(rng),1e-6f),u(rng));
     require(s.valid,"Mie sample invalid");
     float angle=0;
     require(m4mie::scattering_angle(d,s.direction,angle),"invalid sampled scattering angle");
     auto recovered=m4mie::bin_for_theta(t,angle);
     if(recovered!=s.bin){++bin_mismatches;if(angle<0.01f)++bin_mismatches_small;}
     auto ev=m4mie::evaluate(t,d,s.direction,wl,mode);
     require(ev.valid && ev.phase_sr_inv>0,"Mie sample eval invalid");
     const float rel=std::abs(ev.phase_sr_inv-s.pdf_sr_inv)/ev.phase_sr_inv;
     max_relative=std::max(max_relative,rel);
     require(rel<=2e-6f,"sample PDF mismatch");
     ++samples;
   }
 }
 require(bin_mismatches < 128,"FP32 emitted direction crossed too many discretization boundaries");
 std::cout<<"success: nw="<<t.nw<<" cells="<<t.nbins<<" samples="<<samples
          <<" max_relative_pdf_delta="<<max_relative <<" bin_mismatch="<<bin_mismatches <<" near_0.01="<<bin_mismatches_small<<"\n";
}
