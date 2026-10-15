#include "m4_mie/asset.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace m4mie {
namespace {
struct Header {char magic[8];std::uint32_t version,nw,nb;float radius,temp,rho;};
static_assert(sizeof(Header)==32,"Unexpected MIE1LUT header padding");
void invalid(const std::string& reason){throw std::runtime_error("MIE1LUT: "+reason);}
template<typename T> void read(std::ifstream& f,std::vector<T>& a,std::size_t count){
    a.resize(count);
    if(count>0){f.read(reinterpret_cast<char*>(a.data()),static_cast<std::streamsize>(count*sizeof(T)));
        if(!f)invalid("truncated array");}
}
}
Asset load_asset(const std::string& path){
    static_assert(sizeof(float)==4&&sizeof(std::uint32_t)==4&&std::numeric_limits<float>::is_iec559,
                  "MIE1LUT requires IEEE binary32 and uint32");
    const std::uint16_t endian=1;
    if(*reinterpret_cast<const unsigned char*>(&endian)!=1)invalid("requires little-endian host");
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f)invalid("cannot open: "+path);
    const auto bytes=f.tellg();
    if(bytes<32||bytes>std::streamoff(1024LL*1024*1024))invalid("invalid file size");
    f.seekg(0,std::ios::beg);
    Header h{};f.read(reinterpret_cast<char*>(&h),sizeof(h));
    const char magic[8]{'M','I','E','1','L','U','T','\0'};
    if(std::memcmp(h.magic,magic,8)!=0||h.version!=1)invalid("wrong magic/version");
    if(h.nw<2||h.nw>2048||h.nb<2||h.nb>200000)invalid("invalid shape");
    const auto n=std::uint64_t(h.nw),b=std::uint64_t(h.nb);
    const auto expected=32ULL+4ULL*(n+(b+1ULL)+3ULL*n*b+3ULL*n);
    if(expected!=static_cast<std::uint64_t>(bytes))invalid("size does not match shape");
    if(!std::isfinite(h.radius)||std::fabs(h.radius-0.001f)>1e-9f||
       !std::isfinite(h.temp)||std::fabs(h.temp-293.15f)>0.01f||
       !std::isfinite(h.rho)||std::fabs(h.rho-998.20715f)>0.05f)
        invalid("unexpected physical parameters; supported is 1mm/20C reference");
    Asset a{};a.radius=h.radius;a.temperature=h.temp;a.density=h.rho;
    read(f,a.wavelengths,n);read(f,a.theta_edges,b+1);
    read(f,a.masses,n*b);read(f,a.alias_probs,n*b);read(f,a.aliases,n*b);
    read(f,a.qext,n);read(f,a.qsca,n);read(f,a.g,n);
    if(f.peek()!=std::char_traits<char>::eof())invalid("trailing data");
    const auto close=[](float x,float y,float tol){return std::fabs(x-y)<=tol;};
    if(!close(a.wavelengths.front(),380.0f,0.01f)||
       !close(a.wavelengths.back(),830.0f,0.01f))invalid("unexpected wavelength endpoints");
    for(std::size_t k=0;k<n;++k){
        if(!std::isfinite(a.wavelengths[k])||
           (k&&!(a.wavelengths[k]>a.wavelengths[k-1])))invalid("invalid wavelengths");
        if(!std::isfinite(a.qext[k])||!std::isfinite(a.qsca[k])||
           !(a.qext[k]>0)||!(a.qsca[k]>0)||
           a.qext[k]<a.qsca[k]-0.0003f*a.qext[k]||
           !std::isfinite(a.g[k])||std::fabs(a.g[k])>=1.0f)
            invalid("invalid efficiency/g");
    }
    if(!close(a.theta_edges.front(),0,1e-7f)||!close(a.theta_edges.back(),pi,1e-6f))
        invalid("invalid angular endpoints");
    std::vector<float> angles(b);
    for(std::size_t k=0;k<b;++k){
        const float th0=a.theta_edges[k],th1=a.theta_edges[k+1];
        if(!std::isfinite(th0)||!std::isfinite(th1)||!(th1>th0)||
           !(th0>=0)||th1>pi+1e-6f||
           !(solid_angle_width(th0,th1)>0)) invalid("invalid theta cell");
        angles[k]=0.5f*(cos_bounded(th0)+cos_bounded(th1));
    }
    for(std::size_t row=0;row<n;++row){
        double sum=0,moment=0;
        const auto begin=row*b;
        std::vector<double> reconstructed(b,0.0);
        for(std::size_t k=0;k<b;++k){
            const std::size_t index=begin+k;
            const float mass=a.masses[index],prob=a.alias_probs[index];
            const auto alias=a.aliases[index];
            if(!std::isfinite(mass)||mass<0||!std::isfinite(prob)||prob<0||prob>1||alias>=b)
                invalid("invalid alias/mass entries");
            sum+=mass;moment+=double(mass)*angles[k];
            reconstructed[k]+=double(prob)/double(b);
            reconstructed[alias]+=double(1.0f-prob)/double(b);
        }
        if(std::fabs(sum-1.0)>1e-5||std::fabs(moment/sum-a.g[row])>0.005)
            invalid("normalization or g changed");
        for(std::size_t k=0;k<b;++k)
            if(std::fabs(reconstructed[k]-a.masses[begin+k])>1e-6)
                invalid("alias table inconsistent with mass");
    }
    return a;
}
} // namespace m4mie
