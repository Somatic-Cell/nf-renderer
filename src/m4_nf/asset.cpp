#include "m4_nf/asset.hpp"
#include <array>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace m4nf {
ModelView Asset::view()const{
    return {weights.data(),wavelengths.data(),cosines.data(),gs.data(),scattering.data(),extinction.data(),
            angular_constants.data(),static_cast<unsigned>(wavelengths.size()),static_cast<unsigned>(cosines.size()),lambda_min,lambda_max};
}
namespace {
void bytes(std::ifstream& f,void* p,std::size_t n){
    if(!f.read(static_cast<char*>(p),static_cast<std::streamsize>(n)))throw std::runtime_error("Truncated NF asset");
}
std::uint32_t u32(std::ifstream& f){unsigned char b[4];bytes(f,b,4);return b[0]|(std::uint32_t(b[1])<<8)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[3])<<24);}
float f32(std::ifstream& f){const auto b=u32(f);float v;static_assert(sizeof v==4);std::memcpy(&v,&b,4);return v;}
void floats(std::ifstream& f,std::vector<float>& v,std::size_t n){v.resize(n);for(float& x:v){x=f32(f);if(!std::isfinite(x))throw std::runtime_error("Nonfinite NF asset field");}}
void axis(const std::vector<float>& a){for(std::size_t j=1;j<a.size();++j)if(!(a[j]>a[j-1]))throw std::runtime_error("Unsorted/duplicate NF axis");}
}
Asset load_asset(const std::string& filename){
    std::ifstream f(filename,std::ios::binary);if(!f)throw std::runtime_error("Cannot open NF asset: "+filename);
    std::array<char,8> magic{};bytes(f,magic.data(),8);
    const std::array<char,8> expected{'N','F','H','G','4','A','1','\0'};
    if(magic!=expected||u32(f)!=1)throw std::runtime_error("Wrong NF asset format/version");
    const auto nw=u32(f),nl=u32(f),nc=u32(f);
    if(nw!=weight_count||nl<2||nl>4096||nc<2||nc>4096)throw std::runtime_error("Unsupported NF dimensions");
    Asset m;m.selected_step=u32(f);m.lambda_min=f32(f);m.lambda_max=f32(f);
    if(m.lambda_min!=380.0f||m.lambda_max!=830.0f||f32(f)!=min_bin||f32(f)!=min_derivative)
        throw std::runtime_error("Unsupported NF config; no implicit conversion");
    floats(f,m.weights,nw);floats(f,m.wavelengths,nl);floats(f,m.cosines,nc);
    const std::size_t n=std::size_t(nl)*nc;
    floats(f,m.gs,n);floats(f,m.scattering,n);floats(f,m.extinction,n);floats(f,m.angular_constants,194);
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing bytes in NF asset");
    axis(m.wavelengths);axis(m.cosines);
    if(m.wavelengths.front()!=380||m.wavelengths.back()!=830||m.cosines.front()<-1||m.cosines.back()>1)
        throw std::runtime_error("Invalid physical axes");
    for(std::size_t i=0;i<n;++i)if(std::fabs(m.gs[i])>=1||!(m.scattering[i]>0)||m.scattering[i]!=m.extinction[i])
        throw std::runtime_error("Only geometric_nonabsorbing cross sections are supported");
    return m;
}
} // namespace m4nf
