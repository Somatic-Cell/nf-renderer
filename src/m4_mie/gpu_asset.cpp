#include "m4_mie/gpu_asset.hpp"
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace m4mie {
namespace {
void check(CUresult code){if(code!=CUDA_SUCCESS){const char* name=nullptr;cuGetErrorString(code,&name);throw std::runtime_error(name?name:"CUDA Driver error");}}
}
void GpuAsset::release() noexcept {if(buffer_)cuMemFree(buffer_);buffer_=0;view={};}
void GpuAsset::upload(CUstream stream,const Asset& a){
    release();
    const auto v=a.view();
    if(!v.wavelengths_nm||v.nw<2||v.nbins<2)throw std::runtime_error("Invalid Mie host asset");
    std::vector<std::uint8_t> raw;raw.reserve(4ULL*(v.nw*4ULL+v.nbins+1ULL+3ULL*v.nw*v.nbins));
    std::size_t offsets[8]{};
    auto append=[&](unsigned id,const void* source,std::size_t bytes){
        offsets[id]=raw.size();
        raw.resize(raw.size()+bytes);
        std::memcpy(raw.data()+offsets[id],source,bytes);
    };
    const auto count=std::size_t(v.nw)*v.nbins;
    append(0,a.wavelengths.data(),a.wavelengths.size()*sizeof(float));
    append(1,a.theta_edges.data(),a.theta_edges.size()*sizeof(float));
    append(2,a.masses.data(),count*sizeof(float));
    append(3,a.alias_probs.data(),count*sizeof(float));
    append(4,a.aliases.data(),count*sizeof(std::uint32_t));
    append(5,a.qext.data(),a.qext.size()*sizeof(float));
    append(6,a.qsca.data(),a.qsca.size()*sizeof(float));
    append(7,a.g.data(),a.g.size()*sizeof(float));
    try{
        check(cuMemAlloc(&buffer_,raw.size()));
        check(cuMemcpyHtoD(buffer_,raw.data(),raw.size()));
        check(cuStreamSynchronize(stream));
        view=v;
        view.wavelengths_nm=reinterpret_cast<const float*>(buffer_+offsets[0]);
        view.theta_edges_rad=reinterpret_cast<const float*>(buffer_+offsets[1]);
        view.masses=reinterpret_cast<const float*>(buffer_+offsets[2]);
        view.alias_prob=reinterpret_cast<const float*>(buffer_+offsets[3]);
        view.alias=reinterpret_cast<const std::uint32_t*>(buffer_+offsets[4]);
        view.qext=reinterpret_cast<const float*>(buffer_+offsets[5]);
        view.qsca=reinterpret_cast<const float*>(buffer_+offsets[6]);
        view.g=reinterpret_cast<const float*>(buffer_+offsets[7]);
    }catch(...){release();throw;}
}
} // namespace m4mie
