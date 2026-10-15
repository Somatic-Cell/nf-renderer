#include "m4_nf/gpu_asset.hpp"
#include "m4_nf/flow.hpp"
#include "m4_nf/binary16.hpp"
#include <optix_stubs.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace m4nf {
namespace {
void check(CUresult r){if(r!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(s?s:"CUDA driver error");}}
void check(OptixResult r){if(r!=OPTIX_SUCCESS)throw std::runtime_error(std::string("OptiX: ")+optixGetErrorName(r));}
std::size_t align128(std::size_t n){
    constexpr std::size_t mask=127u;
    if(n>(std::numeric_limits<std::size_t>::max)()-mask)
        throw std::overflow_error("CoopVec network size overflow");
    return (n+mask)&~mask;
}
unsigned small(std::size_t n){if(n>(std::numeric_limits<unsigned>::max)())throw std::overflow_error("CoopVec offset overflow");return static_cast<unsigned>(n);}
}
GpuAsset::~GpuAsset(){release();}
void GpuAsset::release()noexcept{
    if(packed_)cuMemFree(packed_);
    if(fields_)cuMemFree(fields_);
    packed_=fields_=0;model={};coop={};
}
void GpuAsset::upload(OptixDeviceContext ctx,CUstream stream,const Asset& a){
    release();
    const auto nodes=a.wavelengths.size()*a.cosines.size();
    if(a.weights.size()!=weight_count||a.wavelengths.size()<2||a.cosines.size()<2||
       a.gs.size()!=nodes||a.scattering.size()!=nodes||a.extinction.size()!=nodes||a.angular_constants.size()!=194)
        throw std::runtime_error("Invalid host NF asset shape");
    CUdeviceptr source=0;
    try{
        // Concatenate the original FP32 data without demoting the flow inputs.
        const std::vector<float>* fields[]={&a.weights,&a.wavelengths,&a.cosines,&a.gs,&a.scattering,&a.extinction,&a.angular_constants};
        std::vector<float> data;std::size_t offsets[7]{};
        for(int j=0;j<7;++j){offsets[j]=data.size()*sizeof(float);data.insert(data.end(),fields[j]->begin(),fields[j]->end());}
        check(cuMemAlloc(&fields_,data.size()*sizeof(float)));
        check(cuMemcpyHtoD(fields_,data.data(),data.size()*sizeof(float)));
        check(cuStreamSynchronize(stream));
        model=a.view();
        const float** pointers[]={&model.weights,&model.wavelengths,&model.cosines,&model.gs,&model.scattering,&model.extinction,&model.angular_constants};
        for(int j=0;j<7;++j)*pointers[j]=reinterpret_cast<const float*>(fields_+offsets[j]);
        std::array<OptixCoopVecMatrixDescription,16> in{},out{};
        std::array<std::vector<std::uint16_t>,16> bias;
        std::vector<std::uint8_t> raw;
        std::size_t packed_bytes=0;
        for(int c=0;c<4;++c)for(int l=0;l<4;++l){
            const unsigned j=static_cast<unsigned>(4*c+l);
            const unsigned k=l?64u:16u,n=l<3?64u:(c%2?112u:208u);
            const unsigned original_k=static_cast<unsigned>(in_dimension(l)),original_n=static_cast<unsigned>(out_dimension(c,l));
            const float* w=a.weights.data()+weight_offset(c,l);const float* b=w+original_n*original_k;
            const std::size_t start=align128(raw.size());raw.resize(start+n*k*2u,0);
            for(unsigned r=0;r<original_n;++r)for(unsigned col=0;col<original_k;++col){
                const auto bits=to_half_bits(w[r*original_k+col]);
                if((bits&0x7c00u)==0x7c00u)throw std::runtime_error("FP16 weight overflow; no clipping");
                std::memcpy(raw.data()+start+2*(r*k+col),&bits,2);
            }
            bias[j].resize(n,0);
            for(unsigned r=0;r<original_n;++r){bias[j][r]=to_half_bits(b[r]);if((bias[j][r]&0x7c00u)==0x7c00u)throw std::runtime_error("FP16 bias overflow");}
            in[j].N=n;in[j].K=k;in[j].elementType=OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16;
            in[j].layout=OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
            in[j].rowColumnStrideInBytes=2*k;in[j].offsetInBytes=small(start);in[j].sizeInBytes=n*k*2;
            out[j]=in[j];out[j].layout=OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;out[j].rowColumnStrideInBytes=0;
            std::size_t size=0;
            check(optixCoopVecMatrixComputeSize(ctx,n,k,out[j].elementType,out[j].layout,0,&size));
            packed_bytes=align128(packed_bytes);out[j].offsetInBytes=small(packed_bytes);out[j].sizeInBytes=small(size);
            coop.matrix_offsets[j]=small(packed_bytes);packed_bytes+=size;
        }
        for(unsigned j=0;j<16;++j){packed_bytes=align128(packed_bytes);coop.bias_offsets[j]=small(packed_bytes);packed_bytes+=bias[j].size()*2;}
        // Align the END of each network as well as the start of every matrix
        // and bias. In particular, the last 112-element FP16 bias is 224 bytes:
        // an aligned starting offset alone leaves the total size 32 mod 64.
        // OptiX validates network strides even for numNetworks == 1 on the
        // reported driver. Keep both strides and their allocations aligned.
        const std::size_t input_network_stride=align128(raw.size());
        const std::size_t output_network_stride=align128(packed_bytes);
        raw.resize(input_network_stride,0);
        for(unsigned j=0;j<16;++j){
            const auto within=[](std::size_t offset,std::size_t bytes,std::size_t stride){
                return offset<=stride && bytes<=stride-offset;
            };
            if(in[j].offsetInBytes%64u || out[j].offsetInBytes%64u ||
               coop.bias_offsets[j]%16u ||
               !within(in[j].offsetInBytes,in[j].sizeInBytes,input_network_stride) ||
               !within(out[j].offsetInBytes,out[j].sizeInBytes,output_network_stride) ||
               !within(coop.bias_offsets[j],bias[j].size()*2u,output_network_stride))
                throw std::runtime_error("Invalid CoopVec network layout before conversion");
        }
        check(cuMemAlloc(&source,input_network_stride));
        check(cuMemAlloc(&packed_,output_network_stride));
        check(cuMemcpyHtoD(source,raw.data(),input_network_stride));
        OptixNetworkDescription in_net{},out_net{};
        in_net.layers=in.data();in_net.numLayers=16;out_net.layers=out.data();out_net.numLayers=16;
        check(optixCoopVecMatrixConvert(ctx,stream,1,&in_net,source,input_network_stride,
                                      &out_net,packed_,output_network_stride));
        // Conversion is asynchronous. Finish it before populating the bias
        // region in the same allocation with synchronous host-to-device copies.
        check(cuStreamSynchronize(stream));
        for(unsigned j=0;j<16;++j)check(cuMemcpyHtoD(packed_+coop.bias_offsets[j],bias[j].data(),bias[j].size()*2));
        check(cuStreamSynchronize(stream));check(cuMemFree(source));source=0;coop.base=packed_;
    }catch(...){
        // Host transfers are synchronous; only device matrix conversion is
        // asynchronous. Finish it before releasing device allocations.
        cuStreamSynchronize(stream);if(source)cuMemFree(source);release();throw;
    }
}
}
