#pragma once
#include <cuda_fp16.h>
#include <optix_device.h>
#include "m4_nf/flow.hpp"
#include "m4_nf/coop_types.hpp"

namespace m4nf {
// No teacher LUT, no FP64, and no activation change. Only the linear maps
// (inputs, weights, biases and outputs) use FP16. SiLU is evaluated in FP32.
struct CoopMlp {
    CoopView data;
    template<unsigned N,unsigned K>
    __device__ __forceinline__ OptixCoopVec<__half,N>
    linear(const OptixCoopVec<__half,K>& input,unsigned j)const{
        using Out=OptixCoopVec<__half,N>;using In=OptixCoopVec<__half,K>;
        return optixCoopVecMatMul<Out,In,OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
            OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,false,N,K,
            OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16>(
                input,static_cast<CUdeviceptr>(data.base),data.matrix_offsets[j],
                static_cast<CUdeviceptr>(data.base),data.bias_offsets[j]);
    }
    __device__ __forceinline__ bool activate(OptixCoopVec<__half,64>& x)const{
        bool valid=true;
        #pragma unroll
        for(unsigned i=0;i<64;++i){
            const float f=__half2float(x[i]);valid=valid&&finite(f);
            x[i]=__float2half_rn(silu(f));
        }
        return valid;
    }
    __device__ __forceinline__ bool operator()(int coupling,const float* x,float* output)const{
        OptixCoopVec<__half,16> input(__float2half_rn(0.0f));
        input[0]=__float2half_rn(x[0]);input[1]=__float2half_rn(x[1]);input[2]=__float2half_rn(x[2]);
        const unsigned j=static_cast<unsigned>(coupling)*4;
        auto h=linear<64,16>(input,j);bool valid=activate(h);
        h=linear<64,64>(h,j+1);valid=activate(h)&&valid;
        h=linear<64,64>(h,j+2);valid=activate(h)&&valid;
        if(coupling%2){
            const auto y=linear<112,64>(h,j+3);
            #pragma unroll
            for(int i=0;i<97;++i){output[i]=__half2float(y[i]);valid=valid&&finite(output[i]);}
        }else{
            const auto y=linear<208,64>(h,j+3);
            #pragma unroll
            for(int i=0;i<193;++i){output[i]=__half2float(y[i]);valid=valid&&finite(output[i]);}
        }
        return valid;
    }
};
}
