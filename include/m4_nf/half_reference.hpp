#pragma once
#include "flow.hpp"
#include "binary16.hpp"
namespace m4nf {
// CPU sensitivity model: FP16 inputs/parameters/output, FP32 sequential FMA.
// Does NOT claim bit equivalence to Tensor Cores or the OptiX driver.
struct HalfReferenceMlp {
    const float* weights;
    bool operator()(int c,const float* input,float* result)const{
        float a[64]{},b[193]{};
        a[0]=round_half(input[0]);a[1]=round_half(input[1]);a[2]=round_half(input[2]);
        for(int layer=0;layer<4;++layer){
            const int n=out_dimension(c,layer),k=in_dimension(layer);
            const float* w=weights+weight_offset(c,layer);const float* bias=w+n*k;
            for(int j=0;j<n;++j){
                float sum=0;
                for(int i=0;i<k;++i)sum=::fmaf(round_half(w[j*k+i]),a[i],sum);
                b[j]=round_half(sum+round_half(bias[j]));if(!finite(b[j]))return false;
            }
            if(layer<3){for(int i=0;i<64;++i)a[i]=round_half(silu(b[i]));}
            else for(int i=0;i<n;++i)result[i]=b[i];
        }
        return true;
    }
};
}
