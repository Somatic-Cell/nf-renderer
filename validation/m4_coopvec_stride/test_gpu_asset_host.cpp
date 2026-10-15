// Host API-mock regression for the REAL src/m4_nf/gpu_asset.cpp.
// Validates allocation/descriptor/stride bookkeeping and copied tensor bytes.
// It does NOT implement NVIDIA's optimal layout or run GPU inference.
#include "m4_nf/gpu_asset.hpp"
#include "m4_nf/binary16.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
std::map<CUdeviceptr,std::vector<unsigned char>> memory;
CUdeviceptr next_address=0x10000000ULL;
std::size_t last_input_stride=0,last_output_stride=0;
std::array<OptixCoopVecMatrixDescription,16> last_input{},last_output{};
unsigned size_call=0, size_variant=0, conversions=0;
bool pending=false, inject_convert_error=false;

void require(bool pass,const char* message){ if(!pass)throw std::runtime_error(message); }
unsigned char* at(CUdeviceptr address,std::size_t bytes){
    auto it=memory.upper_bound(address);
    if(it==memory.begin())throw std::runtime_error("mock: unknown device pointer");
    --it;
    const auto offset=address-it->first;
    require(offset<=it->second.size() && bytes<=it->second.size()-offset,
            "mock: out-of-bounds allocation access");
    return it->second.data()+offset;
}
std::uint16_t bits_at(CUdeviceptr address){
    std::uint16_t value=0;std::memcpy(&value,at(address,sizeof(value)),sizeof(value));return value;
}
[[maybe_unused]] void same_field(const float* device,const std::vector<float>& source){
    require(std::memcmp(at(reinterpret_cast<std::uintptr_t>(device),source.size()*sizeof(float)),
                        source.data(),source.size()*sizeof(float))==0,
            "FP32 field bytes changed during upload");
}
[[maybe_unused]] void check_contents(const m4nf::GpuAsset& gpu,const m4nf::Asset& asset){
    same_field(gpu.model.weights,asset.weights);same_field(gpu.model.wavelengths,asset.wavelengths);
    same_field(gpu.model.cosines,asset.cosines);same_field(gpu.model.gs,asset.gs);
    same_field(gpu.model.scattering,asset.scattering);same_field(gpu.model.extinction,asset.extinction);
    same_field(gpu.model.angular_constants,asset.angular_constants);
    for(unsigned c=0;c<4;++c)for(unsigned l=0;l<4;++l){
        const unsigned j=4*c+l,k=l?64u:16u,n=l<3?64u:(c%2?112u:208u);
        const auto ok=static_cast<unsigned>(m4nf::in_dimension(static_cast<int>(l)));
        const auto on=static_cast<unsigned>(m4nf::out_dimension(static_cast<int>(c),static_cast<int>(l)));
        const float* w=asset.weights.data()+m4nf::weight_offset(static_cast<int>(c),static_cast<int>(l));
        const float* bias=w+on*ok;
        require(gpu.coop.matrix_offsets[j]==last_output[j].offsetInBytes,"matrix offset changed");
        require(gpu.coop.matrix_offsets[j]%64u==0,"matrix not aligned");
        require(gpu.coop.bias_offsets[j]%16u==0,"bias not aligned");
        for(unsigned r=0;r<n;++r){
            for(unsigned col=0;col<k;++col){
                const auto expected=(r<on && col<ok)?m4nf::to_half_bits(w[r*ok+col]):std::uint16_t(0);
                require(bits_at(gpu.coop.base+gpu.coop.matrix_offsets[j]+2u*(r*k+col))==expected,
                        "padded FP16 matrix bytes changed");
            }
            const auto expected=r<on?m4nf::to_half_bits(bias[r]):std::uint16_t(0);
            require(bits_at(gpu.coop.base+gpu.coop.bias_offsets[j]+2u*r)==expected,
                    "padded FP16 bias bytes changed");
        }
        require(last_output[j].offsetInBytes+last_output[j].sizeInBytes<=gpu.coop.bias_offsets[0],
                "matrix/bias overlap");
        if(j+1<16){
            require(last_output[j].offsetInBytes+last_output[j].sizeInBytes<=last_output[j+1].offsetInBytes,
                    "matrix/matrix overlap");
            require(gpu.coop.bias_offsets[j]+2u*n<=gpu.coop.bias_offsets[j+1],"bias/bias overlap");
        }
    }
    require(memory.size()==2,"source allocation not released");
}
}

CUresult cuGetErrorString(CUresult,const char** s){*s="mock CUDA error";return CUDA_SUCCESS;}
CUresult cuMemAlloc(CUdeviceptr* ptr,std::size_t bytes){
    *ptr=next_address;next_address+=(bytes+255u)&~CUdeviceptr(255u);
    memory[*ptr]=std::vector<unsigned char>(bytes,0xCD);return CUDA_SUCCESS;
}
CUresult cuMemFree(CUdeviceptr ptr){
    if(pending)return CUDA_ERROR_INVALID_VALUE;
    return memory.erase(ptr)==1?CUDA_SUCCESS:CUDA_ERROR_INVALID_VALUE;
}
CUresult cuMemcpyHtoD(CUdeviceptr ptr,const void* source,std::size_t bytes){
    if(pending)return CUDA_ERROR_INVALID_VALUE; // Require the new explicit synchronization.
    std::memcpy(at(ptr,bytes),source,bytes);return CUDA_SUCCESS;
}
CUresult cuStreamSynchronize(CUstream){pending=false;return CUDA_SUCCESS;}
const char* optixGetErrorName(OptixResult r){return r==OPTIX_SUCCESS?"OPTIX_SUCCESS":"OPTIX_ERROR_INVALID_VALUE";}
OptixResult optixCoopVecMatrixComputeSize(OptixDeviceContext,unsigned n,unsigned k,
    OptixCoopVecElemType,OptixCoopVecMatrixLayout,std::size_t,std::size_t* bytes){
    // Synthetic sizes with the SDK's multiple-of-64 property, NOT an optimal-layout implementation.
    *bytes=std::size_t(n)*k*2u+(size_variant?64u*(size_call%3u):0u);++size_call;return OPTIX_SUCCESS;
}
OptixResult optixCoopVecMatrixConvert(OptixDeviceContext,CUstream,unsigned count,
    const OptixNetworkDescription* in,CUdeviceptr source,std::size_t input_stride,
    const OptixNetworkDescription* out,CUdeviceptr target,std::size_t output_stride){
    ++conversions;last_input_stride=input_stride;last_output_stride=output_stride;
    require(count==1 && in->numLayers==16 && out->numLayers==16,"unexpected network shape");
    if(output_stride%64u){
        std::cerr<<"MOCK: outputNetworkStrideInBytes (0x"<<std::hex<<output_stride
                 <<std::dec<<") must be 64 byte aligned.\n";
        return OPTIX_ERROR_INVALID_VALUE;
    }
    require(input_stride%64u==0,"input stride not aligned");
    require(input_stride%128u==0 && output_stride%128u==0,"packer's 128-byte policy not maintained");
    at(source,input_stride);at(target,output_stride);
    for(unsigned j=0;j<16;++j){
        last_input[j]=in->layers[j];last_output[j]=out->layers[j];
        const auto& a=last_input[j];const auto& b=last_output[j];
        require(a.N==b.N && a.K==b.K,"matrix dimensions changed");
        require(a.elementType==OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 && a.elementType==b.elementType,"dtype changed");
        require(a.layout==OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR &&
                b.layout==OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,"layout request changed");
        require(a.offsetInBytes%64u==0 && b.offsetInBytes%64u==0,"matrix offset not aligned");
        require(a.rowColumnStrideInBytes==a.K*2u && a.rowColumnStrideInBytes%16u==0,"input row stride");
        require(a.offsetInBytes+a.sizeInBytes<=input_stride && b.offsetInBytes+b.sizeInBytes<=output_stride,
                "matrix outside network stride");
        require(b.sizeInBytes>=a.sizeInBytes,"synthetic matrix result too small");
        std::memcpy(at(target+b.offsetInBytes,a.sizeInBytes),at(source+a.offsetInBytes,a.sizeInBytes),a.sizeInBytes);
    }
    if(inject_convert_error)return OPTIX_ERROR_INVALID_VALUE;
    pending=true;return OPTIX_SUCCESS;
}

int main(int argc,char** argv){
    try{
        require(argc==2,"usage: test_gpu_asset_host model.bin");
        const auto asset=m4nf::load_asset(argv[1]);
#ifdef M4_EXPECT_OLD_ALIGNMENT_FAILURE
        bool failed=false;
        try{m4nf::GpuAsset gpu;gpu.upload(nullptr,nullptr,asset);}catch(const std::runtime_error&){failed=true;}
        require(failed && last_output_stride==0x26be0u,"old alignment error was not reproduced");
        require(memory.empty(),"old error path leaks memory");
        std::cout<<"PASS: reproduced old 0x26BE0 output stride rejection with API mock; no GPU was used.\n";
#else
        for(unsigned variant=0;variant<2;++variant){
            size_variant=variant;size_call=0;
            {
                m4nf::GpuAsset gpu;gpu.upload(nullptr,nullptr,asset);check_contents(gpu,asset);
                if(!variant)require(last_output_stride==0x26c00u,"expected padded size 0x26C00");
                std::cout<<"PASS: size_variant="<<variant<<" input_stride="<<last_input_stride
                         <<" output_stride="<<last_output_stride<<"; all tensor bytes/offsets checked.\n";
                size_call=0;gpu.upload(nullptr,nullptr,asset);check_contents(gpu,asset);
                gpu.release();gpu.release();require(memory.empty(),"release/reupload leaked memory");
            }
            require(memory.empty(),"destructor leaked memory");
        }
        inject_convert_error=true;bool caught=false;
        try{m4nf::GpuAsset gpu;gpu.upload(nullptr,nullptr,asset);}catch(const std::runtime_error&){caught=true;}
        require(caught && memory.empty(),"conversion failure cleanup is not correct");
        std::cout<<"PASS: reupload, repeated release and injected conversion error cleanup.\n"
                 <<"HOST API MOCK ONLY: no CUDA runtime, OptiX JIT, optimal-layout conversion or GPU inference was executed.\n";
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
