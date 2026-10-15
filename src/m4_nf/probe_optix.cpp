// Standalone OptiX inference probe: no triangles, no atmosphere, no teacher data.
#include "m4_nf/gpu_asset.hpp"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(CUresult r){if(r!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(s?s:"CUDA error");}}
void check(OptixResult r){if(r!=OPTIX_SUCCESS)throw std::runtime_error(std::string("OptiX: ")+optixGetErrorName(r));}
void log_message(unsigned level,const char* tag,const char* text,void*){std::cerr<<"["<<level<<"]["<<tag<<"] "<<text<<'\n';}
struct Session {
    CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;bool primary=false,pushed=false;
    OptixDeviceContext optix=nullptr;OptixModule module=nullptr;OptixProgramGroup group=nullptr;OptixPipeline pipeline=nullptr;
    CUdeviceptr queries=0,output=0,params=0,record=0;CUevent start=nullptr,stop=nullptr;
    ~Session(){
        if(stream)cuStreamSynchronize(stream);
        if(start)cuEventDestroy(start);if(stop)cuEventDestroy(stop);
        if(record)cuMemFree(record);if(params)cuMemFree(params);if(output)cuMemFree(output);if(queries)cuMemFree(queries);
        if(pipeline)optixPipelineDestroy(pipeline);if(group)optixProgramGroupDestroy(group);if(module)optixModuleDestroy(module);
        if(optix)optixDeviceContextDestroy(optix);if(stream)cuStreamDestroy(stream);
        if(pushed){CUcontext previous=nullptr;cuCtxPopCurrent(&previous);}
        if(primary)cuDevicePrimaryCtxRelease(device);
    }
    Session()=default;Session(const Session&)=delete;Session& operator=(const Session&)=delete;
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord{char header[OPTIX_SBT_RECORD_HEADER_SIZE];};
}
int main(int argc,char** argv){
    try{
        if(argc!=6&&argc!=7){
            std::cerr<<"Usage: m4_nf_probe_gpu model.bin probe.optixir queries.txt output.txt fp32|coopvec [device]\n";return 2;
        }
        const std::string mode=argv[5];if(mode!="fp32"&&mode!="coopvec")throw std::runtime_error("Unknown backend");
        const int ordinal=argc==7?std::stoi(argv[6]):0;
        const auto asset=m4nf::load_asset(argv[1]);
        std::ifstream in(argv[3]);if(!in)throw std::runtime_error("Cannot open queries");
        std::vector<m4nf::Query> queries;m4nf::Query q;
        while(in>>q.op>>q.lambda>>q.cosine>>q.a>>q.b>>q.c)queries.push_back(q);
        if(!in.eof()||queries.empty()||queries.size()>(std::numeric_limits<unsigned>::max)())throw std::runtime_error("Invalid query file");
        std::ifstream file(argv[2],std::ios::binary);if(!file)throw std::runtime_error("Cannot read OptiX IR");
        const std::string ir((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        Session s;check(cuInit(0));check(cuDeviceGet(&s.device,ordinal));
        check(cuDevicePrimaryCtxRetain(&s.context,s.device));s.primary=true;check(cuCtxPushCurrent(s.context));s.pushed=true;
        check(cuStreamCreate(&s.stream,CU_STREAM_DEFAULT));check(optixInit());
        OptixDeviceContextOptions context_options{};context_options.logCallbackFunction=log_message;context_options.logCallbackLevel=3;
        check(optixDeviceContextCreate(s.context,&context_options,&s.optix));
        // gpu_asset is destroyed before Session; the CUDA context stays valid.
        m4nf::GpuAsset gpu_asset;gpu_asset.upload(s.optix,s.stream,asset);
        OptixModuleCompileOptions compile{};
        compile.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        compile.optLevel=OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;compile.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        OptixPipelineCompileOptions pipeline_options{};
        pipeline_options.usesMotionBlur=0;pipeline_options.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipeline_options.numPayloadValues=0;pipeline_options.numAttributeValues=2;
        pipeline_options.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;pipeline_options.pipelineLaunchParamsVariableName="params";
        char log[16384];std::size_t log_size=sizeof log;
        auto r=optixModuleCreate(s.optix,&compile,&pipeline_options,ir.data(),ir.size(),log,&log_size,&s.module);
        if(log_size>1)std::cerr<<log<<'\n';check(r);
        OptixProgramGroupDesc desc{};desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module=s.module;desc.raygen.entryFunctionName="__raygen__m4_nf_probe";
        OptixProgramGroupOptions group_options{};log_size=sizeof log;
        r=optixProgramGroupCreate(s.optix,&desc,1,&group_options,log,&log_size,&s.group);
        if(log_size>1)std::cerr<<log<<'\n';check(r);
        OptixPipelineLinkOptions link{};link.maxTraceDepth=1;log_size=sizeof log;
        r=optixPipelineCreate(s.optix,&pipeline_options,&link,&s.group,1,log,&log_size,&s.pipeline);
        if(log_size>1)std::cerr<<log<<'\n';check(r);
        OptixStackSizes sizes{};check(optixUtilAccumulateStackSizes(s.group,&sizes,s.pipeline));
        unsigned direct_trav=0,direct_state=0,continuation=0;
        check(optixUtilComputeStackSizes(&sizes,1,0,0,&direct_trav,&direct_state,&continuation));
        check(optixPipelineSetStackSize(s.pipeline,direct_trav,direct_state,continuation,1));
        RaygenRecord rg{};check(optixSbtRecordPackHeader(s.group,&rg));
        check(cuMemAlloc(&s.record,sizeof rg));check(cuMemcpyHtoD(s.record,&rg,sizeof rg));
        OptixShaderBindingTable sbt{};sbt.raygenRecord=s.record;
        std::vector<m4nf::ProbeOutput> output(queries.size());
        check(cuMemAlloc(&s.queries,queries.size()*sizeof(m4nf::Query)));
        check(cuMemAlloc(&s.output,output.size()*sizeof(m4nf::ProbeOutput)));
        check(cuMemcpyHtoD(s.queries,queries.data(),queries.size()*sizeof(m4nf::Query)));
        m4nf::ProbeParams params;params.model=gpu_asset.model;params.coop=gpu_asset.coop;
        params.queries=reinterpret_cast<const m4nf::Query*>(s.queries);params.output=reinterpret_cast<m4nf::ProbeOutput*>(s.output);
        params.count=static_cast<unsigned>(queries.size());params.backend=mode=="coopvec"?1u:0u;
        check(cuMemAlloc(&s.params,sizeof params));check(cuMemcpyHtoD(s.params,&params,sizeof params));
        check(optixLaunch(s.pipeline,s.stream,s.params,sizeof params,&sbt,params.count,1,1));check(cuStreamSynchronize(s.stream));
        check(cuEventCreate(&s.start,CU_EVENT_DEFAULT));check(cuEventCreate(&s.stop,CU_EVENT_DEFAULT));
        check(cuEventRecord(s.start,s.stream));
        check(optixLaunch(s.pipeline,s.stream,s.params,sizeof params,&sbt,params.count,1,1));
        check(cuEventRecord(s.stop,s.stream));check(cuEventSynchronize(s.stop));
        float ms=0;check(cuEventElapsedTime(&ms,s.start,s.stop));
        check(cuMemcpyDtoH(output.data(),s.output,output.size()*sizeof(m4nf::ProbeOutput)));
        std::ofstream out(argv[4]);if(!out)throw std::runtime_error("Cannot open output");out<<std::setprecision(9);
        unsigned invalid=0;
        for(const auto& o:output){invalid+=o.status!=0;out<<o.status<<' '<<o.log_pdf<<' '<<o.x<<' '<<o.y<<' '<<o.z<<' '<<o.reevaluated_log_pdf<<'\n';}
        std::cout<<"backend="<<mode<<" queries="<<queries.size()<<" warm_launch_ms="<<ms<<" invalid="<<invalid<<'\n';
        return invalid?3:0;
    }catch(const std::exception& e){std::cerr<<"M4 NF PROBE ERROR: "<<e.what()<<'\n';return 1;}
}
