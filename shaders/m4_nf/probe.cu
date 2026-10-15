#include "coop_mlp.cuh"
extern "C" { __constant__ m4nf::ProbeParams params; }

template<class Mlp>
__device__ __forceinline__ m4nf::ProbeOutput evaluate(const m4nf::Query& q,const Mlp& mlp){
    using namespace m4nf;
    ProbeOutput out{};const auto context=prepare(params.model,q.lambda,q.cosine);
    if(q.op==0){
        const auto p=log_prob(Vec3{q.a,q.b,q.c},context,mlp);
        out.status=static_cast<unsigned>(p.status);out.log_pdf=p.log_pdf;out.reevaluated_log_pdf=p.log_pdf;
    }else if(q.op==1){
        const auto s=sample(q.a,q.b,context,mlp);
        const auto p=log_prob(s.direction,context,mlp);
        out.status=static_cast<unsigned>(s.status==Status::success?p.status:s.status);
        out.log_pdf=s.log_pdf;out.x=s.direction.x;out.y=s.direction.y;out.z=s.direction.z;
        out.reevaluated_log_pdf=p.log_pdf;
    }else out.status=static_cast<unsigned>(Status::bad_input);
    return out;
}
extern "C" __global__ void __raygen__m4_nf_probe(){
    const unsigned index=optixGetLaunchIndex().x;
    if(index>=params.count)return;
    const auto query=params.queries[index];
    if(params.backend==0)params.output[index]=evaluate(query,m4nf::NativeMlp{params.model.weights});
    else params.output[index]=evaluate(query,m4nf::CoopMlp{params.coop});
}
