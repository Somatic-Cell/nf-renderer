#pragma once
#include "parameters.hpp"
#include "m4_nf/flow.hpp"
namespace m4rain {
// Used unchanged by CPU tests and the OptiX direct callable. No assumption of
// exact reciprocity of the learned NF, no kernel symmetrization/renormalization.
template<class Mlp>
NF_HD ScatterResponse scatter(const m4nf::ModelView& model,const Mlp& mlp,
    m4nf::Vec3 axis,m4nf::EndcapPolicy policy,const ScatterRequest& request){
    using namespace m4nf;
    ScatterResponse out{};out.status=m4nf::Status::bad_input;
    Frame camera_frame;
    if(!nf_frame(request.cameraDirection,axis,camera_frame))return out;
    const Vec3 d=camera_frame.z;const float cd=direction_cosine(axis,d);
    out.proposalCosine=cd;
    const Context proposal=prepare(model,request.wavelength,cd,policy);
    if(proposal.status!=m4nf::Status::success){out.status=proposal.status;return out;}
    Vec3 w;float logq;
    if(request.op==ScatterOp::Sample){
        const Sample s=sample(request.u0,request.u1,proposal,mlp);
        if(s.status!=m4nf::Status::success){out.status=s.status;return out;}
        w=normalize(camera_frame.world(s.direction));logq=s.log_pdf;
    }else if(request.op==ScatterOp::Evaluate){
        if(!unit(request.nextDirection))return out;
        w=normalize(request.nextDirection);
        const Result q=log_prob(camera_frame.local(w),proposal,mlp);
        if(q.status!=m4nf::Status::success){out.status=q.status;return out;}
        logq=q.log_pdf;
    }else return out;
    out.direction=w;out.logProposal=logq;
    const auto camera_cs=cross_sections(model,request.wavelength,direction_cosine(axis,-d),policy);
    if(camera_cs.status!=m4nf::Status::success){out.status=camera_cs.status;return out;}
    Frame physical_frame;
    if(!nf_frame(-w,axis,physical_frame))return out;
    const float ci=direction_cosine(axis,physical_frame.z);out.incidentCosine=ci;
    const Context physical=prepare(model,request.wavelength,ci,policy);
    const Result p=log_prob(physical_frame.local(-d),physical,mlp);
    if(p.status!=m4nf::Status::success){out.status=p.status;return out;}
    out.logPhysicalPhase=p.log_pdf;out.cscaIncident=physical.csca;out.cextCamera=camera_cs.extinction;
    // n(x) cancels against the selected rain event's n(x)*Cext(-d).
    const float logv=p.log_pdf+::logf(physical.csca)-::logf(camera_cs.extinction);
    out.value=::expf(logv);out.pdf=::expf(logq);out.weight=::expf(logv-logq);
    out.status=(unit(w)&&finite(out.value)&&out.value>0&&finite(out.pdf)&&out.pdf>0&&
                finite(out.weight)&&out.weight>0)?m4nf::Status::success:m4nf::Status::nonfinite;
    return out;
}
}
