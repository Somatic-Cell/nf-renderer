#pragma once
#include "flow.hpp"
namespace m4nf {
// ALL directions below are physical propagation directions, not all-away BSDF
// arguments. P is the NF's outgoing-normalized SINGLE-PARTICLE density.
// The volumetric kernel is n*Csca(ki)*P(ko|ki), never n*Csca(ko)*P(ko|ki).
struct ForwardEvaluation {
    float phase=0,kernel=0,sigma_s_in=0,sigma_t_in=0;
    Status status=Status::bad_input;
};
template<class Mlp>
NF_HD ForwardEvaluation forward_kernel(const ModelView& m,const Mlp& mlp,Vec3 axis,
                                      Vec3 ki,Vec3 ko,float lambda,float number_density,
                                      EndcapPolicy policy=EndcapPolicy::Reject){
    ForwardEvaluation out;Frame f;
    if(!unit(ko)||!nf_frame(ki,axis,f)||!finite(number_density)||number_density<0)return out;
    const float cosine=direction_cosine(axis,f.z);
    const auto ctx=prepare(m,lambda,cosine,policy);
    const auto p=log_prob(f.local(ko),ctx,mlp);out.status=p.status;
    if(p.status!=Status::success)return out;
    out.phase=::expf(p.log_pdf);out.sigma_s_in=number_density*ctx.csca;
    out.sigma_t_in=number_density*ctx.cext;out.kernel=out.sigma_s_in*out.phase;
    if(!finite(out.kernel))out.status=Status::nonfinite;
    return out;
}
// Reference-direction parameterization. Setting sigma_ref(h)=n(h)*Cref
// gives sigma(h,k)=sigma_ref(h)*C(k)/Cref. No change to angular distribution.
NF_HD float directional_coefficient(float sigma_ref,float cross_section,float reference_cross_section){
    return sigma_ref>=0&&cross_section>0&&reference_cross_section>0?
        sigma_ref*(cross_section/reference_cross_section):nan_value();
}
// A collision was sampled using sigma_t(camera physical direction). q is the
// actual reverse-path proposal PDF. The kernel is evaluated with ki=-w,
// ko=-d. This does not assume that the learned NF is exactly reciprocal.
NF_HD float camera_angular_weight(float kernel,float sigma_t_camera,float proposal_pdf){
    return kernel>=0&&sigma_t_camera>0&&proposal_pdf>0?
        kernel/(sigma_t_camera*proposal_pdf):nan_value();
}
// Jakob's output-factor form: fp(ki -> ko) = K(ki,ko)/sigma_s(ko).
// It is incoming-normalized only when the supplied kernel/coefficient pair
// satisfies the physical balance relation. Multiplying by C_in/C_out does
// NOT impose reciprocity on an arbitrary learned P.
NF_HD float rte_phase_factor(float kernel,float sigma_s_out){
    return kernel>=0&&sigma_s_out>0?kernel/sigma_s_out:nan_value();
}
}
