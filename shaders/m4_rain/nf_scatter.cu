#include "../m4_nf/coop_mlp.cuh"
#include "../../include/launch_params.h"
#include "m4_rain/scattering.hpp"
extern "C" __constant__ __align__(16) LaunchParams optixLaunchParams;
// One direct-call frame, no nested direct callable or trace. The renderer sizes
// its stack from ALL compiled program groups, including this program.
extern "C" __device__ void __direct_callable__m4_nf(
    const m4rain::ScatterRequest* request,m4rain::ScatterResponse* output){
    const auto& r=optixLaunchParams.rain;
    if(r.parameters.backend==m4rain::Backend::Fp32)
        *output=m4rain::scatter(r.model,m4nf::NativeMlp{r.model.weights},
                              r.parameters.particleAxis,r.parameters.endcapPolicy,*request);
    else if(r.parameters.backend==m4rain::Backend::CoopVec)
        *output=m4rain::scatter(r.model,m4nf::CoopMlp{r.coop},
                              r.parameters.particleAxis,r.parameters.endcapPolicy,*request);
    else { *output={}; output->status=m4nf::Status::bad_input; }
}
