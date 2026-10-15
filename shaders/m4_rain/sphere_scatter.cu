#include "../../include/launch_params.h"
#include "m4_mie/table.hpp"
#include "m4_rain/parameters.hpp"
// External OptiX launch symbol: DO NOT use extern "C" { ... }, which defines
// a __constant__ object and forces unsupported device-side initialization.
extern "C" __constant__ __align__(16) LaunchParams optixLaunchParams;
extern "C" __device__ void __direct_callable__m4_mie(
    const m4rain::ScatterRequest* request,m4rain::ScatterResponse* output){
    m4rain::ScatterResponse result{};
    result.status=m4nf::Status::bad_input;
    const auto& rain=optixLaunchParams.rain;
    const auto& d=request->cameraDirection;
    const m4mie::Vec3 camera{d.x,d.y,d.z};
    m4mie::Vec3 direction{};
    m4mie::Response evaluated{};
    if(request->op==m4rain::ScatterOp::Sample){
        const auto s=m4mie::sample(rain.mie,camera,request->wavelength,
                                  rain.parameters.crossSectionMode,request->u0,request->u1);
        if(!s.valid){*output=result;return;}
        direction=s.direction;
        result.pdf=s.pdf_sr_inv;
        result.cscaIncident=s.csca_m2;
        result.cextCamera=s.cext_m2;
    }else if(request->op==m4rain::ScatterOp::Evaluate){
        const auto& w=request->nextDirection;
        direction={w.x,w.y,w.z};
    }else{*output=result;return;}
    evaluated=m4mie::evaluate(rain.mie,camera,direction,request->wavelength,
                              rain.parameters.crossSectionMode);
    if(!evaluated.valid||!(evaluated.phase_sr_inv>0)||!(evaluated.cext_m2>0)){
        *output=result;return;
    }
    const float albedo=evaluated.csca_m2/evaluated.cext_m2;
    result.direction={direction.x,direction.y,direction.z};
    result.pdf=evaluated.phase_sr_inv;
    result.value=albedo*result.pdf;
    result.weight=albedo;
    result.cscaIncident=evaluated.csca_m2;
    result.cextCamera=evaluated.cext_m2;
    result.logPhysicalPhase=::logf(result.pdf);
    result.logProposal=result.logPhysicalPhase;
    result.status=(m4nf::unit(result.direction)&&m4nf::finite(result.value)&&result.value>0&&
                   m4nf::finite(result.weight)&&result.weight>0)?m4nf::Status::success:m4nf::Status::nonfinite;
    *output=result;
}
