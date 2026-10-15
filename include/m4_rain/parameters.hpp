#pragma once
#include "m1/math.hpp"
#include "m4_nf/coop_types.hpp"
#include <cstdint>
#include <type_traits>
namespace m4rain {
// Finite world-space rain shaft in metres. Fixed particle orientation.
// Density is particles/m^3. Its vertical profile uses world Y, not altitude.
enum class DensityModel : std::uint32_t { Constant=0, ExponentialY=1 };
enum class Backend : std::uint32_t { Fp32=0, CoopVec=1 };
struct Parameters {
    std::uint32_t enabled;
    DensityModel densityModel;
    Backend backend;
    m4nf::EndcapPolicy endcapPolicy;
    m1::Vec3 boundsMin,boundsMax;
    m4nf::Vec3 particleAxis;
    float numberDensityAtMinY,densityScaleHeight;
};
inline Parameters default_parameters(){
    return {0,DensityModel::Constant,Backend::CoopVec,m4nf::EndcapPolicy::ConstantPhysical,
            {-3000.0f,0.0f,-3000.0f},{3000.0f,2000.0f,3000.0f},
            {0.0f,-1.0f,0.0f},100.0f,1000.0f};
}
struct DeviceData {
    Parameters parameters;
    m4nf::ModelView model;
    m4nf::CoopView coop;
    std::uint32_t selectedStep;
};
enum class ScatterOp : std::uint32_t { Evaluate=0, Sample=1 };
struct ScatterRequest {
    ScatterOp op;
    float wavelength,u0,u1;
    m4nf::Vec3 cameraDirection,nextDirection;
};
struct ScatterResponse {
    m4nf::Status status;
    m4nf::Vec3 direction;
    // value=Csca(-w)/Cext(-d)*P(-d|-w), pdf=P(w|d).
    float value,pdf,weight;
    float logPhysicalPhase,logProposal,incidentCosine,proposalCosine,cscaIncident,cextCamera;
};
static_assert(std::is_standard_layout<DeviceData>::value&&std::is_trivially_copyable<DeviceData>::value,"Rain ABI must be plain data");
static_assert(std::is_trivially_default_constructible<m4nf::ModelView>::value,"No launch-symbol defaults");
}
