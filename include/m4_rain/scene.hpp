#pragma once
// Host-only JSON layer; never included by a CUDA shader.
#include "tracking.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>
namespace m4rain {
struct SceneSettings {
    Parameters parameters=default_parameters();
    std::string modelFile="assets/m4_nf/best/model.bin";
};
inline nlohmann::ordered_json to_json(const SceneSettings& s){
    const auto& p=s.parameters;
    return {{"enabled",p.enabled!=0},{"modelFile",s.modelFile},
      {"backend",p.backend==Backend::CoopVec?"coopvec":"fp32"},
      {"endcapPolicy",p.endcapPolicy==m4nf::EndcapPolicy::ConstantPhysical?"constant_physical":"reject"},
      {"densityModel",p.densityModel==DensityModel::Constant?"constant":"exponential_y"},
      {"boundsMin",{p.boundsMin.x,p.boundsMin.y,p.boundsMin.z}},
      {"boundsMax",{p.boundsMax.x,p.boundsMax.y,p.boundsMax.z}},
      {"particleAxis",{p.particleAxis.x,p.particleAxis.y,p.particleAxis.z}},
      {"numberDensityAtMinY",p.numberDensityAtMinY},{"densityScaleHeight",p.densityScaleHeight}};
}
inline float json_float(const nlohmann::ordered_json& j,const char* name){
    if(!j.is_number())throw std::invalid_argument(std::string(name)+" must be numeric");
    const double x=j.get<double>();const float f=static_cast<float>(x);
    if(!std::isfinite(x)||!std::isfinite(f)||(x!=0.0&&f==0.0f))
        throw std::invalid_argument(std::string(name)+" is outside the finite FP32 range");
    return f;
}
inline void read(const nlohmann::ordered_json& j,SceneSettings& s){
    if(!j.is_object())throw std::invalid_argument("rain must be an object");
    static const std::unordered_set<std::string> names{"enabled","modelFile","backend","endcapPolicy",
        "densityModel","boundsMin","boundsMax","particleAxis","numberDensityAtMinY","densityScaleHeight"};
    for(auto i=j.begin();i!=j.end();++i)if(!names.count(i.key()))throw std::invalid_argument("Unknown rain field: "+i.key());
    auto& p=s.parameters;
    if(j.contains("enabled")){
        if(!j["enabled"].is_boolean())throw std::invalid_argument("rain.enabled must be boolean");
        p.enabled=j["enabled"].get<bool>()?1u:0u;
    }
    s.modelFile=j.value("modelFile",s.modelFile);
    const std::string backend=j.value("backend",p.backend==Backend::CoopVec?std::string("coopvec"):std::string("fp32"));
    if(backend=="coopvec")p.backend=Backend::CoopVec;
    else if(backend=="fp32")p.backend=Backend::Fp32;
    else throw std::invalid_argument("rain.backend must be fp32 or coopvec");
    const std::string policy=j.value("endcapPolicy",std::string("constant_physical"));
    if(policy=="constant_physical")p.endcapPolicy=m4nf::EndcapPolicy::ConstantPhysical;
    else if(policy=="reject")p.endcapPolicy=m4nf::EndcapPolicy::Reject;
    else throw std::invalid_argument("rain.endcapPolicy must be constant_physical or reject");
    const std::string profile=j.value("densityModel",std::string("constant"));
    if(profile=="constant")p.densityModel=DensityModel::Constant;
    else if(profile=="exponential_y")p.densityModel=DensityModel::ExponentialY;
    else throw std::invalid_argument("rain.densityModel must be constant or exponential_y");
    auto vector=[&](const char* key){
        const auto& v=j.at(key);
        if(!v.is_array()||v.size()!=3)throw std::invalid_argument(std::string("rain.")+key+" must have 3 components");
        return m1::Vec3{json_float(v[0],key),json_float(v[1],key),json_float(v[2],key)};
    };
    if(j.contains("boundsMin"))p.boundsMin=vector("boundsMin");
    if(j.contains("boundsMax"))p.boundsMax=vector("boundsMax");
    if(j.contains("particleAxis")){const auto v=vector("particleAxis");p.particleAxis={v.x,v.y,v.z};}
    if(j.contains("numberDensityAtMinY"))p.numberDensityAtMinY=json_float(j["numberDensityAtMinY"],"numberDensityAtMinY");
    if(j.contains("densityScaleHeight"))p.densityScaleHeight=json_float(j["densityScaleHeight"],"densityScaleHeight");
    if(validate(p)!=Status::Ok)throw std::invalid_argument("Invalid rain geometry/density/particle axis");
    if(p.enabled&&s.modelFile.empty())throw std::invalid_argument("rain.modelFile must not be empty");
}
inline void validate_scene(const SceneSettings& s,const m3::TrackingParameters& a,float lo,float hi){
    if(validate(s.parameters)!=Status::Ok)throw std::invalid_argument("Invalid rain parameters");
    if(!std::isfinite(lo)||!std::isfinite(hi)||lo<=0||!(lo<hi))throw std::invalid_argument("Invalid spectral wavelength range");
    const auto& p=s.parameters;if(!p.enabled)return;
    if(!a.enabled)throw std::invalid_argument("rain.enabled requires rayleigh.enabled (beta550 may be zero)");
    if(lo<380.0f||hi>830.0f)throw std::invalid_argument("Rain NF requires spectrum.wavelengthMinNm >= 380 and wavelengthMaxNm <= 830; no wavelength extrapolation");
    if(p.boundsMin.y<a.groundY)throw std::invalid_argument("The supported rain box must lie above world Y=groundY");
    for(unsigned k=0;k<8;++k){
        const m1::Vec3 corner{(k&1)?p.boundsMax.x:p.boundsMin.x,(k&2)?p.boundsMax.y:p.boundsMin.y,(k&4)?p.boundsMax.z:p.boundsMin.z};
        const auto h=mt::height(a,corner);
        if(h.status!=Status::Ok||h.height_m<0||h.height_m>a.atmosphereHeight)
            throw std::invalid_argument("The rain box must fit entirely within the atmosphere");
    }
}
}
