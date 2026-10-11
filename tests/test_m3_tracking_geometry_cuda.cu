// Standalone regression with a real CUDA device, independent of OptiX/windowing.
// From the repository root, for example:
// nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 --fmad=true --ftz=false -Iinclude tests/test_m3_tracking_geometry_cuda.cu -o m3_tracking_geometry_cuda_check
// Run the executable. Exit 0 = PASS, 1 = FAIL, 77 = SKIP (no usable CUDA device).
// The normal default FMA setting is intentional. Do not compile this test with
// --fmad=false: it must exercise the production FloatPair rounding barriers.

#include <cuda_runtime.h>
#include "m3/tracking_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace g=m3::tracking;

struct RayInput { g::RaySegment ray{}; bool test_forged_endpoint=false; };
struct RayOutput {
    g::ShellClip clipped{};
    g::PlanetHit planet{},forged{};
    g::ShellClip outward{},inward{};
    g::Status interior_point_status=g::Status::Ok;
};
struct EftInput { g::FloatPair a{},b{}; float c=0; };
struct EftOutput { g::FloatPair product{},pair_product{},sum{},recovered{}; };

__global__ void geometry_kernel(const m3::TrackingParameters* parameters,
    const RayInput* inputs,RayOutput* outputs,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto p=*parameters;
    const auto input=inputs[i];
    RayOutput out;
    out.clipped=g::clip(p,input.ray);
    if(out.clipped.status==g::Status::Ok&&out.clipped.hits_ground) {
        out.planet=g::planet_hit(p,input.ray,out.clipped);
        if(out.planet.status==g::Status::Ok) {
            const g::RaySegment outward{out.planet.position,out.planet.normal,HUGE_VALF};
            const g::RaySegment inward{out.planet.position,-out.planet.normal,HUGE_VALF};
            out.outward=g::clip(p,outward);
            out.inward=g::clip(p,inward);
        }
        if(input.test_forged_endpoint) {
            auto forged=out.clipped;
            forged.ground_parameter=g::add(forged.ground_parameter,1.0f);
            out.forged=g::planet_hit(p,input.ray,forged);
        }
    }
    // This is genuinely inside the planet, not an outside point rounded in.
    out.interior_point_status=g::represent_medium_point(p,
        g::Vec3{0.0f,p.groundY-1.0f,0.0f}).status;
    outputs[i]=out;
}

__global__ void eft_kernel(const EftInput* inputs,EftOutput* outputs,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    // Inputs come from cudaMemcpy, so NVCC cannot constant-fold this into the
    // host oracle. Compose the real inlined production EFTs as point_at does.
    const auto input=inputs[i];
    EftOutput out;
    out.product=g::product(input.a.hi,input.b.hi);
    out.pair_product=g::mul(input.a,input.b);
    out.sum=g::add(out.pair_product,input.c);
    out.recovered=g::sub(out.sum,input.c);
    outputs[i]=out;
}

void check(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
void cuda_check(cudaError_t error,const char* operation) {
    if(error!=cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
template<class T> struct DeviceArray {
    T* pointer=nullptr;
    explicit DeviceArray(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer),count*sizeof(T)),"cudaMalloc");
    }
    ~DeviceArray() { if(pointer)cudaFree(pointer); }
    DeviceArray(const DeviceArray&)=delete;
    DeviceArray& operator=(const DeviceArray&)=delete;
};
long double exact(g::FloatPair a) { return static_cast<long double>(a.hi)+a.lo; }
g::FloatPair split(long double a) {
    const float hi=static_cast<float>(a);
    return {hi,static_cast<float>(a-hi)};
}
void near(long double actual,long double expected,long double absolute,long double relative,
    const std::string& message) {
    const long double error=std::abs(actual-expected);
    if(!(error<=absolute+relative*std::abs(expected))) {
        std::cerr<<std::setprecision(22)<<message<<": actual="<<actual<<" expected="<<expected
                 <<" error="<<error<<'\n';
        throw std::runtime_error(message);
    }
}

// Host-only long-double oracle. No production compensated arithmetic, clip,
// planet_hit, or CPU atmosphere integrator is used by the reference functions.
struct RefRoots { bool hit=false; long double first=0,last=0,half_b=0; };
RefRoots reference_roots(const m3::TrackingParameters& p,const g::RaySegment& ray) {
    const long double x=ray.origin.x,y=static_cast<long double>(ray.origin.y)-p.groundY;
    const long double z=ray.origin.z,r=p.planetRadius;
    const long double dx=ray.direction.x,dy=ray.direction.y,dz=ray.direction.z;
    const long double a=dx*dx+dy*dy+dz*dz;
    const long double b=x*dx+(r+y)*dy+z*dz;
    const long double c=x*x+z*z+y*(2*r+y);
    const long double discriminant=b*b-a*c;
    if(discriminant<0)return {};
    const long double root=std::sqrt(discriminant);
    const long double q=-b-std::copysign(root,b);
    long double first=q==0 ? -b/a : q/a;
    long double last=q==0 ? first : c/q;
    if(first>last)std::swap(first,last);
    return {true,first,last,b};
}
long double reference_height(const m3::TrackingParameters& p,g::Vec3 point) {
    const long double x=point.x,y=static_cast<long double>(point.y)-p.groundY+p.planetRadius,z=point.z;
    return std::sqrt(x*x+y*y+z*z)-p.planetRadius;
}

struct NamedRay { std::string name; RayInput input; };
std::vector<NamedRay> ray_cases(const m3::TrackingParameters& p) {
    std::vector<NamedRay> result;
    // Exact binary32-round-trip values supplied in the runtime failure report.
    const g::RaySegment reported{{3374.39648f,14831.293f,-8108.75244f},
        {-0.702025175f,-0.673400521f,-0.231716275f},INFINITY};
    result.push_back({"reported frame0 pixel113,60 bounce2",{reported,true}});
    for(int changed=0;changed<6;++changed)for(int side=-1;side<=1;side+=2) {
        auto ray=reported;
        float& component=changed==0 ? ray.origin.x : changed==1 ? ray.origin.y :
            changed==2 ? ray.origin.z : changed==3 ? ray.direction.x :
            changed==4 ? ray.direction.y : ray.direction.z;
        component=std::nextafter(component,side<0 ? -INFINITY : INFINITY);
        result.push_back({"reported adjacent component="+std::to_string(changed)+" side="+
            std::to_string(side),{ray,false}});
    }
    result.push_back({"radial",{{{0,p.groundY+1000,0},{0,-1,0},INFINITY},true}});
    result.push_back({"radial nonunit direction",{{{0,p.groundY+1000,0},{0,-2,0},INFINITY},true}});
    result.push_back({"outward surface",{{{0,p.groundY,0},{0,1,0},INFINITY},false}});
    result.push_back({"exact future tangent",{{{-100000,p.groundY,0},{1,0,0},200000},false}});
    const long double h=1,r=p.planetRadius;
    const float critical=static_cast<float>(-std::sqrt(h*(2*r+h))/r);
    for(int side=-1;side<=1;side+=2) {
        float dy=critical;
        for(int step=0;step<2;++step)dy=std::nextafter(dy,side<0 ? -INFINITY : INFINITY);
        result.push_back({"tangent neighbor side="+std::to_string(side),
            {{{0,p.groundY+1,0},{1,dy,0},INFINITY},false}});
    }
    result.push_back({"genuinely interior origin",{{{0,p.groundY-1,0},{0,1,0},INFINITY},false}});
    return result;
}

std::vector<EftInput> eft_cases(const m3::TrackingParameters& p,const g::RaySegment& reported) {
    std::vector<EftInput> result;
    const auto root=reference_roots(p,reported);
    const auto t=split(root.first);
    result.push_back({t,{reported.direction.x,0},reported.origin.x});
    result.push_back({t,{reported.direction.y,0},reported.origin.y});
    result.push_back({t,{reported.direction.z,0},reported.origin.z});
    for(float a:{6371000.0f,1000000.0f,22070.8789f,-22070.8789f}) {
        for(float b:{-0.673400521f,0.702025175f}) {
            const float cancellation=-static_cast<float>(static_cast<long double>(a)*b);
            result.push_back({{a,0},{b,0},cancellation});
        }
    }
    result.push_back({{1000000.0f,0.0001f},{0.6f,1e-8f},-600000.0f});
    return result;
}

void verify_rays(const m3::TrackingParameters& p,const std::vector<NamedRay>& cases,
    const std::vector<RayOutput>& results) {
    unsigned hits=0,misses=0,interior=0,forged=0;
    long double largest_root_error=0;
    for(std::size_t i=0;i<cases.size();++i) {
        const auto& entry=cases[i];const auto& out=results[i];
        check(out.interior_point_status==g::Status::InsideGround,entry.name+": interior point was repaired");
        if(reference_height(p,entry.input.ray.origin)<0) {
            check(out.clipped.status==g::Status::InsideGround,entry.name+": interior ray was not rejected");
            ++interior;continue;
        }
        check(out.clipped.status==g::Status::Ok,entry.name+": clip failed: "+g::status_name(out.clipped.status));
        const auto reference=reference_roots(p,entry.input.ray);
        const bool expected=reference.hit&&reference.first>=0&&reference.first<=entry.input.ray.t_max&&
            (reference.first>0||reference.half_b<0);
        check(out.clipped.hits_ground==expected,entry.name+": ground support differs from the reference");
        if(!expected){++misses;continue;}
        ++hits;
        near(exact(out.clipped.ground_parameter),reference.first,2e-5L,2e-10L,entry.name+": ground root");
        largest_root_error=std::max(largest_root_error,
            std::abs(exact(out.clipped.ground_parameter)-reference.first));
        check(out.planet.status==g::Status::Ok,entry.name+": planet_hit failed: "+g::status_name(out.planet.status));
        check(m1::finite(out.planet.position)&&m1::finite(out.planet.normal),entry.name+": nonfinite planet result");
        const auto n=out.planet.normal;
        const long double length=std::sqrt(static_cast<long double>(n.x)*n.x+
            static_cast<long double>(n.y)*n.y+static_cast<long double>(n.z)*n.z);
        near(length,1,2e-7L,0,entry.name+": unit normal");
        check(reference_height(p,out.planet.position)>=0,entry.name+": represented planet point is inside");
        check(out.outward.status==g::Status::Ok&&out.outward.has_medium&&!out.outward.hits_ground,
            entry.name+": outward continuation is self occluded");
        check(out.inward.status==g::Status::Ok&&out.inward.hits_ground&&out.inward.ground_occludes,
            entry.name+": inward continuation did not hit the planet");
        const auto inward_reference=reference_roots(p,{out.planet.position,-out.planet.normal,INFINITY});
        check(inward_reference.hit,entry.name+": reference inward continuation missed");
        near(exact(out.inward.ground_parameter),inward_reference.first,2e-5L,2e-10L,
             entry.name+": inward continuation root");
        if(entry.input.test_forged_endpoint) {
            check(out.forged.status==g::Status::UnresolvedGeometry,
                entry.name+": a forged interior endpoint was silently projected");
            ++forged;
        }
    }
    check(hits>0&&misses>0&&interior==1&&forged==3,"incomplete CUDA geometric coverage");
    std::cout<<"METRIC cuda_planet_cases="<<cases.size()<<" hits="<<hits<<" misses="<<misses
             <<" rejected_interior="<<interior<<" rejected_forged="<<forged
             <<" max_root_error_t="<<largest_root_error<<'\n';
}

void verify_efts(const std::vector<EftInput>& inputs,const std::vector<EftOutput>& results) {
    constexpr long double eps=std::numeric_limits<float>::epsilon();
    for(std::size_t i=0;i<inputs.size();++i) {
        const auto& in=inputs[i];const auto& out=results[i];
        const auto name="CUDA EFT case "+std::to_string(i);
        const long double plain_product=static_cast<long double>(in.a.hi)*in.b.hi;
        check(out.product.hi==static_cast<float>(plain_product),name+": rounded product high part");
        near(exact(out.product),plain_product,0,0,name+": TwoProduct exact residual");
        const long double pair_product=exact(in.a)*exact(in.b);
        const long double scale=std::max(1.0L,std::abs(pair_product)+std::abs(static_cast<long double>(in.c)));
        const long double bound=64*eps*eps*scale;
        near(exact(out.pair_product),pair_product,bound,0,name+": composed pair product");
        near(exact(out.sum),pair_product+in.c,bound,0,name+": product followed by TwoSum");
        near(exact(out.recovered),pair_product,bound,0,name+": cancellation recovers product");
    }
    std::cout<<"METRIC cuda_eft_composition_cases="<<inputs.size()<<'\n';
}
} // namespace

int main() {
    int devices=0;
    const cudaError_t availability=cudaGetDeviceCount(&devices);
    if(availability==cudaErrorNoDevice||availability==cudaErrorInsufficientDriver||
       (availability==cudaSuccess&&devices==0)) {
        std::cout<<"SKIP: CUDA regression was not executed; no usable CUDA device/driver ("
                 <<cudaGetErrorString(availability)<<", device_count="<<devices<<").\n";
        return 77;
    }
    try {
        cuda_check(availability,"cudaGetDeviceCount");
        cuda_check(cudaSetDevice(0),"cudaSetDevice");
        cudaDeviceProp device{};cuda_check(cudaGetDeviceProperties(&device,0),"cudaGetDeviceProperties");
        std::cout<<std::setprecision(15)<<"CUDA device: "<<device.name<<", compute "<<device.major<<'.'<<device.minor<<'\n';
        const m3::TrackingParameters parameters;
        const auto named=ray_cases(parameters);
        std::vector<RayInput> rays;for(const auto& value:named)rays.push_back(value.input);
        const auto efts=eft_cases(parameters,rays.front().ray);
        std::vector<RayOutput> ray_results(rays.size());
        std::vector<EftOutput> eft_results(efts.size());
        DeviceArray<m3::TrackingParameters> device_parameters(1);
        DeviceArray<RayInput> device_rays(rays.size());DeviceArray<RayOutput> device_ray_results(rays.size());
        DeviceArray<EftInput> device_efts(efts.size());DeviceArray<EftOutput> device_eft_results(efts.size());
        cuda_check(cudaMemcpy(device_parameters.pointer,&parameters,sizeof(parameters),cudaMemcpyHostToDevice),"copy parameters");
        cuda_check(cudaMemcpy(device_rays.pointer,rays.data(),rays.size()*sizeof(RayInput),cudaMemcpyHostToDevice),"copy rays");
        cuda_check(cudaMemcpy(device_efts.pointer,efts.data(),efts.size()*sizeof(EftInput),cudaMemcpyHostToDevice),"copy EFT inputs");
        geometry_kernel<<<1,64>>>(device_parameters.pointer,device_rays.pointer,device_ray_results.pointer,
            static_cast<unsigned>(rays.size()));
        cuda_check(cudaGetLastError(),"launch geometry kernel");
        eft_kernel<<<1,64>>>(device_efts.pointer,device_eft_results.pointer,static_cast<unsigned>(efts.size()));
        cuda_check(cudaGetLastError(),"launch EFT kernel");
        cuda_check(cudaDeviceSynchronize(),"execute regression kernels");
        cuda_check(cudaMemcpy(ray_results.data(),device_ray_results.pointer,rays.size()*sizeof(RayOutput),cudaMemcpyDeviceToHost),"copy ray results");
        cuda_check(cudaMemcpy(eft_results.data(),device_eft_results.pointer,efts.size()*sizeof(EftOutput),cudaMemcpyDeviceToHost),"copy EFT results");
        verify_efts(efts,eft_results);verify_rays(parameters,named,ray_results);
        std::cout<<"PASS: real CUDA FP32 geometry and EFT regression executed successfully.\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"FAIL: CUDA tracking geometry regression: "<<error.what()<<'\n';
        return 1;
    }
}
