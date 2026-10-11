#include "m3/tracking_geometry.hpp"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdint>

namespace g=m3::tracking;
volatile float runtime_values[]={3374.39648f,14831.293f,-8108.75244f,
                                -0.702025175f,-0.673400521f,-0.231716275f};
long double exact(g::FloatPair p){return (long double)p.hi+p.lo;}
void pair(const char* name,g::FloatPair p){std::cout<<name<<" = "<<p.hi<<" + "<<p.lo<<" = "<<exact(p)<<"\n";}
void vec(const char* name,g::Vec3 v){std::cout<<name<<" = ("<<v.x<<", "<<v.y<<", "<<v.z<<")\n";}
long double independent_height(const m3::TrackingParameters& p,g::PairVec3 v){
    const long double x=exact(v.x),z=exact(v.z),y=exact(v.y)-p.groundY,r=p.planetRadius;
    const long double c=x*x+z*z+y*(2*r+y);
    return c/(std::sqrt(r*r+c)+r);
}
void run(float limit){
    std::cout<<"\nlimit = "<<limit<<"\n";
    const m3::TrackingParameters p{};
    const g::RaySegment ray{{runtime_values[0],runtime_values[1],runtime_values[2]},
                           {runtime_values[3],runtime_values[4],runtime_values[5]},limit};
    vec("origin",ray.origin);vec("direction",ray.direction);
    g::geometry_detail::PreparedRay r;
    auto status=g::geometry_detail::prepare(p,ray,r);
    std::cout<<"prepare.status = "<<g::status_name(status)<<"\n";
    pair("a",r.a);pair("b",r.b);pair("c",r.ground_c);
    const auto py=g::sub(g::FloatPair{ray.origin.y,0},p.groundY);
    const auto ys=g::add(g::FloatPair{2.0f*p.planetRadius,0},py);
    pair("c.vertical_y",py);pair("c.vertical_sum",ys);
    pair("c.vertical_product_hi",g::product(py.hi,ys.hi));
    pair("c.vertical",g::mul(py,ys));
    pair("c.xsquare",g::product(ray.origin.x,ray.origin.x));
    pair("c.zsquare",g::product(ray.origin.z,ray.origin.z));
    const long double dx=ray.direction.x,dy=ray.direction.y,dz=ray.direction.z;
    const long double x=ray.origin.x,y=(long double)ray.origin.y-p.groundY,z=ray.origin.z,R=p.planetRadius;
    const long double a=dx*dx+dy*dy+dz*dz,b=x*dx+(R+y)*dy+z*dz,c=x*x+z*z+y*(2*R+y);
    const long double disc=b*b-a*c,q=-b-std::copysign(std::sqrt(disc),b);
    const long double ref_t=std::min(c/q,q/a);
    const auto reported_point=g::point_at(ray,22070.87890625f);
    std::cout<<"reported_t.independent_height = "<<independent_height(p,reported_point)<<"\n";
    std::cout<<"reference.a = "<<a<<"\nreference.b = "<<b<<"\nreference.c = "<<c
             <<"\nreference.discriminant = "<<disc<<"\nreference.near = "<<ref_t<<"\n";
    const auto roots=g::geometry_detail::roots(r,r.ground_c);
    std::cout<<"roots.status = "<<g::status_name(roots.status)<<", hit = "<<roots.hit<<"\n";
    pair("roots.near",roots.near_t);pair("roots.far",roots.far_t);
    pair("discriminant",g::sub(g::mul(r.b,r.b),g::mul(r.a,r.ground_c)));
    const auto clip=g::clip(p,ray);
    std::cout<<"clip.status = "<<g::status_name(clip.status)<<", hits_ground = "<<clip.hits_ground<<"\n";
    pair("clip.ground_parameter",clip.ground_parameter);
    const auto source=g::point_at(ray,clip.ground_parameter);
    pair("source.x",source.x);pair("source.y",source.y);pair("source.z",source.z);
    const auto measured=g::geometry_detail::height_unchecked(p,source);
    std::cout<<"measured.status = "<<g::status_name(measured.status)<<", height = "<<measured.height_m
             <<", independent.height = "<<independent_height(p,source)<<"\n";
    pair("measured.squared_ground_distance",measured.squared_ground_distance);
    const float scale=fmaxf(fmaxf(fabsf(ray.origin.x),fabsf(ray.origin.y)),fabsf(ray.origin.z));
    const float travel=fabsf(g::value(clip.ground_parameter))*clip.direction_length;
    const float bound=(128.0f*FLT_EPSILON*FLT_EPSILON)*(p.planetRadius+scale+travel);
    std::cout<<"bound = "<<bound<<", height_over_bound = "<<fabsf(measured.height_m)/bound<<"\n";
    const auto hit=g::planet_hit(p,ray,clip);
    std::cout<<"planet_hit.status = "<<g::status_name(hit.status)<<", radial_correction = "<<hit.radial_correction_m<<"\n";
    vec("hit.position",hit.position);vec("hit.normal",hit.normal);
}
int main(){
    std::cout<<std::setprecision(24);
    run(std::numeric_limits<float>::infinity());
    run(22070.8789f);
    run(1e30f);
}
