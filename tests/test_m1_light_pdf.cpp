#include "m1/light_pdf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance, const char* message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

struct D3 { double x, y, z; };
D3 d3(m1::Vec3 v) { return {v.x, v.y, v.z}; }
D3 operator-(D3 a, D3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
double dot(D3 a, D3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
D3 cross(D3 a, D3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }

// Independent double-precision Moller-Trumbore intersection. The regression
// must hit the bounded emitter triangle, not merely its infinite plane.
bool triangle_hit(m1::Vec3 origin, m1::Vec3 direction,
                  const std::array<m1::Vec3, 3>& triangle, m1::Vec3& point) {
    const D3 e1=d3(triangle[1])-d3(triangle[0]);
    const D3 e2=d3(triangle[2])-d3(triangle[0]);
    const D3 p=cross(d3(direction),e2);
    const double determinant=dot(e1,p);
    if (determinant==0.0) return false;
    const D3 s=d3(origin)-d3(triangle[0]);
    const double u=dot(s,p)/determinant;
    const D3 q=cross(s,e1);
    const double v=dot(d3(direction),q)/determinant;
    const double t=dot(e2,q)/determinant;
    if (!(t>0.0) || u<0.0 || v<0.0 || u+v>1.0) return false;
    point={float((1-u-v)*triangle[0].x+u*triangle[1].x+v*triangle[2].x),
           float((1-u-v)*triangle[0].y+u*triangle[1].y+v*triangle[2].y),
           float((1-u-v)*triangle[0].z+u*triangle[1].z+v*triangle[2].z)};
    return true;
}

// Reproduce the existing shader's finite epsilon + one-ULP displacement.
m1::Vec3 existing_offset(m1::Vec3 p, m1::Vec3 normal,
                         m1::Vec3 direction, float epsilon) {
    const float side=m1::dot(normal,direction)>=0.0f?1.0f:-1.0f;
    const auto n=normal*side;
    auto result=p+n*epsilon;
    const float infinity=std::numeric_limits<float>::infinity();
    if (n.x!=0) result.x=std::nextafter(result.x,n.x>0?infinity:-infinity);
    if (n.y!=0) result.y=std::nextafter(result.y,n.y>0?infinity:-infinity);
    if (n.z!=0) result.z=std::nextafter(result.z,n.z>0?infinity:-infinity);
    return result;
}

void test_offset_support() {
    using namespace m1;
    constexpr float epsilon=1.0e-4f;
    const Vec3 previous{0.0f,-0.25f*epsilon,0.0f};
    const Vec3 receiver_normal=normalize(Vec3{1,1,0});
    const Vec3 direction=normalize(Vec3{1,-0.01f,0});
    require(m1::dot(receiver_normal,direction)>0,
            "counterexample must reflect into the source geometric hemisphere");
    const Vec3 origin=existing_offset(previous,receiver_normal,direction,epsilon);
    const std::array<Vec3,3> triangle{{{-1,0,-1},{0,0,1},{1,0,-1}}};
    Vec3 hit{};
    require(triangle_hit(origin,direction,triangle,hit), "displaced ray hits the emitter triangle");
    const Vec3 edges=m1::cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);
    const float area=0.5f*length(edges);
    const Vec3 normal=normalize(edges);
    const float traced_cosine=m1::dot(normal,-direction);
    const Vec3 connection=hit-previous;
    const float connection_cosine=m1::dot(normal,-normalize(connection));
    require(traced_cosine>0.0f && connection_cosine<0.0f,
            "finite offset reproduces front-hit/competing-support disagreement");
    const auto result=one_sided_area_light_pdf(m1::dot(connection,connection),area,connection_cosine,1);
    require(result.valid && result.pdf==0.0f, "outside NEE support is a valid zero density");
    close(power_heuristic(0.2f,result.pdf),1.0,0.0,"sole supported BSDF strategy has MIS weight one");
    std::cout<<"offset triangle hit: traced cosine="<<traced_cosine
             <<" connection cosine="<<connection_cosine<<" competing PDF="<<result.pdf<<'\n';
}

void test_analytic_pdf_and_range() {
    using namespace m1;
    const auto normal=one_sided_area_light_pdf(4.0f,2.0f,1.0f,4);
    require(normal.valid,"normal-incidence PDF is valid");
    close(normal.pdf,0.5,0.0,"uniform triangle PMF and area Jacobian");
    const auto oblique=one_sided_area_light_pdf(25.0f,2.0f,0.6f,5);
    require(oblique.valid,"oblique PDF is valid");
    close(oblique.pdf,25.0/6.0,3e-7,"oblique area-to-solid-angle Jacobian");
    for(float cosine:std::array<float,3>{0.0f,-0.0f,-0.25f}) {
        const auto result=one_sided_area_light_pdf(4,2,cosine,4);
        require(result.valid && result.pdf==0,"finite nonpositive cosine has no one-sided support");
    }
    const auto small_product=one_sided_area_light_pdf(1e-30f,1e-30f,1e-20f,1);
    require(small_product.valid,"small denominator product must not underflow before division");
    close(small_product.pdf/1e20,1.0,2e-7,"small-product quotient remains representable");
    const auto large_product=one_sided_area_light_pdf(1e30f,1e30f,1.0f,2000000000);
    require(large_product.valid,"large denominator product must not overflow before division");
    close(large_product.pdf,5e-10,1e-16,"large-product quotient retains positive density");
}

void test_invalid_geometry_and_modes() {
    using namespace m1;
    const float nan=std::numeric_limits<float>::quiet_NaN();
    const float infinity=std::numeric_limits<float>::infinity();
    for(float bad:std::array<float,5>{0,-1,nan,infinity,-infinity}) {
        require(!one_sided_area_light_pdf(bad,1,1,1).valid,"invalid distance squared is rejected");
        require(!one_sided_area_light_pdf(1,bad,1,1).valid,"invalid area is rejected");
        require(!one_sided_area_light_pdf(bad,1,-1,1).valid,"zero support must not hide invalid distance");
        require(!one_sided_area_light_pdf(1,bad,-1,1).valid,"zero support must not hide invalid area");
    }
    for(float bad:std::array<float,3>{nan,infinity,-infinity})
        require(!one_sided_area_light_pdf(1,1,bad,1).valid,"nonfinite cosine is rejected");
    for(int bad:std::array<int,2>{0,-1})
        require(!one_sided_area_light_pdf(1,1,-1,bad).valid,"invalid light count is rejected before support check");
    require(!one_sided_area_light_pdf(1,1e-30f,1e-30f,1).valid,"unrepresentable positive PDF is rejected");
    require(!one_sided_area_light_pdf(std::numeric_limits<float>::denorm_min(),
            (std::numeric_limits<float>::max)(),1,1).valid,"positive PDF underflow is not reclassified as zero support");
    for(int bounce:std::array<int,3>{0,1,8})
      for(bool delta:std::array<bool,2>{false,true})
       for(unsigned mode:std::array<unsigned,3>{0,1,2})
        require(light_hit_needs_pdf(bounce,delta,mode)==(bounce>0&&!delta&&mode==0),
                "only a continuous MIS light hit needs the competing PDF");
}

void test_fp32_exponent_range_against_reference() {
    // Inputs cover every float exponent regime, not just ordinary scene units.
    // The independent host quotient keeps all four input factors in double.
    const std::array<int,12> exponents{{-148,-127,-126,-100,-40,-1,0,1,40,100,126,128}};
    const std::array<int,8> cosine_exponents{{-148,-127,-100,-40,-10,-1,0,1}};
    const std::array<int,6> counts{{1,3,257,16777217,2000000000,2147483647}};
    constexpr double epsilon=std::numeric_limits<float>::epsilon();
    const double smallest=std::numeric_limits<float>::denorm_min();
    const double largest=(std::numeric_limits<float>::max)();
    double maximum_normal_relative_error=0,maximum_subnormal_absolute_error=0;
    unsigned checked=0,rejected_range=0;
    for(int de:exponents) for(int ae:exponents) for(int ce:cosine_exponents) for(int count:counts) {
        const float distance=std::ldexp(0.8125f,de);
        const float area=std::ldexp(0.6875f,ae);
        const float cosine=std::ldexp(0.5625f,ce);
        if(!(distance>0.0f && area>0.0f && cosine>0.0f && cosine<=1.0f))continue;
        const double reference=double(distance)/(double(area)*double(cosine)*double(count));
        const auto result=m1::one_sided_area_light_pdf(distance,area,cosine,count);
        if(reference>largest*(1.0+4.0*epsilon) || reference<smallest*(0.5-4.0*epsilon)) {
            require(!result.valid,"unrepresentable final density must remain an error");
            ++rejected_range;
            continue;
        }
        if(reference>largest || reference<=smallest*(0.5+4.0*epsilon))continue;
        require(result.valid && result.pdf>0.0f,"representable quotient was lost to intermediate range");
        const double error=std::abs(double(result.pdf)-reference);
        require(error<=4.0*epsilon*reference+0.5*smallest,
                "FP32 Jacobian error exceeds operation rounding plus subnormal rounding");
        if(reference>=std::numeric_limits<float>::min())
            maximum_normal_relative_error=std::max(maximum_normal_relative_error,error/reference);
        else maximum_subnormal_absolute_error=std::max(maximum_subnormal_absolute_error,error);
        ++checked;
    }
    const auto maximum=m1::one_sided_area_light_pdf((std::numeric_limits<float>::max)(),1,1,1);
    require(maximum.valid && maximum.pdf==(std::numeric_limits<float>::max)(),
            "largest finite PDF remains representable");
    const auto minimum=m1::one_sided_area_light_pdf(std::numeric_limits<float>::denorm_min(),1,1,1);
    require(minimum.valid && minimum.pdf==std::numeric_limits<float>::denorm_min(),
            "smallest positive PDF is not replaced by zero");
    std::cout<<"FP32 PDF exponent sweep: finite="<<checked<<" range_rejections="<<rejected_range
             <<" max_normal_relative_error="<<maximum_normal_relative_error
             <<" max_subnormal_absolute_error="<<maximum_subnormal_absolute_error<<'\n';
}
} // namespace

int main() {
    try {
        std::cout<<std::setprecision(10);
        test_offset_support();
        test_analytic_pdf_and_range();
        test_invalid_geometry_and_modes();
        test_fp32_exponent_range_against_reference();
        std::cout<<"M1 area-light PDF: finite offset support, analytic Jacobian, "
                     "intermediate range, invalid geometry and mode checks passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"M1 area-light PDF check failed: "<<error.what()<<'\n';
        return 1;
    }
}
