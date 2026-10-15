#pragma once
#include "types.hpp"
namespace m4nf {
struct CoopView {
    std::uint64_t base;
    std::uint32_t matrix_offsets[16];
    std::uint32_t bias_offsets[16];
};
struct Query { std::uint32_t op=0;float lambda=0,cosine=0,a=0,b=0,c=0; };
struct ProbeOutput { std::uint32_t status=0;float log_pdf=0,x=0,y=0,z=0,reevaluated_log_pdf=0; };
struct ProbeParams {
    ModelView model;
    CoopView coop;
    const Query* queries;
    ProbeOutput* output;
    std::uint32_t count,backend;
};
}
