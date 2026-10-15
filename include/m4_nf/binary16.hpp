#pragma once
#include <cstdint>
#include <cstring>
namespace m4nf {
// Host-only IEEE 754 binary16 round-to-nearest-even conversion. Includes
// subnormals. Used for packing and for a CPU sensitivity test, not GPU emulation.
inline std::uint16_t to_half_bits(float x){
    std::uint32_t b;std::memcpy(&b,&x,4);
    const std::uint16_t sign=static_cast<std::uint16_t>((b>>16)&0x8000u);
    const unsigned e=(b>>23)&255u,m=b&0x7fffffu;
    if(e==255)return static_cast<std::uint16_t>(sign|0x7c00u|(m?0x0200u:0));
    const int he=static_cast<int>(e)-127+15;
    if(he>=31)return static_cast<std::uint16_t>(sign|0x7c00u);
    if(he<=0){
        if(he < -10)return sign;
        const unsigned full=m|0x800000u,shift=static_cast<unsigned>(14-he);
        const unsigned a=full>>shift,rem=full&((1u<<shift)-1u),half=1u<<(shift-1u);
        return static_cast<std::uint16_t>(sign|static_cast<std::uint16_t>(a+(rem>half||(rem==half&&(a&1u)))));
    }
    unsigned a=(static_cast<unsigned>(he)<<10)|(m>>13),rem=m&8191u;
    a+=rem>4096u||(rem==4096u&&(a&1u));
    return static_cast<std::uint16_t>(sign|a);
}
inline float from_half_bits(std::uint16_t h){
    std::uint32_t s=std::uint32_t(h&0x8000u)<<16,m=h&1023u,e=(h>>10)&31u,b;
    if(e==0){
        if(m==0)b=s;
        else {int k=-14;while((m&1024u)==0){m<<=1;--k;}m&=1023u;b=s|(std::uint32_t(k+127)<<23)|(m<<13);}
    }else if(e==31)b=s|0x7f800000u|(m<<13);
    else b=s|((e+112u)<<23)|(m<<13);
    float f;std::memcpy(&f,&b,4);return f;
}
inline float round_half(float f){return from_half_bits(to_half_bits(f));}
} // namespace m4nf
