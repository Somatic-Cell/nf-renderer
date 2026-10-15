#pragma once
// Host-side asset identification only. SHA-256, no GPU arithmetic/dependency.
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
namespace m4rain {
inline std::string file_sha256(const std::string& path){
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot hash NF asset: "+path);
    constexpr std::uint32_t k[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto rotr=[](std::uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));};
    auto block=[&](const unsigned char* b){
        std::uint32_t w[64];
        for(unsigned i=0;i<16;++i)w[i]=(std::uint32_t(b[4*i])<<24)|(std::uint32_t(b[4*i+1])<<16)|(std::uint32_t(b[4*i+2])<<8)|b[4*i+3];
        for(unsigned i=16;i<64;++i){auto x=w[i-15],y=w[i-2];w[i]=w[i-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-7]+(rotr(y,17)^rotr(y,19)^(y>>10));}
        auto a=h[0],b1=h[1],c=h[2],d=h[3],e=h[4],f1=h[5],g=h[6],h1=h[7];
        for(unsigned i=0;i<64;++i){const auto t=h1+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f1)^(~e&g))+k[i]+w[i];const auto u=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b1)^(a&c)^(b1&c));h1=g;g=f1;f1=e;e=d+t;d=c;c=b1;b1=a;a=t+u;}
        h[0]+=a;h[1]+=b1;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f1;h[6]+=g;h[7]+=h1;
    };
    std::array<unsigned char,128> b{};std::uint64_t bytes=0;std::size_t n=0;
    for(;;){f.read(reinterpret_cast<char*>(b.data()),64);n=static_cast<std::size_t>(f.gcount());bytes+=n;if(n<64)break;block(b.data());}
    if(!f.eof())throw std::runtime_error("NF asset hash read failed: "+path);
    b[n++]=0x80;const std::size_t size=n<=56?64:128;for(std::size_t j=n;j<size;++j)b[j]=0;
    const std::uint64_t bits=bytes*8;for(unsigned j=0;j<8;++j)b[size-1-j]=static_cast<unsigned char>(bits>>(j*8));
    block(b.data());if(size==128)block(b.data()+64);
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(const auto x:h)out<<std::setw(8)<<x;return out.str();
}
}
