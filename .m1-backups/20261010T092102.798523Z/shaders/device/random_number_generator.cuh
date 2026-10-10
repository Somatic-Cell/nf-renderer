#ifndef RANDOM_NUMBER_GENERATOR_CUH_
#define RANDOM_NUMBER_GENERATOR_CUH_

template<unsigned int N=16>
class LCG {
public:
    inline __host__ __device__ LCG()
    {

    }

    inline __host__ __device__ LCG(unsigned int val0, unsigned int val1)
    {
        init(val0, val1);
    }

    inline __host__ __device__ float operator() ()
    {
        const uint32_t LCG_A = 1664525u;
        const uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return (state >> 8) / (float) 0x01000000;
    }

    inline __host__ __device__ void init(const unsigned int val0, const unsigned int val1)
    {
        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;

        for(unsigned int n = 0; n < N; n++){
            s0 += 0x9e3779b9;
            v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
            v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
        }
        state = v0;
    }

protected:

    uint32_t state;
};

__device__ __forceinline__ uint32_t hash(uint32_t x) {
    x ^= x >> 17; x *= 0xed5ad4bb;
    x ^= x >> 11; x *= 0xac4c1b51;
    x ^= x >> 15; x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

static __host__ __device__ __forceinline__ uint32_t rot32(uint32_t x, uint32_t r)
{
    return (x >> r) | (x << ((-r) & 31));
}

// SplitMix64: seed 展開用（良い初期化ができます）
static __host__ __device__ __forceinline__ uint64_t splitmix64_next(uint64_t& x)
{
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static __host__ __device__ __forceinline__ float u32_to_unit_float24(uint32_t x)
{
    // 上位24bitで [0,1)（ちょうど 2^-24）
    return (x >> 8) * 0x1.0p-24f;
}

class PCG32
{
public:
    __host__ __device__ PCG32() : state(0), inc(1) {}

    __host__ __device__ PCG32(uint32_t val0, uint32_t val1) { init(val0, val1); }

    __host__ __device__ __forceinline__ void init(uint32_t val0, uint32_t val1)
    {
        // 2つの32bit seed から 64bit を構成し、SplitMix64 で state/stream を作る
        uint64_t s = (uint64_t(val0) << 32) | uint64_t(val1);
        uint64_t sm = s;

        uint64_t initstate = splitmix64_next(sm);
        uint64_t initseq   = splitmix64_next(sm);

        seed(initstate, initseq);
    }

    __host__ __device__ __forceinline__ float operator()()
    {
        return u32_to_unit_float24(next_u32());
    }

    __host__ __device__ __forceinline__ uint32_t next_u32()
    {
        uint64_t oldstate = state;
        // PCG-XSH-RR 32: state update
        state = oldstate * 6364136223846793005ULL + inc;

        uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = uint32_t(oldstate >> 59u);
        return rot32(xorshifted, rot);
    }

private:
    __host__ __device__ __forceinline__ void seed(uint64_t initstate, uint64_t initseq)
    {
        state = 0u;
        inc   = (initseq << 1u) | 1u; // 必ず奇数（stream選択）
        next_u32();
        state += initstate;
        next_u32();
    }

    uint64_t state;
    uint64_t inc;
};

#endif // RANDOM_NUMBER_GENERATOR_CUH_