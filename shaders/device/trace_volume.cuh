#ifndef TRACE_VOLUME_CUH_
#define TRACE_VOLUME_CUH_

#include <optix.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/NanoVDB.h>
// #include <nanovdb/math/HDDA.h> 

#include "../config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "../../utils/helper_math.h"
#include "../../utils/my_math.hpp"
#include "../params/per_ray_data.cuh"
#include "shader_common.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

#include <optix_device.h>  
#ifdef __CUDACC__
#include <cuda_fp16.h>      // half を使うなら（OptixCoopVec<half, N> 等）
#endif

// static __forceinline__ __device__
// int floorDivInt(int a, int b)
// {
//     int q = a / b;
//     int r = a % b;
//     if (r != 0 && ((r > 0) != (b > 0))) --q; // C/C++ の /,% は 0 方向丸めなので補正
//     return q;
// }

template <typename AccT>
static __forceinline__ __device__
auto makeSampler(const AccT& acc)
-> decltype(nanovdb::math::createSampler<1>(acc))
{
    return nanovdb::math::createSampler<1>(acc);
}

static __forceinline__ __device__
int floorToInt(float x)
{
    return __float2int_rd(x);
}


struct MacroDDAState
{
    int3 cell = make_int3(0, 0, 0);
    int3 step = make_int3(0, 0, 0);

    float3 tMax   = make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F);
    float3 tDelta = make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F);

    float t      = 0.0f;   // 現在セグメントの開始
    float tEnd   = 0.0f;   // 現在セグメントの終端
    float tExit  = 0.0f;   // macro grid を出る時刻
    float majorant = 0.0f;

    uint32_t valid = 0;
};

struct MacrocellView
{
    const float* majorants = nullptr;   // density の最大値を格納
    int3  baseCoord = make_int3(0, 0, 0);
    int3  dims      = make_int3(0, 0, 0);
    int   cellSizeVoxels = 0;
    float densityScale   = 1.0f;
    float sigmaTScale    = 1.0f;
    float fallbackDensityMax = 0.0f;   // macro grid 不在時の fallback
};

static __forceinline__ __device__
MacrocellView makeMacrocellView(const VDBGeomData& vdb, float sigmaTScale)
{
    MacrocellView out;
    out.majorants = reinterpret_cast<const float*>(vdb.macroMajorant);
    out.baseCoord = make_int3(vdb.macroBaseCoord.x, vdb.macroBaseCoord.y, vdb.macroBaseCoord.z);
    out.dims      = make_int3(vdb.macroDims.x,      vdb.macroDims.y,      vdb.macroDims.z);
    out.cellSizeVoxels    = vdb.macroCellSizeVoxels;
    out.densityScale      = vdb.densityScale;
    out.sigmaTScale       = sigmaTScale;
    out.fallbackDensityMax = vdb.macroMaxDensity;
    return out;
}

static __forceinline__ __device__
bool macrocellGridValid(const MacrocellView& macro)
{
    return macro.majorants != nullptr &&
           macro.cellSizeVoxels > 0 &&
           macro.dims.x > 0 &&
           macro.dims.y > 0 &&
           macro.dims.z > 0;
}

static __forceinline__ __device__
size_t flattenMacrocellCoord(const int3& c, const int3& dims)
{
    return (static_cast<size_t>(c.z) * static_cast<size_t>(dims.y)
          + static_cast<size_t>(c.y)) * static_cast<size_t>(dims.x)
          + static_cast<size_t>(c.x);
}

static __forceinline__ __device__
bool isInsideMacrocellGrid(const int3& c, const int3& dims)
{
    return (c.x >= 0 && c.y >= 0 && c.z >= 0 &&
            c.x < dims.x && c.y < dims.y && c.z < dims.z);
}

static __forceinline__ __device__
int3 pointIndexToMacrocellCoord(const MacrocellView& macro, const nanovdb::Vec3f& pI)
{
    const float invCell = 1.0f / float(macro.cellSizeVoxels);
    return make_int3(
        floorToInt((pI[0] - float(macro.baseCoord.x)) * invCell),
        floorToInt((pI[1] - float(macro.baseCoord.y)) * invCell),
        floorToInt((pI[2] - float(macro.baseCoord.z)) * invCell)
    );
}

static __forceinline__ __device__
float lookupMacrocellMajorantDensity(const MacrocellView& macro, const int3& cell)
{
    if (!macrocellGridValid(macro)) {
        return fmaxf(macro.fallbackDensityMax, 0.0f);
    }
    if (!isInsideMacrocellGrid(cell, macro.dims)) {
        return 0.0f;
    }
    const size_t idx = flattenMacrocellCoord(cell, macro.dims);
    return fmaxf(macro.majorants[idx], 0.0f);
}

static __forceinline__ __device__
float macrocellBoundaryT(
    const MacrocellView& macro,
    const nanovdb::Vec3f& rayOriginIndex,
    const nanovdb::Vec3f& rayDirectionIndex,
    const int3& cell,
    int axis)
{
    const float d = rayDirectionIndex[axis];
    if (fabsf(d) < 1e-20f) {
        return CUDART_INF_F;
    }

    const int base = (&macro.baseCoord.x)[axis];
    const int c    = (&cell.x)[axis];
    const int cs   = macro.cellSizeVoxels;

    const float boundaryIndex =
        (d > 0.0f)
        ? float(base + (c + 1) * cs)
        : float(base + c * cs);

    return (boundaryIndex - rayOriginIndex[axis]) / d;
}

// static __forceinline__ __device__
// LocalSegment getMacrocellSegment(
//     const MacrocellView& macro,
//     const nanovdb::Vec3f& rayOriginIndex,
//     const nanovdb::Vec3f& rayDirectionIndex,
//     float t,
//     float tExit)
// {
//     LocalSegment seg;
//     seg.tEnd = tExit;
//     seg.majorant = 0.0f;

//     if (!(tExit > t)) {
//         return seg;
//     }

//     // 現在セルの曖昧さを避けるため，ごく小さく前進してからセルを決める
//     const float dirLen = fmaxf(length(make_float3(rayDirectionIndex[0], rayDirectionIndex[1], rayDirectionIndex[2])), 1e-20f);
//     const float dt = 1e-3f / dirLen;
//     const float tt = fminf(tExit, nextafterf(t, tExit) + dt);

//     const nanovdb::Vec3f pI = rayOriginIndex + tt * rayDirectionIndex;
//     const int3 cell = pointIndexToMacrocellCoord(macro, pI);

//     // macro grid の外は 0 扱いにしてそのまま tExit まで飛ばす
//     if (!isInsideMacrocellGrid(cell, macro.dims)) {
//         seg.tEnd = tExit;
//         seg.majorant = 0.0f;
//         return seg;
//     }

//     const float densityUpper = lookupMacrocellMajorantDensity(macro, cell);
//     seg.majorant = densityUpper * macro.densityScale * macro.sigmaTScale;

//     float tEnd = tExit;
//     tEnd = fminf(tEnd, macrocellBoundaryT(macro, rayOriginIndex, rayDirectionIndex, cell, 0));
//     tEnd = fminf(tEnd, macrocellBoundaryT(macro, rayOriginIndex, rayDirectionIndex, cell, 1));
//     tEnd = fminf(tEnd, macrocellBoundaryT(macro, rayOriginIndex, rayDirectionIndex, cell, 2));

//     seg.tEnd = fminf(tExit, fmaxf(tEnd, nextafterf(t, tExit)));
//     return seg;
// }

static __forceinline__ __device__
nanovdb::Coord floorToCoord(const nanovdb::Vec3f& p)
{
    return nanovdb::Coord(floorToInt(p[0]), floorToInt(p[1]), floorToInt(p[2]));
}

// world 空間から vdb の index への変換
static __forceinline__ __device__ 
nanovdb::Vec3f worldToIndexPoint(
    const nanovdb::FloatGrid* grid,
    const nanovdb::Vec3f& pW
)
{
    // Vec3f return values alone do not select FP32: the unsuffixed NanoVDB
    // map API evaluates with its double coefficient copy. Select its float
    // coefficients explicitly for the GPU's float coordinate path.
    return grid->map().applyInverseMapF(pW);
}

static __forceinline__ __device__ 
nanovdb::Vec3f worldToIndexVector(
    const nanovdb::FloatGrid* grid,
    const nanovdb::Vec3f& pW,
    const nanovdb::Vec3f& vW
)
{
    return grid->map().applyInverseJacobianF(vW);
}

static __forceinline__ __device__
bool intersectMacrocellAABB(
    const MacrocellView& macro,
    const nanovdb::Vec3f& rayOriginIndex,
    const nanovdb::Vec3f& rayDirectionIndex,
    float tEnter,
    float tExit,
    float& outEnter,
    float& outExit)
{
    if (!(tExit > tEnter)) return false;

    if (!macrocellGridValid(macro)) {
        outEnter = tEnter;
        outExit  = tExit;
        return true;
    }

    float t0 = tEnter;
    float t1 = tExit;

    #pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        const float mn = float((&macro.baseCoord.x)[ax]);
        const float mx = float((&macro.baseCoord.x)[ax] +
                               (&macro.dims.x)[ax] * macro.cellSizeVoxels);

        const float o = rayOriginIndex[ax];
        const float d = rayDirectionIndex[ax];

        if (fabsf(d) < 1e-20f) {
            if (o < mn || o > mx) return false;
            continue;
        }

        const float invD = 1.0f / d;
        float tNear = (mn - o) * invD;
        float tFar  = (mx - o) * invD;
        if (tNear > tFar) {
            const float tmp = tNear;
            tNear = tFar;
            tFar  = tmp;
        }

        t0 = fmaxf(t0, tNear);
        t1 = fminf(t1, tFar);
        if (!(t1 > t0)) return false;
    }

    outEnter = t0;
    outExit  = t1;
    return true;
}


static __forceinline__ __device__
void refreshMacroDDA(
    const MacrocellView& macro,
    MacroDDAState& s)
{
    const float densityUpper = lookupMacrocellMajorantDensity(macro, s.cell);
    s.majorant = densityUpper * macro.densityScale * macro.sigmaTScale;

    const float tNext = fminf(s.tMax.x, fminf(s.tMax.y, s.tMax.z));
    s.tEnd = fminf(s.tExit, tNext);
    s.tEnd = fmaxf(s.tEnd, nextafterf(s.t, s.tExit));
}

static __forceinline__ __device__
bool initMacroDDA(
    const MacrocellView& macro,
    const nanovdb::Vec3f& rayOriginIndex,
    const nanovdb::Vec3f& rayDirectionIndex,
    float tEnter,
    float tExit,
    MacroDDAState& s)
{
    s.valid = 0;

    if (!(tExit > tEnter)) return false;

    // fallback: macro grid なしなら単一区間
    if (!macrocellGridValid(macro)) {
        s.t = tEnter;
        s.tEnd = tExit;
        s.tExit = tExit;
        s.majorant = fmaxf(macro.fallbackDensityMax *
                           macro.densityScale *
                           macro.sigmaTScale, 0.0f);
        s.valid = 1;
        return true;
    }

    float t0, t1;
    if (!intersectMacrocellAABB(macro, rayOriginIndex, rayDirectionIndex,
                                tEnter, tExit, t0, t1)) {
        return false;
    }

    const float dirLen = fmaxf(length(make_float3(
        rayDirectionIndex[0], rayDirectionIndex[1], rayDirectionIndex[2])), 1e-20f);
    const float dt = 1e-3f / dirLen;
    const float tt = fminf(t1, nextafterf(t0, t1) + dt);

    const nanovdb::Vec3f pI = rayOriginIndex + tt * rayDirectionIndex;
    s.cell = pointIndexToMacrocellCoord(macro, pI);

    if (!isInsideMacrocellGrid(s.cell, macro.dims)) {
        return false;
    }

    s.t = t0;
    s.tExit = t1;

    #pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        const float d = rayDirectionIndex[ax];
        const int base = (&macro.baseCoord.x)[ax];
        const int c    = (&s.cell.x)[ax];
        const int cs   = macro.cellSizeVoxels;

        if (d > 1e-20f) {
            (&s.step.x)[ax]   = 1;
            (&s.tDelta.x)[ax] = float(cs) / d;
            const float boundary = float(base + (c + 1) * cs);
            (&s.tMax.x)[ax] = (boundary - rayOriginIndex[ax]) / d;
        }
        else if (d < -1e-20f) {
            (&s.step.x)[ax]   = -1;
            (&s.tDelta.x)[ax] = float(cs) / (-d);
            const float boundary = float(base + c * cs);
            (&s.tMax.x)[ax] = (boundary - rayOriginIndex[ax]) / d;
        }
        else {
            (&s.step.x)[ax]   = 0;
            (&s.tDelta.x)[ax] = CUDART_INF_F;
            (&s.tMax.x)[ax]   = CUDART_INF_F;
        }
    }

    refreshMacroDDA(macro, s);
    s.valid = 1;
    return true;
}

static __forceinline__ __device__
float sampleFreeFlight(
    float u,
    float sigma
)
{
    u = fminf(fmaxf(u, 0.0f), 0.99999994f);
    sigma = fmaxf(sigma, 1e-7f);

    return - log1pf(-u) / sigma;
}

static __forceinline__ __device__
bool advanceMacroDDA(
    const MacrocellView& macro,
    MacroDDAState& s)
{
    if (!s.valid) return false;

    const float tNext = s.tEnd;
    if (!(s.tExit > tNext)) {
        s.valid = 0;
        return false;
    }

    s.t = tNext;

    const float eps = 1e-6f * fmaxf(1.0f, fabsf(tNext));

    if (s.step.x != 0 && s.tMax.x <= tNext + eps) {
        s.cell.x += s.step.x;
        s.tMax.x += s.tDelta.x;
    }
    if (s.step.y != 0 && s.tMax.y <= tNext + eps) {
        s.cell.y += s.step.y;
        s.tMax.y += s.tDelta.y;
    }
    if (s.step.z != 0 && s.tMax.z <= tNext + eps) {
        s.cell.z += s.step.z;
        s.tMax.z += s.tDelta.z;
    }

    if (!isInsideMacrocellGrid(s.cell, macro.dims)) {
        s.valid = 0;
        return false;
    }

    refreshMacroDDA(macro, s);
    return true;
}


template <class PRD, class AccT, class SamplerT>
// static __forceinline__ __device__
static __noinline__ __device__
bool deltaTrack_localMajorant(
    PRD& prd,
    const VDBGeomData& vdb,
    const float densityScale,
    const nanovdb::FloatGrid* grid,
    const AccT& acc,
    const SamplerT& sampler,
    const float3& rayOriginObject,
    const float3& rayDirectionObject,
    float tEnter,
    float tExit,
    float sigmaTScale,
    float& outT
)
{
    
    if(!(tExit > tEnter)) return false;

    // if (!optixLaunchParams.vdbs) return false;

    (void)acc;

    const nanovdb::Vec3f rayOriginWorld(rayOriginObject.x, rayOriginObject.y, rayOriginObject.z);
    const nanovdb::Vec3f rayDirectionWorld(rayDirectionObject.x, rayDirectionObject.y, rayDirectionObject.z);
    
    const nanovdb::Vec3f rayOriginIndex     = worldToIndexPoint(grid, rayOriginWorld);
    const nanovdb::Vec3f rayDirectionIndex  = worldToIndexVector(grid, rayOriginWorld, rayDirectionWorld);

    const MacrocellView macro = makeMacrocellView(vdb, sigmaTScale);
    MacroDDAState dda;
    // RNG state をローカルへ退避（レジスタに乗りやすい）
    Random random = prd.random;

    if (!initMacroDDA(macro, rayOriginIndex, rayDirectionIndex, tEnter, tExit, dda)) {
       prd.random = random;
        return false;
    }

    float t = dda.t;
    
    // LocalSegment seg;
    
    // seg = getLocalSegment(
    //     grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
    //     t, tExit, densityScale, sigmaTScale
    // );
    
    // seg.tEnd = tExit;
    // seg.majorant = grid->tree().root().maximum() * densityScale * sigmaTScale;
    // const MacrocellView macro = makeMacrocellView(vdb, sigmaTScale);
    // LocalSegment seg = getMacrocellSegment(macro, rayOriginIndex, rayDirectionIndex, t, tExit);

    
    for(int i = 0; i < 4096; ++i){
        // はみ出た場合
        if (!dda.valid || t >= tExit) break;

        
        // if(t >= tExit) return false;

        // if(seg.majorant <= 0.0f){
        //     t = seg.tEnd;
        //     if(t >= tExit) return false;
        //     seg = getLocalSegment(
        //         grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
        //         t, tExit, densityScale, sigmaTScale
        //     );
        //     continue;
        // }
        if (dda.majorant <= 0.0f) {
            t = dda.tEnd;
            if (!advanceMacroDDA(macro, dda)) break;
            continue;
        }
        
        const float tCand = t + sampleFreeFlight(random(), dda.majorant);
        // セルをはみ出すくらい大きな工程をサンプリングしたら，境界で止める
        if(tCand >= dda.tEnd){
            t = dda.tEnd;
            if(t >= tExit) {
                prd.random = random;
                return false;
            }
            // seg = getLocalSegment(
            //     grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
            //     t, tExit, densityScale, sigmaTScale
            // );
            if (!advanceMacroDDA(macro, dda)) break;
            continue;
        }

        t = tCand;

        // const nanovdb::Vec3f positionWorld = rayOriginWorld + t * rayDirectionWorld;
        const nanovdb::Vec3f positionIndex = rayOriginIndex + t * rayDirectionIndex;
        const float density = sampler(positionIndex);

        const float sigmaT = fmaxf(density * densityScale * sigmaTScale, 0.0f);

        // null collision かどうか
        float ratio = sigmaT / dda.majorant;
        ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

        if(random() < ratio){
            outT = t;
            prd.random = random;
            return true;
        }
    }

    prd.random = random;
    return false;
}

template <class PRD, class AccT, class SamplerT>
// static __forceinline__ __device__
static __noinline__ __device__
float ratioTrack_localMajorant(
    PRD& prd,
    const VDBGeomData& vdb,
    const float densityScale,
    const nanovdb::FloatGrid* grid,
    const AccT& acc,
    const SamplerT& sampler,
    const float3& rayOriginObject,
    const float3& rayDirectionObject,
    float tEnter,
    float tExit,
    float sigmaTScale
)
{
    if(!(tExit > tEnter)) return 1.0f;

    // if (!optixLaunchParams.vdbs) return 1.0f;

    // if(!(tExit > tEnter)) return 1.0f;
    (void) acc;

    const nanovdb::Vec3f rayOriginWorld(rayOriginObject.x, rayOriginObject.y, rayOriginObject.z);
    const nanovdb::Vec3f rayDirectionWorld(rayDirectionObject.x, rayDirectionObject.y, rayDirectionObject.z);
    const nanovdb::Vec3f rayOriginIndex     = worldToIndexPoint(grid, rayOriginWorld);
    const nanovdb::Vec3f rayDirectionIndex  = worldToIndexVector(grid, rayOriginWorld, rayDirectionWorld);
    
    const MacrocellView macro = makeMacrocellView(vdb, sigmaTScale);
    MacroDDAState dda;

    Random random = prd.random;

    if (!initMacroDDA(macro, rayOriginIndex, rayDirectionIndex, tEnter, tExit, dda)) {
        prd.random = random;
        return 1.0f;
    }

    float t = tEnter;
    float transmittance = 1.0f;

    for(int i = 0; i < 4096; ++i){
        // はみ出た場合はレイマーチング修了
        if (!dda.valid || t >= tExit) break;
        // if(t >= tExit) break;

        // 空だった場合はまとめてスキップ
        // if(seg.majorant <= 0.0f){
        //     t = seg.tEnd;
        //     if(t >= tExit) break;
        //     seg = getLocalSegment(
        //         grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
        //         t, tExit, densityScale, sigmaTScale
        //     );
        //     continue;
        // }
        if (dda.majorant <= 0.0f) {
           t = dda.tEnd;
            if (!advanceMacroDDA(macro, dda)) break;
            continue;
        }
        
        // 自由行程サンプリング
        const float tCand = t + sampleFreeFlight(random(), dda.majorant);

        // セルをはみ出すくらい大きな工程をサンプリングしたら，境界で止める
        if(tCand >= dda.tEnd){
            t = dda.tEnd;
            if(!advanceMacroDDA(macro, dda)) break;
            continue;
        }

        // 候補点で密度を評価
        t = tCand;

        const nanovdb::Vec3f positionIndex = rayOriginIndex + t * rayDirectionIndex;
        const float density = sampler(positionIndex);
        const float sigmaT = fmaxf(density * densityScale * sigmaTScale, 0.0f);

        // null collision かどうか
        float ratio = 1.0f - (sigmaT / dda.majorant);
        ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

        transmittance *= ratio;

        // ほぼ透過しない場合は打ち切る
        if(transmittance < 1e-4f) {
            prd.random = random;
            return 0.0f;
        }
    }
    prd.random = random;
    return transmittance;

}
#endif // TRACE_VOLUME_CUH_
