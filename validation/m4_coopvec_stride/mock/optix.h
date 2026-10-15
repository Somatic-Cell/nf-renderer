#pragma once
// TEST DOUBLE ONLY. Minimal host API signatures used by gpu_asset.cpp.
#include "cuda.h"
#define OPTIX_VERSION 90000
using OptixDeviceContext = void*;
enum OptixResult { OPTIX_SUCCESS = 0, OPTIX_ERROR_INVALID_VALUE = 7001 };
enum OptixCoopVecElemType { OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 = 0 };
enum OptixCoopVecMatrixLayout {
    OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR = 0,
    OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL = 1
};
struct OptixCoopVecMatrixDescription {
    unsigned N=0, K=0;
    OptixCoopVecElemType elementType{};
    OptixCoopVecMatrixLayout layout{};
    std::size_t rowColumnStrideInBytes=0, offsetInBytes=0, sizeInBytes=0;
};
struct OptixNetworkDescription {
    OptixCoopVecMatrixDescription* layers=nullptr;
    unsigned numLayers=0;
};
const char* optixGetErrorName(OptixResult);
OptixResult optixCoopVecMatrixComputeSize(OptixDeviceContext, unsigned, unsigned,
    OptixCoopVecElemType, OptixCoopVecMatrixLayout, std::size_t, std::size_t*);
OptixResult optixCoopVecMatrixConvert(OptixDeviceContext, CUstream, unsigned,
    const OptixNetworkDescription*, CUdeviceptr, std::size_t,
    const OptixNetworkDescription*, CUdeviceptr, std::size_t);
