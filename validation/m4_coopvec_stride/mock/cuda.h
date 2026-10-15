#pragma once
// TEST DOUBLE ONLY. This is not a CUDA implementation or an SDK header.
#include <cstddef>
#include <cstdint>
using CUdeviceptr = unsigned long long;
using CUstream = void*;
enum CUresult { CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1 };
CUresult cuGetErrorString(CUresult, const char**);
CUresult cuMemAlloc(CUdeviceptr*, std::size_t);
CUresult cuMemFree(CUdeviceptr);
CUresult cuMemcpyHtoD(CUdeviceptr, const void*, std::size_t);
CUresult cuStreamSynchronize(CUstream);
