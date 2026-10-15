cmake_minimum_required(VERSION 3.18)
project(M4NFValidation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(M4_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_library(m4_nf_asset STATIC "${M4_ROOT}/src/m4_nf/asset.cpp")
target_include_directories(m4_nf_asset PUBLIC "${M4_ROOT}/include")
add_executable(m4_nf_probe_cpu "${M4_ROOT}/src/m4_nf/probe_cpu.cpp")
target_link_libraries(m4_nf_probe_cpu PRIVATE m4_nf_asset)
if(MSVC)
  target_compile_options(m4_nf_asset PRIVATE /utf-8 /fp:precise)
  target_compile_options(m4_nf_probe_cpu PRIVATE /utf-8 /fp:precise)
else()
  target_compile_options(m4_nf_asset PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
  target_compile_options(m4_nf_probe_cpu PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()

option(M4_BUILD_OPTIX "Build the standalone FP32/FP16 OptiX inference probe" OFF)
set(M4_CUDA_CC "86" CACHE STRING "Virtual architecture for the OptiX inference module")
set(OPTIX_ROOT "" CACHE PATH "OptiX SDK root, containing include/optix.h")
if(M4_BUILD_OPTIX)
  find_package(CUDAToolkit 12.0 REQUIRED)
  find_path(M4_OPTIX_INCLUDE optix.h HINTS "${OPTIX_ROOT}/include" "${OPTIX_ROOT}" REQUIRED)
  if(NOT CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc is required for the OptiX probe")
  endif()
  set(m4_cuda_options --optix-ir "--gpu-architecture=compute_${M4_CUDA_CC}"
      -std=c++17 --relocatable-device-code=true --generate-line-info
      "-I${M4_OPTIX_INCLUDE}" "-I${M4_ROOT}/include")
  if(MSVC)
    list(APPEND m4_cuda_options "-Xcompiler=/utf-8" "-Xcompiler=/Zc:__cplusplus")
  endif()
  file(GLOB m4_headers CONFIGURE_DEPENDS "${M4_ROOT}/include/m4_nf/*.hpp")
  set(m4_ir "${CMAKE_CURRENT_BINARY_DIR}/m4_nf_probe.optixir")
  add_custom_command(OUTPUT "${m4_ir}"
      COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" ${m4_cuda_options}
              "${M4_ROOT}/shaders/m4_nf/probe.cu" -o "${m4_ir}"
      DEPENDS ${m4_headers} "${M4_ROOT}/shaders/m4_nf/probe.cu" "${M4_ROOT}/shaders/m4_nf/coop_mlp.cuh"
      VERBATIM)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(m4_ptx "${CMAKE_CURRENT_BINARY_DIR}/m4_nf_probe.ptx")
  set(m4_ptx_options ${m4_cuda_options})
  list(REMOVE_ITEM m4_ptx_options --optix-ir)
  list(APPEND m4_ptx_options --ptx)
  add_custom_command(OUTPUT "${m4_ptx}"
      COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" ${m4_ptx_options}
              "${M4_ROOT}/shaders/m4_nf/probe.cu" -o "${m4_ptx}"
      DEPENDS ${m4_headers} "${M4_ROOT}/shaders/m4_nf/probe.cu" "${M4_ROOT}/shaders/m4_nf/coop_mlp.cuh"
              "${M4_ROOT}/tools/m4_nf/check_fp64.py"
      VERBATIM)
  # The audit always runs, even if an earlier failed audit left a PTX file.
  add_custom_target(m4_nf_fp64_audit
      COMMAND "${Python3_EXECUTABLE}" "${M4_ROOT}/tools/m4_nf/check_fp64.py"
              "${m4_ptx}" --report "${CMAKE_CURRENT_BINARY_DIR}/m4_nf_fp64_check.json"
      DEPENDS "${m4_ptx}"
      VERBATIM)
  add_executable(m4_nf_probe_gpu "${M4_ROOT}/src/m4_nf/probe_optix.cpp"
                 "${M4_ROOT}/src/m4_nf/gpu_asset.cpp" "${m4_ir}")
  add_dependencies(m4_nf_probe_gpu m4_nf_fp64_audit)
  target_include_directories(m4_nf_probe_gpu PRIVATE "${M4_OPTIX_INCLUDE}")
  target_link_libraries(m4_nf_probe_gpu PRIVATE m4_nf_asset CUDA::cuda_driver ${CMAKE_DL_LIBS})
  # optix_stubs.h includes Windows.h on Windows. Disable its min/max macros
  # before any header is parsed, including optix_stack_size.h and <limits>.
  # This standalone project does not inherit the renderer root's definitions.
  if(WIN32)
    target_compile_definitions(m4_nf_probe_gpu PRIVATE NOMINMAX)
  endif()
  if(MSVC)
    target_compile_options(m4_nf_probe_gpu PRIVATE /utf-8 /fp:precise)
  else()
    target_compile_options(m4_nf_probe_gpu PRIVATE -Wall -Wextra -fno-fast-math)
  endif()
endif()

enable_testing()
add_executable(m4_nf_core_tests "${CMAKE_CURRENT_LIST_DIR}/test_core.cpp")
target_link_libraries(m4_nf_core_tests PRIVATE m4_nf_asset)
if(MSVC)
  target_compile_options(m4_nf_core_tests PRIVATE /utf-8 /fp:precise)
else()
  target_compile_options(m4_nf_core_tests PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()
add_test(NAME m4_nf_core COMMAND m4_nf_core_tests "${M4_ROOT}/assets/m4_nf/best/model.bin")
