# =========================================
# File Name : CMakeLists.txt
# Encoding  : UTF-8
# =========================================

### JP: CMake のバージョン
cmake_minimum_required(VERSION 3.18)

### CUDA
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures for host CUDA targets")

set(PROJECT_NAME "NeuralPFRenderer")
set(TARGET_NAME ${PROJECT_NAME})

project(${PROJECT_NAME}
    VERSION 0.1
    DESCRIPTION "This is a simple project to develop a renderer"
    LANGUAGES C CXX CUDA                # JP: CUDA をリストに登録することで， FindCUDA module の呼び出しが不要
)

# -----------------------------------------------------
# ビルドの設定
# -----------------------------------------------------

### JP: C++ 標準の設定
set(CMAKE_CXX_STANDARD 17)              # JP: 使用する C++ のバージョン
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # JP: 上記の標準を必須として使用することを要求
set(CMAKE_CXX_EXTENSIONS OFF)           # JP: ベンダー拡張は OFF

set(utils_dir ${PROJECT_SOURCE_DIR}/utils)
set(
    CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "${utils_dir}/cmake")

find_package(CUDAToolkit 12.0 REQUIRED) 
find_package(OptiX9)

if(OptiX9_FOUND)
    set(OPTIX_INCLUDE_DIR "${OPTIX9_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OptiX SDK 9.0.0 not found.")
endif()

include(${utils_dir}/cmake/BuildSettings.cmake)
include(${utils_dir}/cmake/nvcuda_compile_module.cmake)

# -----------------------------------------------------
# サブモジュールの読み込み
# -----------------------------------------------------

### utils 
add_subdirectory(${utils_dir}) # ${utils_dir} 以下にある CMakeList.txt をサブプロジェクトしてビルドに追加

### ext
set(EXT_DIR ${PROJECT_SOURCE_DIR}/ext)

### assimp
set(ASSIMP_BUILD_ZLIB            ON  CACHE BOOL "Build your own zlib" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS    OFF CACHE BOOL "If the supplementary tools for Assimp are built in addition to the library."  FORCE)
set(ASSIMP_BUILD_SAMPLES         OFF CACHE BOOL "If the official samples are built as well (needs Glut)."  FORCE)
set(ASSIMP_BUILD_TESTS           OFF CACHE BOOL "If the test suite for Assimp is built in addition to the library." FORCE)
set(ASSIMP_INSTALL               OFF CACHE BOOL "Disable this if you want to use assimp as a submodule." FORCE)
set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF CACHE BOOL "Disable this if ." FORCE)
set(ASSIMP_NO_EXPORT             ON  CACHE BOOL "Disable this if you want to use assimp as a submodule." FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS    OFF CACHE BOOL ""  FORCE)
set(ASSIMP_BUILD_NO_EXCEPTIONS   OFF CACHE BOOL ""  FORCE)

# Windows 環境
if(MSVC)
    option(ASSIMP_BUILD_ASSIMP_VIEW "If the Assimp view tool is built. (requires DirectX)" OFF)
endif()
add_subdirectory(${EXT_DIR}/assimp)

# Assimp が作る「毎回走る」ターゲットを既定ビルドから外す
if(TARGET UpdateAssimpLibsDebugSymbolsAndDLLs)
  set_target_properties(UpdateAssimpLibsDebugSymbolsAndDLLs PROPERTIES
    EXCLUDE_FROM_DEFAULT_BUILD TRUE
    EXCLUDE_FROM_ALL TRUE
  )
endif()

### DirectXTex
add_subdirectory(${EXT_DIR}/DirectXTex)

# OpenGL
find_package(OpenGL REQUIRED)

# glfw
set(OpenGL_GL_PREFERENCE LEGACY) # LEGACY - OLD, GLVND - NEW
if (WIN32)
    set(glfw_dir ${PROJECT_SOURCE_DIR}/ext/glfw)
    include_directories(${glfw_dir}/include)
    add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
    find_package(glfw3 REQUIRED)
endif()

# imgui
set(IMGUI_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ext/imgui)
set(IMGUI_BACKENDS_DIR ${IMGUI_SOURCE_DIR}/backends)

set(IMGUI_SOURCES
    ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
    ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
    ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
    ${IMGUI_SOURCE_DIR}/imgui.cpp
    ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
    ${IMGUI_SOURCE_DIR}/imgui.h
)

set(IMGUI_BACKEND_SOURCES
    ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.h
    ${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.h
)

add_library(imgui STATIC
    ${IMGUI_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
)

target_include_directories(imgui PUBLIC
    ${IMGUI_SOURCE_DIR}
    ${IMGUI_BACKENDS_DIR}
)

target_link_libraries(imgui
    ${OPENGL_gl_LIBRARY}
    glfw
)

# json
add_subdirectory(${EXT_DIR}/json)

# fpng
add_subdirectory(${EXT_DIR}/fpng)

# openvdb
option(OPENVDB_BUILD_NANOVDB "to enable NanoVDB" ON)
option(OPENVDB_ENABLE_UNINSTALL "uninstall OpenVDB" OFF)
option(NANOVDB_USE_OPENVDB "to use OpenVDB in NanoVDB" ON)


add_subdirectory(${EXT_DIR}/openvdb)

set(NANOVDB_INCLUDE_DIR "${EXT_DIR}/openvdb/nanovdb")

# include(${utils_dir}/cmake/BuildVDB.cmake)

### viewer
add_subdirectory(viewer)
target_link_libraries(viewer
    PRIVATE 
    utils 
    imgui
    # ${OPENGL_gl_LIBRARY}
 )

# include_directories(${OPTIX_INCLUDE_DIR})


# -----------------------------------------------------
# .cu ファイルのコンパイル
# -----------------------------------------------------

set(CPP_VER_CUDA c++20)
set(USE_OPTIX_IR FALSE)
set(CUDA_CC 86 CACHE STRING "Virtual CUDA architecture for OptiX/PTX modules")


### コンパイルするファイルを列挙
# ptx ファイルに変換したいファイル群
set(
    CUDA_KERNELS
    ${PROJECT_SOURCE_DIR}/kernels/toneMap.cu
    ${PROJECT_SOURCE_DIR}/kernels/copy_buffer_to_surface.cu
)

# optixir に変換したいファイル群
set(
    OPTIX_KERNELS
    ${PROJECT_SOURCE_DIR}/shaders/entry/raygen.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ah_radiance.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ah_shadow.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ch_radiance.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ch_radiance_vdb.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/ch_shadow.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/miss_radiance.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/miss_shadow.cu
    ${PROJECT_SOURCE_DIR}/shaders/entry/intersection_vdb.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/disney_principled_brdf.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/diffuse_brdf.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/glass_bsdf.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/light_sample.cu
    ${PROJECT_SOURCE_DIR}/shaders/callable/lens.cu
    ${PROJECT_SOURCE_DIR}/shaders/m4_rain/nf_scatter.cu
    ${PROJECT_SOURCE_DIR}/shaders/m4_rain/sphere_scatter.cu
)

if(OptiX9_FOUND)
    set(USE_OPTIX_IR TRUE)
else()
    set(
        CUDA_KERNELS
        ${CUDA_KERNELS}
        ${OPTIX_KERNELS}
    )
    set(OPTIX_KERNELS)
endif()

# .cu ファイルをコンパイルするのに必要なヘッダファイルたち
set(
    GPU_KERNEL_DEPENDENCIES
    ${PROJECT_SOURCE_DIR}/utils/helper_math.h
    ${PROJECT_SOURCE_DIR}/utils/my_math.hpp
    ${PROJECT_SOURCE_DIR}/include/launch_params.h
    ${PROJECT_SOURCE_DIR}/shaders/params/per_ray_data.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/random_number_generator.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/shader_common.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/trace_volume.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/phase_function.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/neural_phase_function.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/atmosphere_sampling_device.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/nsf_simd_constant_flow_flexible_data.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/nsf_simd_constant_backend.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/atmosphere_sampling_device.cuh
    ${PROJECT_SOURCE_DIR}/shaders/config.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/m1_surface.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/m1_diagnostics.cuh
    ${PROJECT_SOURCE_DIR}/shaders/device/m3_atmosphere.cuh
    ${PROJECT_SOURCE_DIR}/include/m1/render_params.h
    ${PROJECT_SOURCE_DIR}/include/m1/diagnostics.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/light_pdf.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/spectral_types.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/rgb2spec_device.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/bsdf.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/math.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/fp32_numeric.hpp
    ${PROJECT_SOURCE_DIR}/include/m1/bounded_trig.hpp
    ${PROJECT_SOURCE_DIR}/include/m2/alpha.hpp
    ${PROJECT_SOURCE_DIR}/include/m2/tangent_frame.hpp
    ${PROJECT_SOURCE_DIR}/include/m3/tracking_parameters.hpp
    ${PROJECT_SOURCE_DIR}/include/m3/tracking_float.hpp
    ${PROJECT_SOURCE_DIR}/include/m3/tracking_geometry.hpp
    ${PROJECT_SOURCE_DIR}/include/m3/tracking.hpp
    ${PROJECT_SOURCE_DIR}/include/m3/transport.hpp
)

# Keep M4 headers in the existing custom nvcc dependency graph.
file(GLOB M4_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/m4_nf/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/m4_rain/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/m4_mie/*.hpp")
list(APPEND GPU_KERNEL_DEPENDENCIES ${M4_HEADERS}
    "${PROJECT_SOURCE_DIR}/shaders/m4_nf/coop_mlp.cuh")

# .cu ファイルを .ptx ファイルにコンパイル (実装は nvcuda_compile_module.cmake モジュール内)
set(CUDA_NVCC_OPTIONS
    "--generate-line-info"                      # 行番号を付与し，デバッグツール等でどの行が実行中か追跡可能にするオプション
    "--gpu-architecture=compute_${CUDA_CC}"     # GPU の構成
    "--gpu-code=sm_${CUDA_CC}" 
    "-std=${CPP_VER_CUDA}"                  
    "-cudart" "none"                          # PTX module, no host runtime stub
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-Xptxas=-v" "-Xptxas=--warn-spills"
    "-I${NANOVDB_INCLUDE_DIR}"
    "-I${PROJECT_SOURCE_DIR}/include"
)

# Accurate sqrt/division are used by the shared spectral BSDF and PDF code.
# A fast-math build is not part of the M1 numerical validation.
if(MSVC)
    list(APPEND CUDA_NVCC_OPTIONS "-Xcompiler=/utf-8" "-Xcompiler=/Zc:__cplusplus")
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Debug")   # Debug ビルドのときだけ
    list(APPEND 
        CUDA_NVCC_OPTIONS
        "-D_DEBUG=1"                    # _DEBUG を定義
        "-G"                            # GPU のデバッグ情報を含める                    
    )
endif()


nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    # 例えば build/Debug/ptxes　以下に .ptx ファイルが生成
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS ${CUDA_NVCC_OPTIONS}
)

message(STATUS "CUDA_KERNELS=${CUDA_KERNELS}")


# .cu ファイルを .optixir ファイルにコンパイル (実装は nvcuda_compile_module.cmake モジュール内) 
if(USE_OPTIX_IR)
    set(OPTIX_NVCC_OPTIONS
        "--generate-line-info"                      # 行番号を付与し，デバッグツール等でどの行が実行中か追跡可能にするオプション
        "--gpu-architecture=compute_${CUDA_CC}"     # GPU の構成
        "-std=${CPP_VER_CUDA}"
        "-cudart" "none"
        "--relocatable-device-code=true"
        "-I${OPTIX_INCLUDE_DIR}"
        "-Xptxas=-v" "-Xptxas=--warn-spills"
        "-I${PROJECT_SOURCE_DIR}/include"
    )

    if(MSVC)
        list(APPEND OPTIX_NVCC_OPTIONS
            "-Xcompiler=/utf-8"
            "-Xcompiler=/Zc:__cplusplus"
        )
    endif()

    set(NANOVDB_INCLUDE_DIR "${EXT_DIR}/openvdb/nanovdb")
    list(APPEND
        OPTIX_NVCC_OPTIONS
        "-I${NANOVDB_INCLUDE_DIR}"
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug") # Debug ビルドのときだけ
        list(APPEND 
            OPTIX_NVCC_OPTIONS
            "-D_DEBUG=1"             # _DEBUG を定義
            "-G"                     # GPU のデバッグ情報を含める                    
            "-O0"                    # 最適化を無効化 
        )
    endif()
    
    nvcuda_compile_optix_ir(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS ${OPTIX_NVCC_OPTIONS}
)
endif()


# ビルドした結果をまとめる
set(BUILD_CUDA_KERNELS
  ${CUDA_PTXES} 
  ${OPTIX_PTXES}
)


# -----------------------------------------------------
# 実行ファイルの定義
# -----------------------------------------------------
set(SOURCE_DIR src)
set(INCLUDE_DIR include)

# The pinned upstream RGB2Spec runtime is compiled as C. The optimizer and its
# source CMake settings are not part of the renderer build; the model is shipped.
add_library(m1_rgb2spec STATIC third_party/rgb2spec/rgb2spec.c)
set_target_properties(m1_rgb2spec PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)
target_include_directories(m1_rgb2spec PUBLIC third_party/rgb2spec)

add_library(m1_spectral_assets STATIC src/m1/spectral_assets.cpp)
target_compile_features(m1_spectral_assets PUBLIC cxx_std_17)
target_include_directories(m1_spectral_assets PUBLIC include)
target_link_libraries(m1_spectral_assets PRIVATE m1_rgb2spec)

add_library(m4_nf_assets STATIC src/m4_nf/asset.cpp src/m4_nf/gpu_asset.cpp)
add_library(m4_mie_assets STATIC src/m4_mie/asset.cpp src/m4_mie/gpu_asset.cpp)
target_include_directories(m4_mie_assets PUBLIC include)
target_compile_features(m4_mie_assets PUBLIC cxx_std_17)
target_link_libraries(m4_mie_assets PUBLIC CUDA::cuda_driver)
if(WIN32)
    target_compile_definitions(m4_mie_assets PRIVATE NOMINMAX)
endif()
if(MSVC)
    target_compile_options(m4_mie_assets PRIVATE /utf-8 /fp:precise)
endif()
target_include_directories(m4_nf_assets PUBLIC include "${OPTIX_INCLUDE_DIR}")
target_compile_features(m4_nf_assets PUBLIC cxx_std_17)
target_link_libraries(m4_nf_assets PUBLIC CUDA::cuda_driver)
if(WIN32)
    target_compile_definitions(m4_nf_assets PRIVATE NOMINMAX)
endif()
if(MSVC)
    target_compile_options(m4_nf_assets PRIVATE /utf-8 /fp:precise)
endif()

add_executable(${TARGET_NAME}
    "${BUILD_CUDA_KERNELS}"
    "${SOURCE_DIR}/main.cpp"
    "${SOURCE_DIR}/renderer.cpp"
    "${SOURCE_DIR}/texture.cpp"
    "${SOURCE_DIR}/model.cpp"
    "${SOURCE_DIR}/application.cpp"
    "${SOURCE_DIR}/atmosphere_lut.cpp"
    "${INCLUDE_DIR}/vdb_loader.hpp"
    "${INCLUDE_DIR}/optix8.hpp"
    "${INCLUDE_DIR}/renderer.hpp"
    "${INCLUDE_DIR}/model.h"
    "${INCLUDE_DIR}/launch_params.h"
    "${INCLUDE_DIR}/cuda_buffer.h"
    "${INCLUDE_DIR}/cuda_texture.h"
    "${INCLUDE_DIR}/atmosphere_lut.h"
    "${INCLUDE_DIR}/ptx_data_reader.hpp"
    "${INCLUDE_DIR}/application.hpp"
    "${INCLUDE_DIR}/mie_tables_txt_loader.h"
    "${INCLUDE_DIR}/safetensors_loader.hpp"
    "${INCLUDE_DIR}/nsf_hyper_checkpoint.hpp"
    "${INCLUDE_DIR}/nsf_hyper_coopvec_pack.hpp"
    "${EXT_DIR}/stb/stb_image_write.h"
    "${EXT_DIR}/fpng/src/fpng.h"
    "${EXT_DIR}/fpng/src/fpng.cpp"
)

# Audit the new NF transport callable, including math-library lowering.
# This target always checks the existing PTX, so a previous failure cannot
# silently pass on a subsequent incremental build.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(M4_RAIN_AUDIT_DIR "${CMAKE_BINARY_DIR}/m4_rain_validation")
set(M4_RAIN_AUDIT_PTX "${M4_RAIN_AUDIT_DIR}/nf_scatter.ptx")
add_custom_command(OUTPUT "${M4_RAIN_AUDIT_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${M4_RAIN_AUDIT_DIR}"
    COMMAND "${CMAKE_CUDA_COMPILER}" ${OPTIX_NVCC_OPTIONS} --ptx
        "${PROJECT_SOURCE_DIR}/shaders/m4_rain/nf_scatter.cu" -o "${M4_RAIN_AUDIT_PTX}"
    DEPENDS ${GPU_KERNEL_DEPENDENCIES} "${PROJECT_SOURCE_DIR}/shaders/m4_rain/nf_scatter.cu"
    VERBATIM)
add_custom_target(m4_rain_fp64_audit
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/m4_nf/check_fp64.py"
        "${M4_RAIN_AUDIT_PTX}" --report "${M4_RAIN_AUDIT_DIR}/fp64_check.json"
    DEPENDS "${M4_RAIN_AUDIT_PTX}" "${PROJECT_SOURCE_DIR}/tools/m4_nf/check_fp64.py"
    VERBATIM)
set(M4_MIE_AUDIT_PTX "${M4_RAIN_AUDIT_DIR}/sphere_scatter.ptx")
add_custom_command(OUTPUT "${M4_MIE_AUDIT_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${M4_RAIN_AUDIT_DIR}"
    COMMAND "${CMAKE_CUDA_COMPILER}" ${OPTIX_NVCC_OPTIONS} --ptx
        "${PROJECT_SOURCE_DIR}/shaders/m4_rain/sphere_scatter.cu" -o "${M4_MIE_AUDIT_PTX}"
    DEPENDS ${GPU_KERNEL_DEPENDENCIES} "${PROJECT_SOURCE_DIR}/shaders/m4_rain/sphere_scatter.cu"
    VERBATIM)
add_custom_target(m4_mie_fp64_audit
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/m4_nf/check_fp64.py"
        "${M4_MIE_AUDIT_PTX}" --report "${M4_RAIN_AUDIT_DIR}/sphere_fp64_check.json"
    DEPENDS "${M4_MIE_AUDIT_PTX}" "${PROJECT_SOURCE_DIR}/tools/m4_nf/check_fp64.py"
    VERBATIM)
add_dependencies(${TARGET_NAME} m4_rain_fp64_audit m4_mie_fp64_audit)

# -----------------------------------------------------
# ターゲットの設定
# -----------------------------------------------------
# コンパイル時のマクロ定義の追加
target_compile_options(
    ${TARGET_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/utf-8>
                      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# CMake 上の変数を C++ に渡す
if (WIN32)
    add_definitions(-DNOMINMAX)
endif()

if(USE_OPTIX_IR)
    add_definitions("-DUSE_OPTIX_IR")
endif()


# インクルードパスの追加
target_include_directories(
    ${TARGET_NAME} PRIVATE
    "${OPTIX_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}"
    ${EXT_DIR}/stb
    ${EXT_DIR}/DirectXTex
    ${EXT_DIR}/assimp/include
    ${EXT_DIR}/json
)

# ライブラリやターゲットをリンク
target_link_libraries(
    ${TARGET_NAME}
    ${OPENGL_gl_LIBRARY}
    utils
    viewer
    glfw
    imgui
    assimp
    DirectXTex
    nlohmann_json::nlohmann_json
    CUDA::cudart        # FindCudaToolkit の成果物
    CUDA::cuda_driver   # FindCudaToolkit の成果物 (cuCtxGetCurrent() はここにいる)
    openvdb
    nanovdb
    m1_spectral_assets
    m4_nf_assets
    m4_mie_assets
)


message(STATUS "----------------------------------------")
message(STATUS "Info :")
message(STATUS "Project name = ${PROJECT_NAME}")
message(STATUS "Project version = ${PROJECT_VERSION}")
message(STATUS "CMAKE_VERSION = ${CMAKE_VERSION}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_LIBDIR = ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "OPTIX = ${optix_LIBRARY}")
message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
