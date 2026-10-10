cmake_minimum_required(VERSION 3.18)
project(NeuralPFRenderer_M1_CPU_Checks LANGUAGES C CXX)

# Configure this directory directly. No GPU, DirectXTex, OpenGL or OptiX is
# required for the portable mathematical and asset checks.
get_filename_component(M1_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

add_library(m1_rgb2spec STATIC "${M1_ROOT}/third_party/rgb2spec/rgb2spec.c")
set_target_properties(m1_rgb2spec PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)
target_include_directories(m1_rgb2spec PUBLIC "${M1_ROOT}/third_party/rgb2spec")

add_library(m1_spectral_assets STATIC "${M1_ROOT}/src/m1/spectral_assets.cpp")
target_include_directories(m1_spectral_assets PUBLIC "${M1_ROOT}/include")
target_link_libraries(m1_spectral_assets PUBLIC m1_rgb2spec)

add_executable(m1_bsdf_check "${M1_ROOT}/tests/test_m1_bsdf.cpp")
target_include_directories(m1_bsdf_check PRIVATE "${M1_ROOT}/include")
add_test(NAME m1_bsdf_check COMMAND m1_bsdf_check)

add_executable(m1_light_pdf_check "${M1_ROOT}/tests/test_m1_light_pdf.cpp")
target_include_directories(m1_light_pdf_check PRIVATE "${M1_ROOT}/include")
add_test(NAME m1_light_pdf_check COMMAND m1_light_pdf_check)

add_executable(m1_diagnostics_check "${M1_ROOT}/tests/test_m1_diagnostics.cpp")
target_include_directories(m1_diagnostics_check PRIVATE "${M1_ROOT}/include")
add_test(NAME m1_diagnostics_check COMMAND m1_diagnostics_check)

add_executable(m2_alpha_check "${M1_ROOT}/tests/test_m2_alpha.cpp")
target_include_directories(m2_alpha_check PRIVATE "${M1_ROOT}/include")
add_test(NAME m2_alpha_check COMMAND m2_alpha_check)

add_executable(m2_tangent_frame_check "${M1_ROOT}/tests/test_m2_tangent_frame.cpp")
target_include_directories(m2_tangent_frame_check PRIVATE "${M1_ROOT}/include")
add_test(NAME m2_tangent_frame_check COMMAND m2_tangent_frame_check)

add_executable(m1_spectral_check "${M1_ROOT}/tests/test_m1_spectral_assets.cpp")
target_link_libraries(m1_spectral_check PRIVATE m1_spectral_assets)
add_test(NAME m1_spectral_check COMMAND m1_spectral_check "${M1_ROOT}/spectrum/m1")

set(M1_JSON_INCLUDE_DIR "${M1_ROOT}/ext/json/single_include" CACHE PATH "Folder containing nlohmann/json.hpp")
if(NOT EXISTS "${M1_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "Initialize the existing ext/json submodule or pass -DM1_JSON_INCLUDE_DIR=...")
endif()
add_executable(m1_scene_check "${M1_ROOT}/tests/test_m1_scene.cpp")
target_include_directories(m1_scene_check PRIVATE "${M1_ROOT}/include" "${M1_JSON_INCLUDE_DIR}")
add_test(NAME m1_scene_check COMMAND m1_scene_check "${M1_ROOT}")

if(MSVC)
    foreach(target m1_bsdf_check m1_light_pdf_check m1_diagnostics_check m2_alpha_check m2_tangent_frame_check m1_spectral_check m1_scene_check m1_spectral_assets)
        target_compile_options(${target} PRIVATE /utf-8 /W4)
    endforeach()
endif()
