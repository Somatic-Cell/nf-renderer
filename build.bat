@echo off
@REM  rmdir /S build
@REM  mkdir build
@REM  cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
@REM      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
@REM  :: cmake --build . --config Debug --verbose 
@REM  cmake --build build --config Release --verbose 

cmake -S tests/m4_nf -B build-m4 ^
  -DM4_BUILD_OPTIX=ON ^
  -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"

cmake --build build-m4 --config Release

ctest --test-dir build-m4 -C Release --output-on-failure