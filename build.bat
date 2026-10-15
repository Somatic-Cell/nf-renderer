@echo off
rmdir /S build
mkdir build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
:: cmake --build . --config Debug --verbose 
cmake --build build --config Release --verbose 

@REM  cmake -S tests/m4_nf -B build-m4 ^
@REM    -DM4_BUILD_OPTIX=ON ^
@REM    -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"

@REM  cmake --build build-m4 --config Release

@REM  ctest --test-dir build-m4 -C Release --output-on-failure