#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <DirectXTex.h>
#include <vector>


struct Texture{
    ~Texture()
    {
        if(pixel){
                delete[] pixel;
        }
    }

    uint32_t    *pixel {nullptr};
    int2        resolution = make_int2(-1, -1);
    bool        isDiffuseTexture       {false};
};

struct LoadedTexture {
    uint32_t width = 0;
    uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    // One uint32_t per RGBA8 pixel, exactly width * height elements. Rows are
    // tightly packed; DirectXTex rowPitch padding is not stored here. RGB bytes
    // retain their encoding for role-specific sRGB decoding in the CUDA sampler.
    std::vector<uint32_t> pixels;

    template <typename T>
    const T* dataAs() const {
        return reinterpret_cast<const T*>(pixels.data());
    }

    template <typename T>
    T* dataAs() const {
        return reinterpret_cast<T*>(pixels.data());
    }
};

// The renderer's storage contract accepts R8G8B8A8_UNORM only. Load the full-size
// base mip of a single 2D image. Array/cube/volume input is an explicit error.
// Returns false with a path/stage diagnostic on failure; outTex is unchanged.
bool textureLoader(
    const std::string &imgFilePath,
    LoadedTexture& outTex,
    DXGI_FORMAT targetFormat
);

// Requires finite glossiness and the same tightly packed RGBA8 contract.
// Invalid input throws; dstARMTexture is unchanged on failure.
void convertTextureSpecularToRoughnessMetallic(
    const LoadedTexture& srcSpecular,
    const bool isMetal,
    const float glossiness,
    LoadedTexture& dstARMTexture
);

std::string getFileExtension(const std::string& filePath);

std::wstring toWString(const std::string & str);


#endif // TEXTURE_HPP_
