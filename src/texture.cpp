#include "texture.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

// Pure storage operations: no DirectXTex, CUDA, colour conversion, or resampling.
// Kept together so the actual production implementation can be exercised by
// portable stride/overflow tests without emulating a Windows image decoder.
namespace texture_storage {

std::size_t rgba8PixelCount(std::size_t width, std::size_t height)
{
    static_assert(sizeof(uint32_t) == 4, "RGBA8 storage requires 4-byte words");
    if (width == 0 || height == 0) {
        throw std::invalid_argument("texture width and height must be positive");
    }
    // Texture::resolution and the existing GPU upload path use signed int2.
    const auto maximumDimension = static_cast<std::size_t>((std::numeric_limits<int>::max)());
    if (width > maximumDimension || height > maximumDimension) {
        throw std::length_error("texture dimensions do not fit the renderer's int2 resolution");
    }
    const auto maximumSize = (std::numeric_limits<std::size_t>::max)();
    if (width > maximumSize / height) {
        throw std::length_error("texture pixel count overflows size_t");
    }
    const std::size_t count = width * height;
    if (count > maximumSize / sizeof(uint32_t) ||
        count > std::vector<uint32_t>().max_size()) {
        throw std::length_error("RGBA8 pixel storage is not representable");
    }
    return count;
}

std::vector<uint32_t> packRgba8Rows(std::size_t width, std::size_t height,
    std::size_t rowPitch, std::size_t slicePitch, const uint8_t* source)
{
    const std::size_t count = rgba8PixelCount(width, height);
    const std::size_t rowBytes = width * sizeof(uint32_t);
    if (!source) {
        throw std::invalid_argument("texture pixel pointer is null");
    }
    if (rowPitch < rowBytes) {
        throw std::invalid_argument("texture rowPitch is smaller than an RGBA8 row");
    }
    // Image::slicePitch includes row padding. Validate it before pointer
    // arithmetic, including height * rowPitch overflow on malformed metadata.
    if (rowPitch > (std::numeric_limits<std::size_t>::max)() / height) {
        throw std::length_error("texture source rowPitch * height overflows size_t");
    }
    if (slicePitch < rowPitch * height) {
        throw std::invalid_argument("texture slicePitch does not contain every source row");
    }

    std::vector<uint32_t> packed(count);
    for (std::size_t y = 0; y < height; ++y) {
        std::memcpy(packed.data() + y * width, source + y * rowPitch, rowBytes);
    }
    return packed;
}

} // namespace texture_storage


std::string getFileExtension(const std::string& filePath)
{
    size_t dotPos = filePath.find_last_of('.');
    if(dotPos == std::string::npos) return "";
    
    std::string ext = filePath.substr(dotPos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

std::wstring toWString(const std::string & str)
{
    if (str.empty()) return std::wstring();

    if (str.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
        throw std::length_error("UTF-8 texture path is too long for the Windows API");
    }
    if (str.find('\0') != std::string::npos) {
        throw std::invalid_argument("texture path contains an embedded NUL");
    }
    const int length = static_cast<int>(str.size());
    const int neededSize = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
        str.data(), length, nullptr, 0);
    if (neededSize == 0) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
            "invalid UTF-8 texture path");
    }
    std::wstring wstr(static_cast<std::size_t>(neededSize), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
        str.data(), length, wstr.data(), neededSize) != neededSize) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
            "failed to convert the texture path to UTF-16");
    }
    return wstr;
}

const char* DxgiFormatToString(DXGI_FORMAT format)
{
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
        case DXGI_FORMAT_BC7_UNORM: return "BC7_UNORM";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R32_FLOAT: return "R32_FLOAT";
        case DXGI_FORMAT_BC1_UNORM: return "DXGI_FORMAT_BC1_UNORM";
        case DXGI_FORMAT_BC3_UNORM: return "DXGI_FORMAT_BC3_UNORM";
        case DXGI_FORMAT_BC5_UNORM: return "DXGI_FORMAT_BC5_UNORM";
        default: return "Unknown Format";
    }
}

bool textureLoader(
    const std::string &imgFilePath,
    LoadedTexture& outTex,
    DXGI_FORMAT targetFormat
)
{
    const char* stage = "validate target format";
    try {
        // This loader preserves encoded RGB / data-channel values. Colour-space
        // decoding is selected later by the texture's role in the CUDA sampler.
        if (targetFormat != DXGI_FORMAT_R8G8B8A8_UNORM) {
            std::cerr << "Texture load failed [" << stage << "]: " << imgFilePath
                      << ": target DXGI format " << static_cast<unsigned>(targetFormat)
                      << " is unsupported; GPU storage requires R8G8B8A8_UNORM" << std::endl;
            return false;
        }

        DirectX::ScratchImage image;
        HRESULT hr = E_FAIL;

        std::string ext = getFileExtension(imgFilePath);
        stage = "convert file path";
        if (imgFilePath.empty()) {
            throw std::invalid_argument("texture path is empty");
        }
        std::wstring wPath = toWString(imgFilePath);

        stage = ext == "dds" ? "decode DDS" : "decode WIC";
        if(ext == "dds") {
            hr = DirectX::LoadFromDDSFile(wPath.c_str(), DirectX::DDS_FLAGS_NONE, nullptr, image);
        } else {
            hr = DirectX::LoadFromWICFile(wPath.c_str(), DirectX::WIC_FLAGS_IGNORE_SRGB, nullptr, image);
        }

        if(FAILED(hr)) {
            std::cerr << "Texture load failed [" << stage << "]: " << imgFilePath
                      << ": HRESULT 0x" << std::hex << static_cast<uint32_t>(hr) << std::dec << std::endl;
            return false;
        }

        stage = "validate source dimensions";
        const auto& metadata = image.GetMetadata();
        if (metadata.dimension != DirectX::TEX_DIMENSION_TEXTURE2D || metadata.depth != 1 ||
            metadata.arraySize != 1 || metadata.IsCubemap() || metadata.mipLevels == 0) {
            std::cerr << "Texture load failed [" << stage << "]: " << imgFilePath
                      << ": expected one 2D image; dimension=" << static_cast<unsigned>(metadata.dimension)
                      << ", depth=" << metadata.depth << ", arraySize=" << metadata.arraySize
                      << ", mipLevels=" << metadata.mipLevels << ", cube=" << metadata.IsCubemap() << std::endl;
            return false;
        }
        texture_storage::rgba8PixelCount(metadata.width, metadata.height);

        // DDS can still carry an _SRGB format. OverrideFormat changes only the
        // format metadata; MakeLinear selects its byte-compatible UNORM alias.
        // Do this before decompress/convert, which otherwise implicitly decode
        // _SRGB input. CUDA then decodes colour RGB once; normal/ARM data stays raw.
        stage = "preserve source encoding";
        const DXGI_FORMAT storageFormat = DirectX::MakeLinear(image.GetMetadata().format);
        if (storageFormat != image.GetMetadata().format && !image.OverrideFormat(storageFormat)) {
            std::cerr << "Failed to preserve texture encoding: " << imgFilePath << std::endl;
            return false;
        }

        const DirectX::Image* src = image.GetImage(0, 0, 0);
        if (!src || !src->pixels || src->width != metadata.width || src->height != metadata.height) {
            throw std::invalid_argument("DirectXTex did not provide the full-size base image");
        }
        std::cout << "Loaded texture: " << imgFilePath << std::endl;
        std::cout << "Size: " << src->width << "x" << src->height << std::endl;
        std::cout << "Format: " << src->format << ":"<< DxgiFormatToString(src->format) << std::endl;

        //圧縮フォーマットだった場合，展開する
        DirectX::ScratchImage decompressed;
        if (DirectX::IsCompressed(src->format)) {
            stage = "decompress base image";
            hr = DirectX::Decompress(*src, DXGI_FORMAT_R8G8B8A8_UNORM, decompressed);
            if (FAILED(hr)) {
                std::cerr << "Failed to decompress texture: " << imgFilePath << std::endl;
                std::cerr << "Format: " << src->format << ":"<< DxgiFormatToString(src->format) << std::endl;
                std::cerr << "HRESULT: 0x" << std::hex << static_cast<uint32_t>(hr) << std::dec << std::endl;
                return false;
            }
            src = decompressed.GetImage(0, 0, 0);
            if (!src || !src->pixels) throw std::invalid_argument("DirectXTex returned no decompressed image");
        }
        //　DXGI_FORMAT_R8G8B8A8_UNORMに 変換する
        DirectX::ScratchImage converted;
        if(src->format != targetFormat) {
            stage = "convert base image to RGBA8";
            hr = DirectX::Convert(*src, targetFormat, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, converted);
            if(FAILED(hr)){
                std::cerr << "Failed to convert texture: " << imgFilePath << std::endl;
                std::cerr << "Format: " << src->format << ":"<< DxgiFormatToString(src->format) << std::endl;
                std::cerr << "HRESULT: 0x" << std::hex << static_cast<uint32_t>(hr) << std::dec << std::endl;
                return false;
            }
            src = converted.GetImage(0, 0, 0);
            if (!src || !src->pixels) throw std::invalid_argument("DirectXTex returned no converted image");
        }

        stage = "pack RGBA8 rows";
        if (src->format != targetFormat || src->width != metadata.width || src->height != metadata.height) {
            throw std::invalid_argument("converted texture format or dimensions do not match the source contract");
        }
        LoadedTexture result;
        result.pixels = texture_storage::packRgba8Rows(src->width, src->height,
            src->rowPitch, src->slicePitch, src->pixels);
        result.width = static_cast<uint32_t>(src->width);
        result.height = static_cast<uint32_t>(src->height);
        result.format = src->format;
        outTex = std::move(result);
        return true;
    } catch (const std::exception& error) {
        std::cerr << "Texture load failed [" << stage << "]: " << imgFilePath
                  << ": " << error.what() << std::endl;
        return false;
    }
}

void convertTextureSpecularToRoughnessMetallic(
    const LoadedTexture& srcSpecular,
    const bool isMetal,
    const float glossiness,
    LoadedTexture& dstARMTexture
)
{
    const std::size_t width = srcSpecular.width;
    const std::size_t height = srcSpecular.height;
    const std::size_t count = texture_storage::rgba8PixelCount(width, height);
    if (srcSpecular.format != DXGI_FORMAT_R8G8B8A8_UNORM || srcSpecular.pixels.size() != count) {
        throw std::invalid_argument("specular conversion requires tightly packed RGBA8 pixels");
    }
    if (!std::isfinite(glossiness)) {
        throw std::invalid_argument("specular conversion requires finite glossiness");
    }
    // Build the result separately so in-place conversion is safe and failure
    // does not leave a partially changed destination. Preserve the old mapping.
    LoadedTexture result;
    result.width = srcSpecular.width;
    result.height = srcSpecular.height;
    result.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    result.pixels.resize(count);

    for(std::size_t y = 0; y < height; y++)
    {
        for(std::size_t x = 0; x < width; x++){
            float roughness = 0.5f;
            float metallic = 0.0f;

            if(isMetal){
                metallic = 1.0f;
            }

            if(glossiness > 0.0f){
                roughness  = std::clamp(1.0f - glossiness, 0.0f, 1.0f);
            } else {
                const uint32_t pixel = srcSpecular.pixels[y * width + x]; // RGBA

                uint8_t rUint = static_cast<uint8_t>( pixel & 0xFF);
                uint8_t gUint = static_cast<uint8_t>((pixel >>   8) & 0xFF);
                uint8_t bUint = static_cast<uint8_t>((pixel >>  16) & 0xFF);

                float r = (float)rUint / 255.0f;
                float g = (float)gUint / 255.0f;
                float b = (float)bUint / 255.0f;

                float spec = fmaxf(r, fmaxf(g, b));
                roughness = std::clamp(1.0f - sqrtf(spec), 0.0f, 1.0f);
            }

            uint8_t roughnessUint8 = static_cast<uint8_t>(roughness * 255.0f);
            uint8_t metallicUint8 = static_cast<uint8_t>(metallic * 255.0f);
            uint32_t rgba =  0 | (roughnessUint8 << 8) | (metallicUint8 << 16);

            result.pixels[y * width + x] = rgba;
        }
    }
    dstARMTexture = std::move(result);
}
