#include "texture.hpp"
#include <algorithm>


std::string getFileExtension(const std::string& filePath)
{
    size_t dotPos = filePath.find_last_of('.');
    if(dotPos == std::string::npos) return "";
    
    std::string ext = filePath.substr(dotPos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

std::wstring toWString(const std::string & str)
{
    if (str.empty()) return std::wstring();

    int neededSize = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0);
    std::wstring wstr(neededSize, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], neededSize);
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
    DXGI_FORMAT targetFormat = DXGI_FORMAT_R8G8B8A8_UNORM 
)
{
    DirectX::ScratchImage image;
    HRESULT hr = E_FAIL;

    std::string ext = getFileExtension(imgFilePath);
    std::wstring wPath = toWString(imgFilePath);

    if(ext == "dds") {
        hr = DirectX::LoadFromDDSFile(wPath.c_str(), DirectX::DDS_FLAGS_NONE, nullptr, image);
    } else {
        hr = DirectX::LoadFromWICFile(wPath.c_str(), DirectX::WIC_FLAGS_NONE, nullptr, image);
    }

    if(FAILED(hr)) {
        std::cerr << "Failed to load texture: " << imgFilePath << std::endl;
        return false;
    }

    const DirectX::Image* src = image.GetImage(0, 0, 0);
    std::cout << "Loaded texture: " << imgFilePath << std::endl;
    std::cout << "Size: " << src->width << "x" << src->height << std::endl;
    std::cout << "Format: " << src->format << ":"<< DxgiFormatToString(src->format) << std::endl;

    //圧縮フォーマットだった場合，展開する
    DirectX::ScratchImage decompressed;
    if (DirectX::IsCompressed(src->format)) {
        hr = DirectX::Decompress(*src, DXGI_FORMAT_R8G8B8A8_UNORM, decompressed);
        if (FAILED(hr)) {
            std::cerr << "Failed to decompress texture: " << imgFilePath << std::endl;
            std::cerr << "Format: " << src->format << ":"<< DxgiFormatToString(src->format) << std::endl;
            return false;
        }
        src = decompressed.GetImage(0, 0, 0);
    }
    //　DXGI_FORMAT_R8G8B8A8_UNORMに 変換する
    DirectX::ScratchImage converted;
    if(src->format != targetFormat) {
        hr = DirectX::Convert(*src, targetFormat, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, converted);
        if(FAILED(hr)){
            std::cerr << "Failed to convert texture: " << imgFilePath << std::endl;
            std::cerr << "Format: " << src->format << ":"<< DxgiFormatToString(src->format) << std::endl;
            return false;
        }
        src = converted.GetImage(0, 0, 0);
    }
    
    outTex.width = static_cast<uint32_t>(src->width);
    outTex.height = static_cast<uint32_t>(src->height);
    outTex.format = src->format;
    size_t pixelSize = src->height * src->rowPitch;
    outTex.pixels.resize(pixelSize);
    std::memcpy(outTex.pixels.data(), src->pixels, pixelSize);
    return true;
}

void convertTextureSpecularToRoughnessMetallic(
    const LoadedTexture& srcSpecular,
    const bool isMetal,
    const float glossiness,
    LoadedTexture& dstARMTexture
)
{
    uint32_t width = srcSpecular.width;
    uint32_t height = srcSpecular.height;

    dstARMTexture.width = width;
    dstARMTexture.height = height;
    dstARMTexture.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dstARMTexture.pixels.resize(width * height);

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++){
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

            dstARMTexture.pixels[y * width + x] = rgba;
        }
    }
}
