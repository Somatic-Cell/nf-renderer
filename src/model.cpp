#include "model.h"
#include "m2/orca_material.hpp"
#include "m2/orca_scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <set>
#include <unordered_map>
#include <utility>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <assert.h>
#include <string>
#include <memory>

class MeshLoader 
{
public:
    MeshLoader();
    ~MeshLoader();

    bool load(const std::string &fbxFile, Model *model, const sceneIO::Object objectDesc);

private:
    void parseMesh(TriangleMesh & dstMesh, const aiMesh* pSrcMesh, const std::vector<Material*>& materials);
    void parseMaterial(Material& dstMaterial, const aiMaterial* pSrcMaterial, std::vector<Texture*>& textures, const std::string &fbxPath);
    void parseOrcaMaterial(Material& dstMaterial, const aiMaterial& source,
        std::vector<Texture*>& textures, const std::string& fbxPath);
    int loadOrcaTexture(const m2::OrcaTextureBinding& binding, bool colorTexture,
        const std::string& materialName, std::vector<Texture*>& textures, const std::string& fbxPath);
    bool m_orca = false;
    double m_metresPerUnit = 1.0;
    std::unordered_map<std::string, int> m_textureCache;
};

MeshLoader::MeshLoader()
{}

MeshLoader::~MeshLoader()
{}

bool MeshLoader::load(const std::string &fileName, Model* model, const sceneIO::Object objectDesc)
{
    if(fileName.empty()){
        std::cerr << "ERROR: filename is empty." << std::endl;
        return false;
    }

    if (objectDesc.materialProfile != "legacy" && objectDesc.materialProfile != "orca_metal_rough")
        throw std::invalid_argument("Unsupported materialProfile: " + objectDesc.materialProfile);
    m_orca = objectDesc.materialProfile == "orca_metal_rough";
    m_metresPerUnit = 1.0;
    m_textureCache.clear();

    Assimp::Importer importer;
    int flag = 0;
    flag |= aiProcess_Triangulate;              // 全ての面を三角形メッシュに統一
    flag |= aiProcess_PreTransformVertices;     // ノード階層を無視して，全ての頂点をワールド座標に変換
    // flag |= aiProcess_GenSmoothNormals;         // 法線がない場合，シェーディング用の法線を生成
    flag |= aiProcess_CalcTangentSpace;         // 接ベクトル，従法線を計算
    flag |= aiProcess_GenUVCoords;              // UV 座標がない場合，自動生成
    flag |= aiProcess_RemoveRedundantMaterials; // 未使用のマテリアルを削除して軽量化
    flag |= aiProcess_OptimizeMeshes;           // 複数のメッシュをまとめて描画効率を改善

    if (m_orca) {
        std::string extension = std::filesystem::u8path(fileName).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
            return char(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c);
        });
        if (extension != ".fbx")
            throw std::invalid_argument("orca_metal_rough currently requires FBX axis/unit metadata");
        m2::configure_orca_importer(importer);
        // Keep authored material identities and require actual UV coordinates.
        // Missing data must not turn into a generated/default-looking surface.
        flag = static_cast<int>(m2::orca_import_flags);
    }

    auto pScene = importer.ReadFile(fileName.c_str(), flag);

    if(pScene == nullptr){
        std::cerr << "Assimp Error: " << importer.GetErrorString() << std::endl;
        return false;
    }
    if (m_orca) {
        m_metresPerUnit = m2::orca_metres_per_unit(*pScene);
        if (!pScene->mRootNode || !pScene->mNumMeshes || !pScene->mMeshes)
            throw std::runtime_error("ORCA FBX has no renderable scene meshes");
        std::cout << "ORCA FBX metres per authored unit: " << m_metresPerUnit << std::endl;
    }

    std::cout << "Detected " << pScene->mNumMeshes << "meshes." << std::endl;
    std::cout << "Detected " << pScene->mNumMaterials << "materials." << std::endl;

    // メモリ確保
    model->meshes.clear();
    model->meshes.resize(pScene->mNumMeshes);

    // マテリアルのメモリを確保
    model->materials.clear();
    model->materials.resize(pScene->mNumMaterials);

    // テクスチャのメモリをクリア（何枚あるか不明なので，push_back()で順次登録）
    model->textures.clear();


    // ------------------------
    // マテリアルデータの読み込み
    // ------------------------

    std::cout << "Start to convert material data to use this renderer." << std::endl;

    if(model->materials.empty()){
        throw std::runtime_error("could not parse materials ");
    }

    // マテリアルのデータを，このレンダラで使用できるかたちに変換
    for(size_t i = 0; i < model->materials.size(); i++)
    {
        std::cout << "Start to convert " << i << "th material" << std::endl;
        model->materials[i] = new Material();
        const auto pMaterial = pScene->mMaterials[i];
        parseMaterial(*model->materials[i], pMaterial, model->textures, fileName);
    }
    if (m_orca) {
        size_t textureBytes = 0;
        for (const Texture* texture : model->textures)
            textureBytes += static_cast<size_t>(texture->resolution.x) *
                static_cast<size_t>(texture->resolution.y) * sizeof(uint32_t);
        std::cout << "ORCA unique texture/color-role pairs: " << model->textures.size()
                  << ", full-resolution RGBA8 base-mip storage: "
                  << static_cast<double>(textureBytes) / (1024.0 * 1024.0) << " MiB" << std::endl;
    }

    // ------------------------
    // メッシュデータの読み込み
    // ------------------------

    std::cout << "Start to convert mesh data to use this renderer." << std::endl;
    
    // メッシュデータを，このレンダラで使用できるかたちに変換
    for(size_t i = 0; i < model->meshes.size(); i++)
    {
        model->meshes[i] = new TriangleMesh();
        std::cout << "Start to convert " << i << "th mesh" << std::endl;
        const auto pMesh = pScene->mMeshes[i];
        parseMesh(*model->meshes[i], pMesh, model->materials);
        
    }

    std::cout << "created a total of " << model->meshes.size() << "meshes" << std::endl;
    if (m_orca) {
        uint64_t triangles = 0;
        size_t twoSided = 0, emissive = 0;
        for (const TriangleMesh* mesh : model->meshes) triangles += mesh->index.size();
        for (const Material* material : model->materials) {
            twoSided += material->twoSided ? 1u : 0u;
            emissive += material->isLight ? 1u : 0u;
        }
        std::cout << "ORCA profile summary: meshes=" << model->meshes.size()
                  << ", triangles=" << triangles << ", materials=" << model->materials.size()
                  << ", twoSided=" << twoSided << ", emissive=" << emissive
                  << ", uniqueDecodedTextures=" << model->textures.size()
                  << ", metresPerUnit=" << m_metresPerUnit << std::endl;
    }

    // シーンを囲う Bounding Box を生成
    for(auto mesh : model->meshes){
        if (!mesh) {
            std::cerr << "WARNING: nullptr mesh" << std::endl;
            continue;
        }
        if (mesh->vertex.empty()) {
            std::cerr << "WARNING: empty vertex buffer in mesh" << std::endl;
            continue;
        }
        
        for (auto vtx: mesh->vertex){
            model->bounds.extend(vtx);
        }
    }

    aiMatrix4x4 rootT = pScene->mRootNode->mTransformation;
    std::cout << "Root transform: \n";
    std::cout << rootT.a1 << ", " << rootT.a2 << ", " << rootT.a3 << "\n";
    std::cout << rootT.b1 << ", " << rootT.b2 << ", " << rootT.b3 << "\n";
    std::cout << rootT.c1 << ", " << rootT.c2 << ", " << rootT.c3 << "\n";

    float3 bBoxCenter   = model->bounds.getCenter();
    float3 bBoxMin      = model->bounds.getMin();
    float3 bBoxSpan     = model->bounds.getSpan();
    std::cout << "Bounding box center:" << std::endl;
    std::cout << bBoxCenter.x << ", " << bBoxCenter.y << ", " << bBoxCenter.z << "\n";

    
    const float3 t = make_float3(objectDesc.TRS.transform[0], objectDesc.TRS.transform[1], objectDesc.TRS.transform[2]);
    const float4 q = make_float4(objectDesc.TRS.rotation[0],  objectDesc.TRS.rotation[1],  objectDesc.TRS.rotation[2], objectDesc.TRS.rotation[3]);
    const float3 s = make_float3(objectDesc.TRS.scale[0],     objectDesc.TRS.scale[1],     objectDesc.TRS.scale[2]    );
    model->modelMatrix = mymath::makeInstanceMatrix(
        t,
        q,
        s,
        bBoxCenter,
        bBoxMin,
        objectDesc.placeCenterBBoxAtOrigin,
        objectDesc.placeOnGround
    );
    
    pScene = nullptr;

    return true;
}

// Assimp の aiMesh フォーマットをパースし，このレンダラで使用する Mesh フォーマットに変換
void MeshLoader::parseMesh(TriangleMesh & dstMesh, const aiMesh* pSrcMesh, const std::vector<Material*>& materials)
{
    if (!pSrcMesh || pSrcMesh->mMaterialIndex >= materials.size() ||
        !materials[pSrcMesh->mMaterialIndex])
        throw std::runtime_error("Mesh has an invalid material index");
    int materialID = pSrcMesh->mMaterialIndex;
    dstMesh.materialID = materialID;

    const Material& material = *materials[materialID];

    const std::string meshLabel = "ORCA mesh '" + std::string(pSrcMesh->mName.C_Str()) +
        "' (material '" + material.name + "'): ";
    const auto meshError = [&](const std::string& error) {
        throw std::runtime_error(meshLabel + error);
    };
    if (m_orca) {
        if (!pSrcMesh->mNumVertices || !pSrcMesh->mVertices || !pSrcMesh->mNumFaces || !pSrcMesh->mFaces)
            meshError("empty geometry");
        if (!pSrcMesh->HasNormals()) meshError("authored vertex normals are missing");
        const auto requireUV = [&](int textureID, unsigned int channel, const char* role) {
            if (textureID < 0) return;
            if (channel >= AI_MAX_NUMBER_OF_TEXTURECOORDS || !pSrcMesh->HasTextureCoords(channel) ||
                pSrcMesh->mNumUVComponents[channel] != 2)
                meshError(std::string(role) + " requires a valid 2D UV channel " + std::to_string(channel));
        };
        requireUV(material.diffuseTextureID, material.diffuseUVIndex, "BaseColor");
        requireUV(material.normalTextureID, material.normalUVIndex, "Normal");
        requireUV(material.rmTextureID, material.rmUVIndex, "ORM");
        requireUV(material.emissiveTextureID, material.emissiveUVIndex, "Emissive");
        if (material.normalTextureID >= 0 && !pSrcMesh->HasTangentsAndBitangents())
            meshError("normal map has no valid tangent space");
    }
    
    aiVector3D zero3D(0.0f, 0.0f, 0.0f);
    
    // 頂点データの記録
    dstMesh.vertex.resize(pSrcMesh->mNumVertices);
    dstMesh.normal.resize(pSrcMesh->mNumVertices);
    dstMesh.tangent.resize(pSrcMesh->mNumVertices);
    dstMesh.diffuseTexcoord.resize(pSrcMesh->mNumVertices);
    dstMesh.normalTexcoord.resize(pSrcMesh->mNumVertices);
    dstMesh.emissiveTexcoord.resize(pSrcMesh->mNumVertices);
    
    std::cout << pSrcMesh->mNumVertices << std::endl;
    for(auto i = 0u; i < pSrcMesh->mNumVertices; i++)
    {
        auto pPosition  = &(pSrcMesh->mVertices[i]);

        auto pNormal    = (pSrcMesh->HasNormals()) ? &(pSrcMesh->mNormals[i]) : &zero3D;
        float3 normal = make_float3(pNormal->x, pNormal->y, pNormal->z);
        auto pBiTangent = (pSrcMesh->HasTangentsAndBitangents()) ? &(pSrcMesh->mBitangents[i]) : &zero3D;
        float3 biTangent = make_float3(pBiTangent->x, pBiTangent->y, pBiTangent->z);
        auto pTangent   = (pSrcMesh->HasTangentsAndBitangents()) ? &(pSrcMesh->mTangents[i])  : &zero3D;
        float3 tangent = make_float3(pTangent->x, pTangent->y, pTangent->z);
        float tanW = (dot(cross(normal, tangent), biTangent) < 0.0f) ? -1.0f : 1.0f;

        const aiVector3D* pDiffuseTexCoord   = (pSrcMesh->HasTextureCoords(material.diffuseUVIndex ))    ? &(pSrcMesh->mTextureCoords[material.diffuseUVIndex ][i]) : &zero3D;
        const aiVector3D* pNormalTexCoord    = (pSrcMesh->HasTextureCoords(material.normalUVIndex  ))    ? &(pSrcMesh->mTextureCoords[material.normalUVIndex  ][i]) : &zero3D;
        const aiVector3D* pEmissiveTexCoord  = (pSrcMesh->HasTextureCoords(material.emissiveUVIndex))    ? &(pSrcMesh->mTextureCoords[material.emissiveUVIndex][i]) : &zero3D;

        
        const float3 position = make_float3(
            static_cast<float>(static_cast<double>(pPosition->x) * m_metresPerUnit),
            static_cast<float>(static_cast<double>(pPosition->y) * m_metresPerUnit),
            static_cast<float>(static_cast<double>(pPosition->z) * m_metresPerUnit));
        if (m_orca) {
            const auto finite3 = [](const float3& v) {
                return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
            };
            if (!finite3(position)) meshError("non-finite vertex position after metre conversion");
            if (!finite3(normal) || !(dot(normal, normal) > 0.0f))
                meshError("non-finite or zero vertex normal");
            if (material.normalTextureID >= 0 &&
                (!finite3(tangent) || !finite3(biTangent) || !(dot(tangent, tangent) > 0.0f) ||
                 !(dot(biTangent, biTangent) > 0.0f)))
                meshError("non-finite or zero tangent/bitangent");
            const auto finiteUV = [&](int textureID, const aiVector3D& uv, const char* role) {
                if (textureID >= 0 && (!std::isfinite(uv.x) || !std::isfinite(uv.y)))
                    meshError(std::string(role) + " has a non-finite UV coordinate");
            };
            finiteUV(material.diffuseTextureID, *pDiffuseTexCoord, "BaseColor");
            finiteUV(material.normalTextureID, *pNormalTexCoord, "Normal");
            finiteUV(material.rmTextureID, *pDiffuseTexCoord, "ORM");
            finiteUV(material.emissiveTextureID, *pEmissiveTexCoord, "Emissive");
        }
        dstMesh.vertex[i]   = position;
        dstMesh.normal[i]   = normal;
        dstMesh.tangent[i]  = make_float4(tangent, tanW);
        dstMesh.diffuseTexcoord[i]  = make_float2(pDiffuseTexCoord->x, pDiffuseTexCoord->y);        
        dstMesh.normalTexcoord[i]   = make_float2(pNormalTexCoord->x, pNormalTexCoord->y);        
        dstMesh.emissiveTexcoord[i] = make_float2(pEmissiveTexCoord->x, pEmissiveTexCoord->y);
    }

    // 頂点インデックスの記録
    dstMesh.index.resize(pSrcMesh->mNumFaces);
    for(auto i = 0u; i < pSrcMesh->mNumFaces; i++)
    {
        const auto& face  = (pSrcMesh->mFaces[i]);
        if (face.mNumIndices != 3 || !face.mIndices ||
            face.mIndices[0] >= pSrcMesh->mNumVertices || face.mIndices[1] >= pSrcMesh->mNumVertices ||
            face.mIndices[2] >= pSrcMesh->mNumVertices)
            throw std::runtime_error("Mesh '" + std::string(pSrcMesh->mName.C_Str()) + "' has an invalid triangle");
        dstMesh.index[i]   = make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
    }

    dstMesh.hasTangentSpace = pSrcMesh->HasTangentsAndBitangents();
    dstMesh.hasNormal       = pSrcMesh->HasNormals();
    dstMesh.hasUVCoord      = pSrcMesh->HasTextureCoords(material.diffuseUVIndex);

}

int MeshLoader::loadOrcaTexture(const m2::OrcaTextureBinding& binding, bool colorTexture,
    const std::string& materialName, std::vector<Texture*>& textures, const std::string& fbxPath)
{
    if (!binding.present()) return -1;
    std::string relative = binding.path;
    std::replace(relative.begin(), relative.end(), '\\', '/');
    std::filesystem::path texturePath = std::filesystem::u8path(relative);
    if (!texturePath.is_absolute())
        texturePath = std::filesystem::u8path(fbxPath).parent_path() / texturePath;
    std::error_code error;
    if (!std::filesystem::is_regular_file(texturePath, error) || error)
        m2::orca_material_error(materialName, "referenced texture is missing or not a regular file: " + binding.path);
    texturePath = std::filesystem::canonical(texturePath, error);
    if (error)
        m2::orca_material_error(materialName, "cannot resolve texture: " + binding.path + ": " + error.message());
    const auto utf8 = texturePath.generic_u8string();
    const std::string textureName(utf8.begin(), utf8.end());
    std::string cacheKey = textureName;
#if defined(_WIN32)
    // The verified asset paths are ASCII; match the Windows filesystem's case
    // insensitivity as well as resolving separators, dot segments and symlinks.
    std::transform(cacheKey.begin(), cacheKey.end(), cacheKey.begin(), [](unsigned char c) {
        return char(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c);
    });
#endif
    cacheKey += colorTexture ? "|srgb" : "|linear";
    const auto cached = m_textureCache.find(cacheKey);
    if (cached != m_textureCache.end()) return cached->second;

    static thread_local ComInit com;
    LoadedTexture decoded;
    if (!textureLoader(textureName, decoded, DXGI_FORMAT_R8G8B8A8_UNORM))
        m2::orca_material_error(materialName, "cannot decode referenced texture: " + textureName);
    if (textures.size() >= static_cast<size_t>((std::numeric_limits<int>::max)()))
        throw std::length_error("Texture index exceeds the renderer's integer range");
    auto texture = std::make_unique<Texture>();
    texture->pixel = new uint32_t[decoded.pixels.size()];
    std::memcpy(texture->pixel, decoded.pixels.data(), decoded.pixels.size() * sizeof(uint32_t));
    texture->resolution = make_int2(static_cast<int>(decoded.width), static_cast<int>(decoded.height));
    texture->isDiffuseTexture = colorTexture;
    const int textureID = static_cast<int>(textures.size());
    textures.push_back(texture.get());
    texture.release();
    m_textureCache.emplace(std::move(cacheKey), textureID);
    return textureID;
}

void MeshLoader::parseOrcaMaterial(Material& dstMaterial, const aiMaterial& source,
    std::vector<Texture*>& textures, const std::string& fbxPath)
{
    const m2::OrcaMaterial material = m2::read_orca_material(source);
    dstMaterial.name = material.name;
    dstMaterial.isOrca = true;
    dstMaterial.emissionOnly = false;
    dstMaterial.isGlass = false;
    dstMaterial.twoSided = material.twoSided;
    dstMaterial.opacity = material.opacity;
    dstMaterial.diffuse = make_float3(material.baseColor[0], material.baseColor[1], material.baseColor[2]);
    dstMaterial.roughness = material.roughness;
    dstMaterial.metallic = material.metallic;
    dstMaterial.emissive = make_float3(material.emission[0], material.emission[1], material.emission[2]);
    dstMaterial.isLight = material.emission[0] > 0 || material.emission[1] > 0 || material.emission[2] > 0;
    dstMaterial.diffuseUVIndex = material.baseColorTexture.uvIndex;
    dstMaterial.normalUVIndex = material.normalTexture.uvIndex;
    dstMaterial.rmUVIndex = material.ormTexture.uvIndex;
    dstMaterial.emissiveUVIndex = material.emissiveTexture.uvIndex;
    dstMaterial.diffuseTextureID = loadOrcaTexture(material.baseColorTexture, true, material.name, textures, fbxPath);
    dstMaterial.normalTextureID = loadOrcaTexture(material.normalTexture, false, material.name, textures, fbxPath);
    dstMaterial.rmTextureID = loadOrcaTexture(material.ormTexture, false, material.name, textures, fbxPath);
    dstMaterial.emissiveTextureID = loadOrcaTexture(material.emissiveTexture, true, material.name, textures, fbxPath);
}

// Assimp の aiMesh フォーマットをパースし，このレンダラで使用する Mesh フォーマットに変換
void MeshLoader::parseMaterial(Material& dstMaterial, const aiMaterial* pSrcMaterial, std::vector<Texture*>& textures, const std::string &fbxPath)
{
    if (!pSrcMaterial) throw std::runtime_error("Assimp returned a null material");
    if (m_orca) {
        parseOrcaMaterial(dstMaterial, *pSrcMaterial, textures, fbxPath);
        return;
    }
    int shading = 0;
    pSrcMaterial->Get(AI_MATKEY_SHADING_MODEL, shading);

    // Diffuse 成分
    aiColor3D color(0.0f, 0.0f, 0.0f);

    if(pSrcMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
    {
        dstMaterial.diffuse = make_float3(color.r, color.g, color.b);
    } else {
        dstMaterial.diffuse = make_float3(0.5f);
    }

    // Specular 成分
    if(pSrcMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
    {
        dstMaterial.specular = make_float3(color.r, color.g, color.b);
    } else {
        dstMaterial.specular = make_float3(0.0f);
    }

    // Emissive 成分
    if(pSrcMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS)
    {
        dstMaterial.emissive    = make_float3(color.r, color.g, color.b);
        if(color.r > 0.0f || color.g > 0.0f || color.b > 0.0f ){
            dstMaterial.isLight     = true;
        }
    } else {
        dstMaterial.emissive    = make_float3(0.0f);
    }

    // Glossiness
    ai_real glossiness = -1.0f;
    if (pSrcMaterial->Get(AI_MATKEY_GLOSSINESS_FACTOR, glossiness) == AI_SUCCESS) {
        std::cout << "Glossiness found: " << glossiness << std::endl;
    } 


    // Glass かどうかを推測
    aiString materialName;
    pSrcMaterial->Get(AI_MATKEY_NAME, materialName);
    std::string nameStr(materialName.C_Str());
    dstMaterial.name = nameStr;

    if((nameStr.find("Glass") != std::string::npos) || (nameStr.find("glass") != std::string::npos) || (nameStr.find("water") != std::string::npos) || (shading == aiShadingMode_Fresnel))
    {
        std::cout << "This material may be glass:" << nameStr <<"Raised a glass flag" << std::endl;
        dstMaterial.isGlass = true;
    }

    // 金属かどうかを推測
    bool isMetal = false;
    pSrcMaterial->Get(AI_MATKEY_NAME, materialName);
    nameStr = materialName.C_Str();

    if((nameStr.find("Metal") != std::string::npos || nameStr.find("metal") != std::string::npos))
    {
        std::cout << "This material may be metal:" << nameStr <<"Raised a metal flag" << std::endl;
        isMetal = true;
    }



    // テクスチャ
    // Diffuse (or Base Color) texture
    {
        static thread_local ComInit com;
        aiString path;
        unsigned int uvIndex = 0;

        if(pSrcMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
        {
            dstMaterial.diffuseUVIndex = uvIndex;
            std::filesystem::path diffuseTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
            std::cout << "Found diffuse texture: " << std::string(path.C_Str()) << std::endl;
            std::cerr << "Diffuse Texture of " <<  (path.C_Str()) << " uses " << uvIndex << " th texture" << std::endl;

            LoadedTexture tmpTextureDst;
            if(textureLoader(diffuseTexturePath.string(), tmpTextureDst, DXGI_FORMAT_R8G8B8A8_UNORM)){
                Texture* diffuseColorTexture = new Texture();
                diffuseColorTexture->pixel = new uint32_t[tmpTextureDst.pixels.size()];
                std::memcpy(diffuseColorTexture->pixel, tmpTextureDst.pixels.data(), tmpTextureDst.pixels.size() * sizeof(uint32_t));
                dstMaterial.diffuseTextureID = textures.size();
                diffuseColorTexture->resolution = make_int2(tmpTextureDst.width, tmpTextureDst.height);
                diffuseColorTexture->isDiffuseTexture = true;
                textures.push_back(diffuseColorTexture);
            }
            
        } else if (pSrcMaterial->GetTexture(aiTextureType_BASE_COLOR , 0, &path, nullptr, &uvIndex) == AI_SUCCESS){
            dstMaterial.diffuseUVIndex = uvIndex;
            std::filesystem::path diffuseTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
            std::cout << "Found baseColor texture: " << std::string(path.C_Str()) << std::endl;
            std::cerr << "Base color Texture of " <<  (path.C_Str()) << "uses " << uvIndex << " th texture" << std::endl;

            LoadedTexture tmpTextureDst;
            if(textureLoader(diffuseTexturePath.string(), tmpTextureDst, DXGI_FORMAT_R8G8B8A8_UNORM)){
                Texture* diffuseColorTexture = new Texture();
                diffuseColorTexture->pixel = new uint32_t[tmpTextureDst.pixels.size()];
                std::memcpy(diffuseColorTexture->pixel, tmpTextureDst.pixels.data(), tmpTextureDst.pixels.size() * sizeof(uint32_t));
                dstMaterial.diffuseTextureID = textures.size();
                diffuseColorTexture->resolution = make_int2(tmpTextureDst.width, tmpTextureDst.height);
                diffuseColorTexture->isDiffuseTexture = true;
                textures.push_back(diffuseColorTexture);
            }
        } 
    }

    // Normal texture
    {
        aiString path;
        unsigned int uvIndex = 0;

        if(pSrcMaterial->GetTexture(aiTextureType_NORMALS, 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
        {
            std::filesystem::path normalTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
            std::cout << "Normal Texture of " <<  (path.C_Str()) << "uses " << uvIndex << " th texture" << std::endl;
            std::cerr << std::string(path.C_Str()) << std::endl;
            dstMaterial.normalUVIndex = uvIndex;
            LoadedTexture tmpTextureDst;
            if(textureLoader(normalTexturePath.string(), tmpTextureDst, DXGI_FORMAT_R8G8B8A8_UNORM)){
                Texture* normalTexture = new Texture();
                normalTexture->pixel = new uint32_t[tmpTextureDst.pixels.size()];
                std::memcpy(normalTexture->pixel, tmpTextureDst.pixels.data(), tmpTextureDst.pixels.size() * sizeof(uint32_t));
                dstMaterial.normalTextureID = textures.size();
                normalTexture->resolution = make_int2(tmpTextureDst.width, tmpTextureDst.height);
                textures.push_back(normalTexture);
            }
        }
    }
    
    // Emissive texture
    {
        unsigned int uvIndex = 0;
        aiString path;
        if(pSrcMaterial->GetTexture(aiTextureType_EMISSIVE, 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
        {
            dstMaterial.emissiveUVIndex = uvIndex;
            std::cout << "Emissive Texture of " <<  (path.C_Str()) << "uses " << uvIndex << " th texture" << std::endl;
            std::filesystem::path emissiveTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
            std::cerr << std::string(path.C_Str()) << std::endl;

            LoadedTexture tmpTextureDst;
            if(textureLoader(emissiveTexturePath.string(), tmpTextureDst, DXGI_FORMAT_R8G8B8A8_UNORM)){
                Texture* emissiveTexture = new Texture();
                emissiveTexture->pixel = new uint32_t[tmpTextureDst.pixels.size()];
                std::memcpy(emissiveTexture->pixel, tmpTextureDst.pixels.data(), tmpTextureDst.pixels.size() * sizeof(uint32_t));
                dstMaterial.emissiveTextureID = textures.size();
                emissiveTexture->resolution = make_int2(tmpTextureDst.width, tmpTextureDst.height);
                // RGB colour textures are encoded sRGB; CUDA decodes them once.
                // Alpha remains linear. This flag is historical (also emission).
                emissiveTexture->isDiffuseTexture = true;
                textures.push_back(emissiveTexture);
                dstMaterial.isLight = true;
            }
        } 
    }

    // PBR のためのテクスチャ (roughness + metallic)
    // PBR 形式でテクスチャが保存されてない場合があるので，そのときは Spacular から作る
    // bool hasMetallicTexture     = false;
    // bool hasRoughnessTexture    = false;
    // bool hasSpecularTexture     = false;

    // LoadedTexture tmpTextureRoughness;
    // LoadedTexture tmpTextureMetallic;
    // LoadedTexture tmpTextureSpecular;
    
    // // roughness
    // {
    //     unsigned int uvIndex = 0;
    //     aiString path;
    //     if(pSrcMaterial->GetTexture(aiTextureType_GLTF_METALLIC_ROUGHNESS, 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
    //     {
    //         hasRoughnessTexture = true;
    //         dstMaterial.rmUVIndex = uvIndex;
    //         std::filesystem::path roughnessTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
    //         std::cerr << std::string(path.C_Str()) << std::endl;

    //         if(textureLoader(roughnessTexturePath.string(), tmpTextureRoughness, DXGI_FORMAT_R8G8B8A8_UNORM)){
    //             hasRoughnessTexture = true;
    //         }
    //     }
    // }

    // // metallic
    // {
    //     unsigned int uvIndex = 0;
    //     aiString path;
    //     if(pSrcMaterial->GetTexture(aiTextureType_METALNESS , 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
    //     {
    //         dstMaterial.rmUVIndex = uvIndex;
    //         hasMetallicTexture = true;

    //         std::filesystem::path metallicTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
    //         std::cerr << std::string(path.C_Str()) << std::endl;

    //         if(textureLoader(metallicTexturePath.string(), tmpTextureMetallic, DXGI_FORMAT_R8G8B8A8_UNORM)){
    //             hasMetallicTexture = true;
    //         }
    //     }
    // }

    // Specular texture
    {
        // unsigned int uvIndex = 0;
        // aiString path;
        // if(pSrcMaterial->GetTexture(aiTextureType_SPECULAR, 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
        // {
        //     dstMaterial.rmUVIndex = uvIndex;
        //     hasSpecularTexture = true;

        //     std::filesystem::path specularTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
        //     std::cerr << std::string(path.C_Str()) << std::endl;
            
        //     if(textureLoader(specularTexturePath.string(), tmpTextureSpecular, DXGI_FORMAT_R8G8B8A8_UNORM)){
        //         hasSpecularTexture = true;
        //     }
        // }
        
        unsigned int uvIndex = 0;
        aiString path;
        if(pSrcMaterial->GetTexture(aiTextureType_SPECULAR, 0, &path, nullptr, &uvIndex) == AI_SUCCESS)
        {
            dstMaterial.rmUVIndex = uvIndex;
            std::cout << "Specular Texture of " <<  (path.C_Str()) << "uses " << uvIndex << " th texture" << std::endl;
            std::filesystem::path specularTexturePath = std::filesystem::path(fbxPath).parent_path() / std::string(path.C_Str());
            std::cerr << std::string(path.C_Str()) << std::endl;

            LoadedTexture tmpTextureDst;
            if(textureLoader(specularTexturePath.string(), tmpTextureDst, DXGI_FORMAT_R8G8B8A8_UNORM)){
                Texture* rmTexture = new Texture();
                rmTexture->pixel = new uint32_t[tmpTextureDst.pixels.size()];
                std::memcpy(rmTexture->pixel, tmpTextureDst.pixels.data(), tmpTextureDst.pixels.size() * sizeof(uint32_t));
                dstMaterial.rmTextureID = textures.size();
                rmTexture->resolution = make_int2(tmpTextureDst.width, tmpTextureDst.height);
                textures.push_back(rmTexture);
            }
        } 
    }


    // if(hasRoughnessTexture || hasMetallicTexture) { // そのまま使用
    //     // MEMO: 後で正しく動くようにする
    //     Texture* roughnessMetallicTexture = new Texture();
    //     roughnessMetallicTexture->pixel = new uint32_t[tmpTextureRoughness.width * tmpTextureRoughness.height];
    //     std::memcpy(roughnessMetallicTexture->pixel, tmpTextureRoughness.pixels.data(), tmpTextureRoughness.width * tmpTextureRoughness.height * sizeof(uint32_t));
        
    //     dstMaterial.rmTextureID = textures.size();
    //     roughnessMetallicTexture->resolution = make_int2(tmpTextureRoughness.width, tmpTextureRoughness.height);
    //     textures.push_back(roughnessMetallicTexture);
    // } else if(hasSpecularTexture){ // 変換して使用
    //     Texture* roughnessMetallicTexture = new Texture();
    //     LoadedTexture tmpTextureDst;
    //     convertTextureSpecularToRoughnessMetallic(tmpTextureSpecular, isMetal, glossiness, tmpTextureDst);
    //     roughnessMetallicTexture->pixel = new uint32_t[tmpTextureDst.width * tmpTextureDst.height];
    //     std::memcpy(roughnessMetallicTexture->pixel, tmpTextureDst.pixels.data(), tmpTextureDst.width * tmpTextureDst.height * sizeof(uint32_t));
    //     dstMaterial.rmTextureID = textures.size();
    //     roughnessMetallicTexture->resolution = make_int2(tmpTextureDst.width, tmpTextureDst.height);
    //     textures.push_back(roughnessMetallicTexture);
    // }

}                    

Model *loadModel(const std::string &modelFileName, const sceneIO::Object objectDesc){
    auto model = std::make_unique<Model>();
    MeshLoader loader;

    std::filesystem::path exePath;
#if defined(_WIN32)
    char pathBuffer[MAX_PATH] = {};
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    char pathBuffer[PATH_MAX] = {};
    const ssize_t count = readlink("/proc/self/exe", pathBuffer, sizeof(pathBuffer)-1);
    if(count < 0) throw std::runtime_error("readlink(/proc/self/exe) failed");
    pathBuffer[count] = '\0';
    exePath = std::filesystem::path(pathBuffer);
#endif

    const std::filesystem::path requestedPath = objectDesc.materialProfile == "orca_metal_rough"
        ? std::filesystem::u8path(modelFileName) : std::filesystem::path(modelFileName);
    std::filesystem::path fbxDir = exePath.parent_path().parent_path().parent_path().parent_path() / "model/mesh" / requestedPath;
    if (requestedPath.is_absolute() || std::filesystem::exists(requestedPath))
        fbxDir = std::filesystem::absolute(requestedPath);
    else if (std::filesystem::exists(std::filesystem::path("model/mesh") / requestedPath))
        fbxDir = std::filesystem::absolute(std::filesystem::path("model/mesh") / requestedPath);
    std::cout << "Model Path:" << fbxDir << std::endl;

    std::string err = "";
    // M2 texture decoding expects UTF-8 paths (including a non-ASCII asset root).
    // Keep the historical path encoding for the legacy branch.
    const auto utf8Path = fbxDir.u8string();
    const std::string inputPath = objectDesc.materialProfile == "orca_metal_rough"
        ? std::string(utf8Path.begin(), utf8Path.end()) : fbxDir.string();
    bool readOk = loader.load(inputPath, model.get(), objectDesc);

    if(!readOk) {
        throw std::runtime_error("Could not read model from " + fbxDir.string() + " : " + err);
    }

    return model.release();

}
