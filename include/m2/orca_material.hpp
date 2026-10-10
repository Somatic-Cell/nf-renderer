#pragma once

// ORCA Emerald Square v4.1's documented metal/rough import profile. This file
// depends only on Assimp and the standard library so the actual material
// interpretation can also be exercised by an asset audit without a GPU.
#include <assimp/material.h>
#include <assimp/mesh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace m2 {

struct OrcaTextureBinding {
    std::string path;
    unsigned int uvIndex = 0;
    bool present() const { return !path.empty(); }
};

struct OrcaMaterial {
    std::string name;
    std::array<float, 3> baseColor{1.0f, 1.0f, 1.0f};
    std::array<float, 3> emission{0.0f, 0.0f, 0.0f};
    float roughness = 0.0f;
    float metallic = 0.0f;
    float opacity = 1.0f;
    bool twoSided = false;
    OrcaTextureBinding baseColorTexture;
    OrcaTextureBinding normalTexture;
    OrcaTextureBinding ormTexture;
    OrcaTextureBinding emissiveTexture;
};

[[noreturn]] inline void orca_material_error(const std::string& name, const std::string& what)
{
    throw std::runtime_error("ORCA material '" + name + "': " + what);
}

inline bool orca_double_sided_tag(const std::string& name)
{
    // The .DoubleSided tag is part of Falcor's documented import convention.
    // The ORCA exporter also appends numeric duplicate suffixes (.DoubleSided1
    // through .DoubleSided4 in the verified Day FBX). Recognize those explicitly;
    // material names such as Glass/Metal are never used to infer a BSDF.
    size_t start = name.find('.');
    while (start != std::string::npos) {
        const size_t end = name.find('.', start + 1);
        std::string token = name.substr(start + 1, end - start - 1);
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
            return char(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c);
        });
        constexpr const char* tag = "doublesided";
        constexpr size_t length = 11;
        if (token.compare(0, length, tag) == 0 &&
            std::all_of(token.begin() + length, token.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            return true;
        }
        start = end;
    }
    return false;
}

inline OrcaTextureBinding orca_texture_binding(const aiMaterial& material,
    aiTextureType type, const std::string& name, const char* role)
{
    const unsigned int count = material.GetTextureCount(type);
    if (count == 0) return {};
    if (count != 1) orca_material_error(name, std::string(role) + " has multiple texture layers");

    aiString path;
    aiTextureMapping mapping = aiTextureMapping_UV;
    unsigned int uvIndex = 0;
    ai_real blend = ai_real(1);
    aiTextureOp operation = aiTextureOp_Multiply;
    aiTextureMapMode modes[3] = {aiTextureMapMode_Wrap, aiTextureMapMode_Wrap, aiTextureMapMode_Wrap};
    if (material.GetTexture(type, 0, &path, &mapping, &uvIndex, &blend, &operation, modes) != AI_SUCCESS)
        orca_material_error(name, std::string("cannot read ") + role + " texture binding");
    if (path.length == 0 || path.C_Str()[0] == '*')
        orca_material_error(name, std::string(role) + " needs a nonempty external texture path");
    if (mapping != aiTextureMapping_UV || uvIndex >= AI_MAX_NUMBER_OF_TEXTURECOORDS)
        orca_material_error(name, std::string(role) + " has unsupported texture mapping or UV index");
    if (blend != ai_real(1) || operation != aiTextureOp_Multiply)
        orca_material_error(name, std::string(role) + " uses an unsupported texture blend operation");
    if (modes[0] != aiTextureMapMode_Wrap || modes[1] != aiTextureMapMode_Wrap)
        orca_material_error(name, std::string(role) + " requires a sampler other than wrap");

    aiUVTransform transform;
    if (material.Get(AI_MATKEY_UVTRANSFORM(type, 0), transform) == AI_SUCCESS &&
        (transform.mTranslation.x != 0 || transform.mTranslation.y != 0 ||
         transform.mScaling.x != 1 || transform.mScaling.y != 1 || transform.mRotation != 0)) {
        orca_material_error(name, std::string(role) + " has a non-identity UV transform");
    }
    unsigned int textureFlags = 0;
    if (material.Get(AI_MATKEY_TEXFLAGS(type, 0), textureFlags) == AI_SUCCESS && textureFlags != 0)
        orca_material_error(name, std::string(role) + " has unsupported texture flags");
    return {path.C_Str(), uvIndex};
}

inline OrcaMaterial read_orca_material(const aiMaterial& material)
{
    OrcaMaterial result;
    aiString name;
    material.Get(AI_MATKEY_NAME, name);
    result.name = name.C_Str();
    if (result.name.empty()) result.name = "<unnamed>";

    // Do not quietly discard maps which belong to a different material model.
    for (unsigned int i = 0; i < material.mNumProperties; ++i) {
        const aiMaterialProperty& property = *material.mProperties[i];
        if (std::string(property.mKey.C_Str()) != "$tex.file") continue;
        const auto type = static_cast<aiTextureType>(property.mSemantic);
        if (type != aiTextureType_DIFFUSE && type != aiTextureType_BASE_COLOR &&
            type != aiTextureType_NORMALS && type != aiTextureType_SPECULAR && type != aiTextureType_EMISSIVE) {
            orca_material_error(result.name, "unsupported texture semantic " + std::to_string(property.mSemantic));
        }
    }

    result.baseColorTexture = orca_texture_binding(material, aiTextureType_DIFFUSE, result.name, "BaseColor");
    const auto alternateBase = orca_texture_binding(material, aiTextureType_BASE_COLOR, result.name, "BaseColor");
    if (result.baseColorTexture.present() && alternateBase.present() &&
        (result.baseColorTexture.path != alternateBase.path || result.baseColorTexture.uvIndex != alternateBase.uvIndex)) {
        orca_material_error(result.name, "DIFFUSE and BASE_COLOR identify different textures");
    }
    if (!result.baseColorTexture.present()) result.baseColorTexture = alternateBase;
    result.normalTexture = orca_texture_binding(material, aiTextureType_NORMALS, result.name, "Normal");
    result.ormTexture = orca_texture_binding(material, aiTextureType_SPECULAR, result.name, "ORM");
    result.emissiveTexture = orca_texture_binding(material, aiTextureType_EMISSIVE, result.name, "Emissive");

    aiColor3D color;
    if (material.Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
        result.baseColor = {color.r, color.g, color.b};
    if (material.Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
        // Falcor's metal/rough scalar specular parameters have the same G/B
        // layout as the documented ORM texture. No Phong-to-PBR guess is used.
        result.roughness = color.g;
        result.metallic = color.b;
    }
    if (material.Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS)
        result.emission = {color.r, color.g, color.b};
    material.Get(AI_MATKEY_OPACITY, result.opacity);

    int doubleSided = 0;
    material.Get(AI_MATKEY_TWOSIDED, doubleSided);
    result.twoSided = doubleSided != 0 || orca_double_sided_tag(result.name);

    for (float c : result.baseColor)
        if (!std::isfinite(c) || c < 0 || c > 1) orca_material_error(result.name, "BaseColor is outside linear RGB [0,1]");
    for (float c : result.emission)
        if (!std::isfinite(c) || c < 0) orca_material_error(result.name, "emission must be finite and nonnegative");
    if (!std::isfinite(result.roughness) || result.roughness < 0 || result.roughness > 1 ||
        !std::isfinite(result.metallic) || result.metallic < 0 || result.metallic > 1)
        orca_material_error(result.name, "roughness/metallic scalar is outside [0,1]");
    if (!std::isfinite(result.opacity) || result.opacity < 0 || result.opacity > 1)
        orca_material_error(result.name, "scalar opacity is outside [0,1]");
    if (result.opacity != 1.0f)
        orca_material_error(result.name, "scalar-opacity specular transmission is not supported by this metal/rough profile");

    // Falcor 4.1's TextureSampler returns texels instead of multiplying the RGB
    // fallback color. EmissiveFactor is a separate scalar (1 for the Day scene).
    // Keep the existing GPU coefficient*texture interface by supplying unit RGB.
    if (result.emissiveTexture.present()) result.emission = {1.0f, 1.0f, 1.0f};
    const unsigned int surfaceUV = result.baseColorTexture.present() ? result.baseColorTexture.uvIndex : 0;
    if (result.ormTexture.present() && result.ormTexture.uvIndex != surfaceUV)
        orca_material_error(result.name, "ORM texture uses a different UV set from BaseColor");
    if (result.normalTexture.present() && result.normalTexture.uvIndex != 0)
        orca_material_error(result.name, "normal map needs tangents generated from UV0");
    return result;
}

} // namespace m2
