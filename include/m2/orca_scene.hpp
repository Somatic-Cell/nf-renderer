#pragma once

#include <assimp/Importer.hpp>
#include <assimp/config.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace m2 {

inline constexpr unsigned int orca_import_flags = aiProcess_Triangulate |
    aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_OptimizeMeshes;

inline void configure_orca_importer(Assimp::Importer& importer)
{
    // Keep the file's axes and units until we can validate their metadata. This
    // avoids depending on the pinned FBX converter's AI_FLOAT / Get<double>
    // mismatch for UnitScaleFactor, or on a later Assimp version fixing it.
    importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_IGNORE_UP_DIRECTION, true);
    importer.SetPropertyBool(AI_CONFIG_FBX_CONVERT_TO_M, false);
}

inline double orca_metres_per_unit(const aiScene& scene)
{
    if (!scene.mMetaData)
        throw std::runtime_error("ORCA FBX has no axis/unit metadata");
    const auto& metadata = *scene.mMetaData;
    const auto require_axis = [&](const char* key, int32_t expected) {
        int32_t value = -1;
        if (!metadata.Get(std::string(key), value) || value != expected)
            throw std::runtime_error(std::string("ORCA FBX requires ") + key + "=" + std::to_string(expected));
    };
    require_axis("CoordAxis", 0);
    require_axis("CoordAxisSign", 1);
    require_axis("UpAxis", 1);
    require_axis("UpAxisSign", 1);
    require_axis("FrontAxis", 2);
    require_axis("FrontAxisSign", 1);

    double centimetresPerUnit = 0.0;
    float floatUnits = 0.0f;
    if (metadata.Get(std::string("UnitScaleFactor"), floatUnits))
        centimetresPerUnit = static_cast<double>(floatUnits);
    else if (!metadata.Get(std::string("UnitScaleFactor"), centimetresPerUnit))
        throw std::runtime_error("ORCA FBX UnitScaleFactor must be a float or double");
    const double metres = centimetresPerUnit / 100.0;
    if (!std::isfinite(metres) || metres <= 0 ||
        metres > static_cast<double>((std::numeric_limits<float>::max)()) ||
        static_cast<float>(metres) == 0.0f)
        throw std::runtime_error("ORCA FBX UnitScaleFactor must define a positive, finite metre scale");
    return metres;
}

} // namespace m2
