#include "m2/orca_material.hpp"
#include "m2/orca_scene.hpp"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <set>

using json = nlohmann::ordered_json;

static json binding(const m2::OrcaTextureBinding& b) {
    return {{"path", b.path}, {"uv", b.uvIndex}};
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    Assimp::Importer importer;
    m2::configure_orca_importer(importer);
    constexpr unsigned flags = m2::orca_import_flags;
    const aiScene* scene = importer.ReadFile(argv[1], flags);
    if (!scene) { std::cerr << importer.GetErrorString() << '\n'; return 1; }
    json out = {{"flags", flags}, {"meshes", scene->mNumMeshes}, {"materials", scene->mNumMaterials}};
    const double metresPerUnit = m2::orca_metres_per_unit(*scene);
    out["fbx_ignore_up_direction"] = true;
    out["fbx_convert_to_meters"] = false;
    out["metres_per_unit"] = metresPerUnit;
    out["material_records"] = json::array();
    out["errors"] = json::array();
    std::set<std::string> paths;
    unsigned twoSided = 0, emissive = 0, bindingCount = 0;
    std::vector<m2::OrcaMaterial> materials(scene->mNumMaterials);
    for (unsigned i = 0; i < scene->mNumMaterials; ++i) {
        try {
            const auto m = m2::read_orca_material(*scene->mMaterials[i]);
            materials[i] = m;
            twoSided += m.twoSided;
            emissive += m.emissiveTexture.present() || m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0;
            for (const auto* b : {&m.baseColorTexture, &m.normalTexture, &m.ormTexture, &m.emissiveTexture}) {
                if (b->present()) { paths.insert(b->path); ++bindingCount; }
            }
            out["material_records"].push_back({{"index", i}, {"name", m.name}, {"base_color", m.baseColor},
                {"emission", m.emission}, {"roughness", m.roughness}, {"metallic", m.metallic},
                {"opacity", m.opacity}, {"two_sided", m.twoSided}, {"base_color_texture", binding(m.baseColorTexture)},
                {"normal_texture", binding(m.normalTexture)}, {"orm_texture", binding(m.ormTexture)},
                {"emissive_texture", binding(m.emissiveTexture)}});
        } catch (const std::exception& e) {
            out["errors"].push_back({{"material", i}, {"error", e.what()}});
        }
    }
    unsigned missingUv = 0, invalidUv = 0, invalidMaterialIndex = 0;
    uint64_t triangles = 0, vertices = 0, emissiveTriangles = 0, degenerateTriangles = 0, degenerateEmissive = 0;
    uint64_t invalidPositions = 0, invalidNormals = 0, invalidTangents = 0;
    std::array<double, 3> low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
    double emittingArea = 0;
    const auto finite3 = [](aiVector3D v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };
    for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh& mesh = *scene->mMeshes[i];
        if (mesh.mMaterialIndex >= materials.size()) { ++invalidMaterialIndex; continue; }
        const auto& m = materials[mesh.mMaterialIndex];
        const bool emits = m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0;
        triangles += mesh.mNumFaces;
        vertices += mesh.mNumVertices;
        if (emits) emissiveTriangles += mesh.mNumFaces;
        for (unsigned j = 0; j < mesh.mNumVertices; ++j) {
            const auto v = mesh.mVertices[j];
            if (!finite3(v)) ++invalidPositions;
            for (unsigned c = 0; c < 3; ++c) {
                low[c] = std::min(low[c], v[c] * metresPerUnit);
                high[c] = std::max(high[c], v[c] * metresPerUnit);
            }
            if (!mesh.HasNormals() || !finite3(mesh.mNormals[j]) || !(mesh.mNormals[j].SquareLength() > 0)) ++invalidNormals;
            if (!mesh.HasTangentsAndBitangents() || !finite3(mesh.mTangents[j]) || !finite3(mesh.mBitangents[j]) ||
                !(mesh.mTangents[j].SquareLength() > 0) || !(mesh.mBitangents[j].SquareLength() > 0)) ++invalidTangents;
        }
        for (unsigned j = 0; j < mesh.mNumFaces; ++j) {
            const auto& f = mesh.mFaces[j];
            if (f.mNumIndices != 3 || f.mIndices[0] >= mesh.mNumVertices || f.mIndices[1] >= mesh.mNumVertices || f.mIndices[2] >= mesh.mNumVertices) {
                out["errors"].push_back({{"mesh", i}, {"error", "invalid triangle"}}); continue;
            }
            const auto e1 = mesh.mVertices[f.mIndices[1]] - mesh.mVertices[f.mIndices[0]];
            const auto e2 = mesh.mVertices[f.mIndices[2]] - mesh.mVertices[f.mIndices[0]];
            const auto cross = e1 ^ e2;
            const double area = 0.5 * cross.Length() * metresPerUnit * metresPerUnit;
            if (!(area > 0)) { ++degenerateTriangles; if (emits) ++degenerateEmissive; }
            if (emits) emittingArea += area;
        }
        for (const auto* b : {&m.baseColorTexture, &m.normalTexture, &m.ormTexture, &m.emissiveTexture}) {
            if (!b->present()) continue;
            if (!mesh.HasTextureCoords(b->uvIndex) || mesh.mNumUVComponents[b->uvIndex] != 2) { ++missingUv; continue; }
            for (unsigned j = 0; j < mesh.mNumVertices; ++j) {
                const auto& uv = mesh.mTextureCoords[b->uvIndex][j];
                if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) ++invalidUv;
            }
        }
    }
    out["two_sided_materials"] = twoSided;
    out["emissive_materials"] = emissive;
    out["texture_bindings"] = bindingCount;
    out["unique_texture_paths"] = paths.size();
    out["missing_required_uv_bindings"] = missingUv;
    out["nonfinite_required_uvs"] = invalidUv;
    out["invalid_material_indices"] = invalidMaterialIndex;
    out["vertices"] = vertices;
    out["triangles"] = triangles;
    out["emissive_triangles"] = emissiveTriangles;
    out["emissive_geometric_area_m2"] = emittingArea;
    out["degenerate_triangles"] = degenerateTriangles;
    out["degenerate_emissive_triangles"] = degenerateEmissive;
    out["invalid_positions"] = invalidPositions;
    out["invalid_normals"] = invalidNormals;
    out["invalid_tangents"] = invalidTangents;
    out["bounds_min_m"] = low;
    out["bounds_max_m"] = high;
    out["passed"] = out["errors"].empty() && !missingUv && !invalidUv && !invalidMaterialIndex &&
        !invalidPositions && !invalidNormals && !invalidTangents && !degenerateEmissive;
    std::ofstream(argv[2]) << out.dump(2) << '\n';
    out.erase("material_records");
    std::cout << out.dump(2) << '\n';
    return out["passed"].get<bool>() ? 0 : 1;
}
