#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <nlohmann/json.hpp>
#include "m2/orca_scene.hpp"
#include "m2/orca_material.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using json = nlohmann::ordered_json;
static json vec(aiVector3D v) { return {v.x,v.y,v.z}; }
static bool finite(aiVector3D v) { return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z); }
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    Assimp::Importer importer;
    m2::configure_orca_importer(importer);
    constexpr unsigned flags=m2::orca_import_flags;
    const aiScene* scene=importer.ReadFile(argv[1],flags);
    if(!scene){std::cerr<<importer.GetErrorString()<<'\n';return 1;}
    const double metres=m2::orca_metres_per_unit(*scene);
    const auto to_metres=[&](aiVector3D v){return aiVector3D(
        static_cast<float>(static_cast<double>(v.x)*metres),
        static_cast<float>(static_cast<double>(v.y)*metres),
        static_cast<float>(static_cast<double>(v.z)*metres));};
    json output; output["flags"]=flags;
    output["ignore_up_direction"]=true;output["convert_to_meters_assimp"]=false;
    output["explicit_metres_per_authored_unit"]=metres;
    output["meshes"]=scene->mNumMeshes;output["materials"]=scene->mNumMaterials;
    aiVector3D low(std::numeric_limits<float>::infinity());
    aiVector3D high(-std::numeric_limits<float>::infinity());
    std::uint64_t triangles=0,vertices=0,degenerate=0,bad_vertices=0,bad_normals=0,bad_tangents=0;
    std::uint64_t emissive_triangles=0,degenerate_emissive=0;
    json bad_meshes=json::array();
    for(unsigned mi=0;mi<scene->mNumMeshes;++mi) {
        const auto& mesh=*scene->mMeshes[mi];
        const auto material=m2::read_orca_material(*scene->mMaterials[mesh.mMaterialIndex]);
        const bool emissive=material.emission[0]>0||material.emission[1]>0||material.emission[2]>0;
        vertices+=mesh.mNumVertices;triangles+=mesh.mNumFaces;
        if(emissive)emissive_triangles+=mesh.mNumFaces;
        std::uint64_t local_degenerate=0;
        for(unsigned i=0;i<mesh.mNumVertices;++i) {
            const auto v=to_metres(mesh.mVertices[i]);
            if(!finite(v))++bad_vertices;
            for(unsigned k=0;k<3;++k){low[k]=std::min(low[k],v[k]);high[k]=std::max(high[k],v[k]);}
            if(!mesh.HasNormals()||!finite(mesh.mNormals[i])||!(mesh.mNormals[i].SquareLength()>0))++bad_normals;
            if(!mesh.HasTangentsAndBitangents()||!finite(mesh.mTangents[i])||!finite(mesh.mBitangents[i])||
                !(mesh.mTangents[i].SquareLength()>0)||!(mesh.mBitangents[i].SquareLength()>0))++bad_tangents;
        }
        for(unsigned i=0;i<mesh.mNumFaces;++i) {
            const auto& f=mesh.mFaces[i];
            if(f.mNumIndices!=3){std::cerr<<"Nontriangle\n";return 1;}
            const auto a=to_metres(mesh.mVertices[f.mIndices[1]])-to_metres(mesh.mVertices[f.mIndices[0]]);
            const auto b=to_metres(mesh.mVertices[f.mIndices[2]])-to_metres(mesh.mVertices[f.mIndices[0]]);
            const auto c=a^b;
            if(!finite(c)||!(c.SquareLength()>0))++local_degenerate;
        }
        if(local_degenerate)bad_meshes.push_back({{"mesh",mi},{"name",mesh.mName.C_Str()},
            {"material",mesh.mMaterialIndex},{"degenerate_triangles",local_degenerate}});
        degenerate+=local_degenerate;
        if(emissive)degenerate_emissive+=local_degenerate;
    }
    output["vertices"]=vertices;output["triangles"]=triangles;
    output["emissive_triangles"]=emissive_triangles;
    output["degenerate_emissive_triangles"]=degenerate_emissive;
    output["bounds_min"]=vec(low);output["bounds_max"]=vec(high);
    output["degenerate_triangles"]=degenerate;output["nonfinite_vertices"]=bad_vertices;
    output["invalid_normals"]=bad_normals;output["invalid_tangents"]=bad_tangents;
    output["degenerate_meshes"]=bad_meshes;
    output["cameras"]=json::array();
    for(unsigned i=0;i<scene->mNumCameras;++i) {
        const auto& c=*scene->mCameras[i];
        const auto camera_position=to_metres(c.mPosition);
        // This pinned PretransformVertices implementation transforms mLookAt
        // as a point (w=1), so it contains a target here, not a direction.
        auto direction=c.mLookAt-c.mPosition;direction.Normalize();
        auto up=c.mUp;up.Normalize();
        const float aspect=c.mAspect>0?c.mAspect:16.f/9.f;
        const float vfov=2.f*std::atan(std::tan(c.mHorizontalFOV)/aspect)*180.f/3.14159265358979323846f;
        output["cameras"].push_back({{"name",c.mName.C_Str()},{"from",vec(camera_position)},
            {"direction",vec(direction)},{"at",vec(camera_position+10.f*direction)},
            {"up",vec(up)},{"assimp_transformed_look_at",vec(c.mLookAt)},
            {"aspect",c.mAspect},{"horizontal_half_fov_rad",c.mHorizontalFOV},{"vertical_fov_deg",vfov}});
    }
    output["lights"]=json::array();
    for(unsigned i=0;i<scene->mNumLights;++i) {
        const auto& l=*scene->mLights[i];
        output["lights"].push_back({{"name",l.mName.C_Str()},{"type",l.mType},
            {"position",vec(l.mPosition)},{"direction",vec(l.mDirection)}});
    }
    std::cout<<output.dump(2)<<'\n';
}
