#include "m2/orca_scene.hpp"
#include "m2/orca_material.hpp"
#include "m2/tangent_frame.hpp"
#include "m1/bsdf.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <array>

using m1::Vec3;
using json=nlohmann::ordered_json;
static Vec3 vec(aiVector3D v){return {v.x,v.y,v.z};}
static json jvec(Vec3 v){return {v.x,v.y,v.z};}
int main(int argc,char** argv){
    if(argc!=3)return 2;
    Assimp::Importer importer;m2::configure_orca_importer(importer);
    const auto* scene=importer.ReadFile(argv[1],m2::orca_import_flags);
    if(!scene){std::cerr<<importer.GetErrorString()<<'\n';return 1;}
    const double metres=m2::orca_metres_per_unit(*scene);
    if(metres!=1)return 1;
    std::vector<m2::OrcaMaterial> materials;
    for(unsigned i=0;i<scene->mNumMaterials;++i)materials.push_back(m2::read_orca_material(*scene->mMaterials[i]));
    uint64_t mixed=0,probes=0,valid=0,oldZero=0,oldZeroValid=0,degenerateCharts=0,zeroChartProbes=0;
    uint64_t unitFailures=0,orthogonalityFailures=0;
    float maxDot=0;double minimumAbsDet=INFINITY;
    json failures=json::array(),examples=json::array(),degenerateExamples=json::array();
    const std::array<std::array<float,3>,7> points={{{1.f/3,1.f/3,1.f/3},{.5f,.25f,.25f},
        {.25f,.5f,.25f},{.25f,.25f,.5f},{0,.5f,.5f},{.5f,0,.5f},{.5f,.5f,0}}};
    for(unsigned mi=0;mi<scene->mNumMeshes;++mi){
        const auto& mesh=*scene->mMeshes[mi];const auto& material=materials[mesh.mMaterialIndex];
        if(!material.normalTexture.present())continue;
        for(unsigned fi=0;fi<mesh.mNumFaces;++fi){
            const auto& f=mesh.mFaces[fi];
            std::array<Vec3,3> n,t,p;std::array<float,3> signs;std::array<m1::Vec2,3> uv;
            for(unsigned k=0;k<3;++k){
                const auto i=f.mIndices[k];n[k]=vec(mesh.mNormals[i]);t[k]=vec(mesh.mTangents[i]);p[k]=vec(mesh.mVertices[i]);
                signs[k]=m1::dot(m1::cross(n[k],t[k]),vec(mesh.mBitangents[i]))<0?-1.f:1.f;
                uv[k]={mesh.mTextureCoords[0][i].x,mesh.mTextureCoords[0][i].y};
            }
            if(signs[0]==signs[1]&&signs[0]==signs[2])continue;
            const Vec3 e1=p[1]-p[0],e2=p[2]-p[0],ng=m1::normalize(m1::cross(e1,e2));
            if(!m1::bsdf_unit_direction(ng))continue;
            ++mixed;
            bool first=true;
            for(const auto& w:points){
                ++probes;
                Vec3 ns=m1::normalize(w[0]*n[0]+w[1]*n[1]+w[2]*n[2]);
                if(m1::dot(ns,ng)<0)ns=-ns;
                const float old=w[0]*signs[0]+w[1]*signs[1]+w[2]*signs[2];
                if(old==0)++oldZero;
                const auto result=m2::triangle_uv_tangent_frame(e1,e2,uv[0],uv[1],uv[2],ns);
                if(result.valid){++valid;if(old==0)++oldZeroValid;}
                if(result.used_degenerate_uv_chart){++zeroChartProbes;if(first)++degenerateCharts;}
                if(result.uv_determinant!=0)minimumAbsDet=std::min(minimumAbsDet,std::abs(result.uv_determinant));
                const bool unit=result.valid&&m1::bsdf_unit_direction(result.tangent)&&m1::bsdf_unit_direction(result.bitangent);
                if(!unit)++unitFailures;
                const float orthogonal=result.valid?std::max({std::abs(m1::dot(result.tangent,ns)),
                    std::abs(m1::dot(result.bitangent,ns)),std::abs(m1::dot(result.tangent,result.bitangent))}):INFINITY;
                if(orthogonal>2.e-6f)++orthogonalityFailures;
                maxDot=std::max(maxDot,orthogonal);
                const bool needFailure=(!result.valid||!unit||orthogonal>2.e-6f)&&failures.size()<32;
                const bool needExample=old==0&&examples.size()<8;
                const bool needDegenerate=result.used_degenerate_uv_chart&&degenerateExamples.size()<8;
                if(needFailure||needExample||needDegenerate){
                const json record={{"mesh",mi},{"primitive",fi},{"material",material.name},
                    {"normal_texture",material.normalTexture.path},{"two_sided",material.twoSided},
                    {"ng",jvec(ng)},{"ns",jvec(ns)},{"barycentric",w},{"old_interpolated_sign",old},
                    {"uv_determinant",result.uv_determinant},{"used_degenerate_uv_chart",result.used_degenerate_uv_chart},
                    {"tangent",jvec(result.tangent)},{"bitangent",jvec(result.bitangent)},{"helper_valid",result.valid}};
                if(needFailure)failures.push_back(record);
                if(needExample)examples.push_back(record);
                if(needDegenerate)degenerateExamples.push_back(record);
                }
                first=false;
            }
        }
    }
    json out={{"flags",m2::orca_import_flags},{"mixed_normal_mapped_triangles",mixed},
        {"all_mixed_probes",probes},{"helper_valid_probes",valid},
        {"old_zero_sign_probes",oldZero},{"old_zero_sign_helper_valid_probes",oldZeroValid},
        {"degenerate_uv_mixed_triangles",degenerateCharts},{"degenerate_uv_chart_probes",zeroChartProbes},
        {"nonunit_frame_failures",unitFailures},{"orthogonality_failures",orthogonalityFailures},
        {"max_abs_frame_dot",maxDot},{"min_abs_nonzero_uv_determinant",minimumAbsDet},
        {"failure_examples",failures},{"repaired_actual_face_examples",examples},
        {"degenerate_uv_chart_examples",degenerateExamples},
        {"passed",probes==valid&&oldZero==oldZeroValid&&!unitFailures&&!orthogonalityFailures}};
    std::ofstream(argv[2])<<out.dump(2)<<'\n';
    out.erase("failure_examples");out.erase("repaired_actual_face_examples");out.erase("degenerate_uv_chart_examples");
    std::cout<<out.dump(2)<<'\n';return out["passed"].get<bool>()?0:1;
}
