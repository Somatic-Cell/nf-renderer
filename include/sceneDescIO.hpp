#ifndef SCENE_IO_HPP_
#define SCENE_IO_HPP_

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <array>
#include <vector>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace sceneIO {

    struct Environment {
        std::string file = "symmetrical_garden_4k.hdr";
    };

    struct Transform {
        std::array<float, 3> transform{0.0f, 0.0f, 0.0f};
        std::array<float, 4> rotation{0.0f, 0.0f, 0.0f, 1.0f}; // クォータニオン
        std::array<float, 3> scale{1.0f, 1.0f, 1.0f};
    };

    struct Object {
        std::string name = "Cornell Box";
        std::string type = "mesh";
        std::string file = "CornellBox\\CornellBox-Original.obj";
        // Explicit asset convention. Existing scenes retain the legacy loader.
        std::string materialProfile = "legacy";
        Transform TRS{};
        bool placeCenterBBoxAtOrigin = true;
        bool placeOnGround = true;
        bool normalizeScale = true;
    };

    struct Integrator {
        std::string type = "Path Tracing";
        bool        applySpectralRendering = false;
        int         spp  = 1;
        int         maxBounce = 8;
        bool        isAccumulate  = true;

        // M1 is opt-in: existing scenes continue to select the legacy integrator.
        bool        m1Surface = false;
        std::string m1DirectMode = "mis"; // mis, bsdf, nee
        int         m1RrDepth = 5;
        uint32_t    m1Seed = 1;
        bool        m1SunEnabled = true;
        float       m1SunScale = 1.0f;
        float       m1SunAngularRadiusDeg = 0.25f;
        float       m1SunZenithDeg = 35.0f;
        float       m1SunAzimuthDeg = 130.0f;
        float       m1RayEpsilon = 1.0e-4f;
    };

    struct Outputs {
        std::array<int, 2> resolution{1920, 1090};
    };

    struct Camera {
        // extrinsics
        std::array<float, 3> from{0.0f, 0.0f, 3.0f};
        std::array<float, 3> at{0.0f, 0.0f, -1.0f};
        std::array<float, 3> up{0.f, 1.0f, 0.f};

        //intrinsics
        float focalLength   {50.0f};
        float fValue        {2.8f};
        float fov           {50.f};      // degrees
        float pintDist      {1.0f};
        float sensitivity   {1.0f};
    };

    struct Spectrum {
        std::string xyzFuncFile = "xyzbar_from_lms.csv";
        std::string upSampleBasisFile = "basis_rgb.csv";
        // std::string D65File = "CIE_std_illum_D65.csv";
        std::string D65File = "PVL_Spectrum.csv";
        std::string m1AssetDirectory = "spectrum/m1";
    };

    struct Display {
        float exposure = 1.1f;
        float white = 100.0f;
    };

    struct Scene {
        std::vector<Object>     objects         {Object{}};
        Camera                  camera          {};
        Environment             environment     {};
        Integrator              integrator      {};
        Spectrum                spectrum        {};
        Display                 display         {};
        bool                    enableGroundRendering {true};
    };

    // Environment
    inline void from_json(const nlohmann::ordered_json& j, Environment& v){
        v.file = j.value("file", v.file);
    }

    inline void to_json(nlohmann::ordered_json& j, const Environment& v){
        j = {
            {"file",    v.file}
        };
    }

    // Transform
    inline void from_json(const nlohmann::ordered_json& j, Transform& v){
        v.transform = j.value("transform",  v.transform);
        v.rotation  = j.value("rotation",   v.rotation);
        v.scale     = j.value("scale",      v.scale);
    }

    inline void to_json(nlohmann::ordered_json& j, const Transform& v){
        j = {
            {"transform",   v.transform},
            {"rotation",    v.rotation},
            {"scale",       v.scale}
        };
    }

    // Object
    inline void from_json(const nlohmann::ordered_json& j, Object& v){
        v.name                      = j.value("name", v.name);
        v.type                      = j.value("type", v.type);
        v.file                      = j.value("file", v.file);
        v.materialProfile           = j.value("materialProfile", v.materialProfile);
        if (v.materialProfile != "legacy" && v.materialProfile != "orca_metal_rough")
            throw std::invalid_argument("Unsupported materialProfile: " + v.materialProfile);
        v.TRS                       = j.value("TRS", v.TRS);
        v.placeCenterBBoxAtOrigin   = j.value("placeCenterBBoxAtOrigin", v.placeCenterBBoxAtOrigin);
        v.placeOnGround             = j.value("placeOnGround", v.placeOnGround);
        v.normalizeScale            = j.value("normalizeScale", v.normalizeScale);
    }

    inline void to_json(nlohmann::ordered_json& j, const Object& v){
        j = {
            {"name",                    v.name},
            {"type",                    v.type},
            {"file",                    v.file},
            {"materialProfile",         v.materialProfile},
            {"TRS",                     v.TRS},
            {"placeCenterBBoxAtOrigin", v.placeCenterBBoxAtOrigin},
            {"placeOnGround",           v.placeOnGround},
            {"normalizeScale",          v.normalizeScale},
        };
    }

    // Integrator
    inline void from_json(const nlohmann::ordered_json& j, Integrator& v){
        v.type                      = j.value("type", v.type);
        v.applySpectralRendering    = j.value("applySpectralRendering", v.applySpectralRendering);
        v.spp                       = j.value("spp", v.spp);
        v.maxBounce                 = j.value("maxBounce", v.maxBounce);
        v.isAccumulate              = j.value("isAccumulate", v.isAccumulate);
        v.m1Surface                 = j.value("m1Surface", v.m1Surface);
        v.m1DirectMode              = j.value("m1DirectMode", v.m1DirectMode);
        v.m1RrDepth                 = j.value("m1RrDepth", v.m1RrDepth);
        v.m1Seed                    = j.value("m1Seed", v.m1Seed);
        v.m1SunEnabled              = j.value("m1SunEnabled", v.m1SunEnabled);
        v.m1SunScale                = j.value("m1SunScale", v.m1SunScale);
        v.m1SunAngularRadiusDeg     = j.value("m1SunAngularRadiusDeg", v.m1SunAngularRadiusDeg);
        v.m1SunZenithDeg            = j.value("m1SunZenithDeg", v.m1SunZenithDeg);
        v.m1SunAzimuthDeg           = j.value("m1SunAzimuthDeg", v.m1SunAzimuthDeg);
        v.m1RayEpsilon              = j.value("m1RayEpsilon", v.m1RayEpsilon);
    }

    inline void to_json(nlohmann::ordered_json& j, const Integrator& v){
        j = {
            {"type", v.type},
            {"applySpectralRendering", v.applySpectralRendering},
            {"spp", v.spp},
            {"maxBounce", v.maxBounce},
            {"isAccumulate", v.isAccumulate},
            {"m1Surface", v.m1Surface},
            {"m1DirectMode", v.m1DirectMode},
            {"m1RrDepth", v.m1RrDepth},
            {"m1Seed", v.m1Seed},
            {"m1SunEnabled", v.m1SunEnabled},
            {"m1SunScale", v.m1SunScale},
            {"m1SunAngularRadiusDeg", v.m1SunAngularRadiusDeg},
            {"m1SunZenithDeg", v.m1SunZenithDeg},
            {"m1SunAzimuthDeg", v.m1SunAzimuthDeg},
            {"m1RayEpsilon", v.m1RayEpsilon}
        };
    }

    // Outputs
    inline void from_json(const nlohmann::ordered_json& j, Outputs& v){
        v.resolution = j.value("resolution", v.resolution);
    }

    inline void to_json(nlohmann::ordered_json& j, const Outputs& v){
        j = {
            {"resolution", v.resolution}
        };
    }

    // Camera
    inline void from_json(const nlohmann::ordered_json& j, Camera& v){
        v.from      = j.value("from", v.from);
        v.at        = j.value("at", v.at);
        v.up        = j.value("up", v.up);

        v.focalLength   = j.value("focalLength", v.focalLength);
        v.fValue        = j.value("fValue", v.fValue);
        v.fov           = j.value("fov", v.fov);
        v.sensitivity   = j.value("sensitivity", v.sensitivity);
        v.pintDist      = j.value("pintDist", v.pintDist);
    }

    inline void to_json(nlohmann::ordered_json& j, const Camera& v){
        j = {
            {"from", v.from},
            {"at", v.at},
            {"up", v.up},
            {"focalLength", v.focalLength},
            {"fValue", v.fValue},
            {"fov", v.fov},
            {"sensitivity", v.sensitivity},
            {"pintDist", v.pintDist}
        };
    }

    // Spectrum
    inline void from_json(const nlohmann::ordered_json& j, Spectrum& v){
        v.upSampleBasisFile = j.value("upSampleBasisFile", v.upSampleBasisFile);
        v.xyzFuncFile       = j.value("xyzFuncFile", v.xyzFuncFile);
        v.D65File           = j.value("D65File", v.D65File);
        v.m1AssetDirectory  = j.value("m1AssetDirectory", v.m1AssetDirectory);
    }

    inline void to_json(nlohmann::ordered_json& j, const Spectrum& v){
        j = {
            {"upSampleBasisFile", v.upSampleBasisFile},
            {"xyzFuncFile", v.xyzFuncFile},
            {"D65File", v.D65File},
            {"m1AssetDirectory", v.m1AssetDirectory},
        };
    }



    inline void from_json(const nlohmann::ordered_json& j, Display& v){
        v.exposure = j.value("exposure", v.exposure);
        v.white = j.value("white", v.white);
        if (!std::isfinite(v.exposure) || !(v.exposure > 0.0f) ||
            !std::isfinite(v.white) || !(v.white > 0.0f))
            throw std::invalid_argument("Display exposure and white must be finite and positive");
    }

    inline void to_json(nlohmann::ordered_json& j, const Display& v){
        j = {{"exposure", v.exposure}, {"white", v.white}};
    }

    inline void from_json(const nlohmann::ordered_json& j, Scene& v){
        if(j.contains("objects"))       v.objects       = j["objects"].get<std::vector<Object>>();
        if(j.contains("camera"))        v.camera        = j["camera"].get<Camera>();
        if(j.contains("environment"))   v.environment   = j["environment"].get<Environment>();
        if(j.contains("integrator"))    v.integrator    = j["integrator"].get<Integrator>();
        if(j.contains("spectrum"))      v.spectrum      = j["spectrum"].get<Spectrum>();
        if(j.contains("display"))       v.display       = j["display"].get<Display>();
        v.enableGroundRendering                         = j.value("enableGroundRendering", v.enableGroundRendering);
    }

    inline void to_json(nlohmann::ordered_json& j, const Scene& v) 
    {
        j = {
            {"objects", v.objects},
            {"camera",  v.camera},
            {"environment", v.environment},
            {"integrator", v.integrator},
            {"spectrum", v.spectrum},
            {"display", v.display},
            {"enableGroundRendering", v.enableGroundRendering}
        };
    }


    inline nlohmann::ordered_json defaultJson(){
        Scene s{};
        nlohmann::ordered_json j = s;
        return j;
    }

    inline bool parseJson(const std::string& text, nlohmann::ordered_json& out, std::string* err=nullptr){
        try {
            out = nlohmann::ordered_json::parse(text);
            return true;
        } catch (const std::exception& e){
            if(err) *err = e.what();
            return false;
        }
    }
    
    inline bool saveJson(const std::filesystem::path& path, const nlohmann::ordered_json& j, std::string* err=nullptr){
        try {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream os(path, std::ios::binary);
            if(!os) {
                if(err) *err = "Cannot open for write"; 
                return false; 
            }
            os << std::setw(2) << j << '\n';
            return true;
        } catch (const std::exception& e){
            if(err) *err = e.what();
            return false;
        }
    }

    inline bool loadText(const std::filesystem::path& path, std::string& out, std::string* err=nullptr) {
        try{
            std::ifstream is(path, std::ios::binary);
            if(!is){
                if(err) *err = "Cannot open for read"; 
                return false; 
            }
            is.seekg(0, std::ios::end);
            std::string buf;
            buf.resize(static_cast<size_t>(is.tellg()));
            is.seekg(0);
            is.read(buf.data(), buf.size());
            out.swap(buf);
            return true;
        } catch (const std::exception& e){
            if(err) *err = e.what();
            return false;
        }
    }
    
    inline bool loadOrCreate(
        const std::filesystem::path& jsonPath, 
        Scene& out, std::string* 
        err = nullptr, 
        bool writeBackDefaults=true)
    {
        if(!std::filesystem::exists(jsonPath)){
            if (!writeBackDefaults) {
                if (err) *err = "Scene file does not exist: " + jsonPath.string();
                return false;
            }
            nlohmann::ordered_json j = defaultJson();
            if(writeBackDefaults){
                std::string werr;
                if(!saveJson(jsonPath, j, &werr)){
                    if(err){
                        *err = "create: " + werr;
                    }
                }
            }
            out = j.get<Scene>();
            return true;
        }

        std::string text, lerr;
        if(!loadText(jsonPath, text, &lerr)){
            if(err) *err = lerr;
            return false;
        }

        nlohmann::ordered_json j;
        if(!parseJson(text, j, &lerr)){
            if (!writeBackDefaults) {
                if (err) *err = "Scene JSON parse error: " + lerr;
                return false;
            }
            try {
                std::filesystem::rename(jsonPath, jsonPath.string()+".bad"); 
            } catch(...){}
            nlohmann::ordered_json def = defaultJson();
            if(writeBackDefaults){
                saveJson(jsonPath, def);
            }
            out = def.get<Scene>();
            if(err) *err = "parse error; wrote defaults";
            return true;
        }
        nlohmann::ordered_json def = defaultJson();
        nlohmann::ordered_json merged = def;
        merged.merge_patch(j);
        out = merged.get<Scene>();
        if(writeBackDefaults && merged != j) saveJson(jsonPath, merged);
        return true;
    }

    static std::filesystem::path getJsonAbsPath(const std::filesystem::path& p){
        // An explicit absolute path, or a path relative to the working directory,
        // takes precedence over the original executable-relative scene folder.
        if (p.is_absolute()) return p;
        if (std::filesystem::exists(p)) return std::filesystem::absolute(p);
        if (std::filesystem::exists(std::filesystem::path("scene") / p))
            return std::filesystem::absolute(std::filesystem::path("scene") / p);
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
        std::filesystem::path jsonAbsPath = exePath.parent_path().parent_path().parent_path().parent_path() / "scene" / p;
        // std::cout << "Json path (abs): " << jsonAbsPath << std::endl;
        return jsonAbsPath;
    }

    static inline std::string toLower(std::string s){
        for(auto& c : s) c =(char)std::tolower((unsigned char)c);
        return s;
    }

    static inline bool isVolumeFile(const std::filesystem::path& p){
        auto e = toLower(p.extension().string());
        return (e == ".vdb" || e == ".nvdb");
    }

    static inline bool isMeshFile(const std::filesystem::path& p){
        static const std::unordered_set<std::string> extensions{
            ".fbx", ".obj", ".gltf", ".glb", ".ply"
        };
        return extensions.count(toLower(p.extension().string())) > 0;
    }
}

#endif // SCENE_IO_HPP_
