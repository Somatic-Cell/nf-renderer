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
#include <limits>
#include "m3/tracking.hpp"
#include "m4_rain/scene.hpp"

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
        // Matches the existing initial window size when the field is absent.
        std::array<int, 2> resolution{1280, 720};
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
        float wavelengthMinNm=360.0f,wavelengthMaxNm=830.0f;
    };

    struct Display {
        float exposure = 1.1f;
        float white = 100.0f;
    };

    struct Scene {
        std::vector<Object>     objects         {Object{}};
        Outputs                 outputs         {};
        Camera                  camera          {};
        Environment             environment     {};
        Integrator              integrator      {};
        Spectrum                spectrum        {};
        Display                 display         {};
        // Independent opt-in medium. Absent in existing M1/M2 scenes = vacuum.
        m3::TrackingParameters   rayleigh        {};
        m4rain::SceneSettings    rain            {};
        bool                    enableGroundRendering {true};
    };

    inline void validateRayleighParameters(const m3::TrackingParameters& parameters) {
        const auto status = m3::tracking::validate_parameters(parameters);
        if (status != m3::tracking::Status::Ok)
            throw std::invalid_argument(std::string("Invalid rayleigh parameters: ") + m3::tracking::status_name(status));
    }

    // Keep the scene, launch, and output metadata on the same parameter POD.
    // Method/law tags are fixed contracts, not selectable unimplemented methods.
    inline nlohmann::ordered_json rayleighJson(const m3::TrackingParameters& v) {
        return {
            {"enabled", v.enabled != 0},
            {"areaNeeTransmittance", v.areaNeeTransmittance != 0},
            {"solver", "delta_ratio_fp32"},
            {"wavelengthLaw", "lambda^-4"},
            {"planetRadius", v.planetRadius},
            {"atmosphereHeight", v.atmosphereHeight},
            {"scaleHeight", v.scaleHeight},
            {"groundY", v.groundY},
            {"beta550", v.beta550},
            {"groundAlbedo", v.groundAlbedo},
            {"ratioRrThreshold", v.ratioRrThreshold},
            {"maxTrackingEvents", v.maxTrackingEvents}
        };
    }

    inline void readRayleigh(const nlohmann::ordered_json& j, m3::TrackingParameters& v) {
        if (!j.is_object()) throw std::invalid_argument("rayleigh must be an object");
        if (j.value("solver", std::string("delta_ratio_fp32")) != "delta_ratio_fp32")
            throw std::invalid_argument("rayleigh.solver must be delta_ratio_fp32. Convert an old M3 scene with tools/migrate_m3_tracking_scene.py; numerical_reference is CPU-only.");
        static const std::unordered_set<std::string> fields{
            "enabled", "areaNeeTransmittance", "solver", "wavelengthLaw", "planetRadius",
            "atmosphereHeight", "scaleHeight", "groundY", "beta550", "groundAlbedo",
            "ratioRrThreshold", "maxTrackingEvents"
        };
        for (auto it = j.begin(); it != j.end(); ++it)
            if (fields.count(it.key()) == 0)
                throw std::invalid_argument("Unknown rayleigh field: " + it.key());
        if (j.value("wavelengthLaw", std::string("lambda^-4")) != "lambda^-4")
            throw std::invalid_argument("rayleigh.wavelengthLaw must be lambda^-4");

        const auto flag = [&](const char* field, uint32_t fallback) {
            if (!j.contains(field)) return fallback;
            if (!j[field].is_boolean()) throw std::invalid_argument(std::string("rayleigh.") + field + " must be boolean");
            return j[field].get<bool>() ? 1u : 0u;
        };
        const auto budget = [&](const char* field, uint32_t fallback) {
            if (!j.contains(field)) return fallback;
            const auto& value = j[field];
            if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
                throw std::invalid_argument(std::string("rayleigh.") + field + " must be a positive integer");
            const auto number = value.get<uint64_t>();
            if (number > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::string("rayleigh.") + field + " exceeds uint32 range");
            return static_cast<uint32_t>(number);
        };
        const auto number = [&](const char* field, float fallback) {
            if (!j.contains(field)) return fallback;
            if (!j[field].is_number())
                throw std::invalid_argument(std::string("rayleigh.") + field + " must be a finite number");
            // Parsing and validation are CPU-only. Reject a nonzero parameter
            // rounded to zero, especially a density that would become vacuum.
            const double source = j[field].get<double>();
            const float value = static_cast<float>(source);
            if (!std::isfinite(source) || !std::isfinite(value) || (source != 0.0 && value == 0.0f))
                throw std::invalid_argument(std::string("rayleigh.") + field + " is outside the nonzero FP32 range");
            return value;
        };
        v.enabled = flag("enabled", v.enabled);
        v.areaNeeTransmittance = flag("areaNeeTransmittance", v.areaNeeTransmittance);
        v.planetRadius = number("planetRadius", v.planetRadius);
        v.atmosphereHeight = number("atmosphereHeight", v.atmosphereHeight);
        v.scaleHeight = number("scaleHeight", v.scaleHeight);
        v.groundY = number("groundY", v.groundY);
        v.beta550 = number("beta550", v.beta550);
        v.groundAlbedo = number("groundAlbedo", v.groundAlbedo);
        v.ratioRrThreshold = number("ratioRrThreshold", v.ratioRrThreshold);
        v.maxTrackingEvents = budget("maxTrackingEvents", v.maxTrackingEvents);
        validateRayleighParameters(v);
    }

    inline void validateRayleighScene(const Scene& scene) {
        validateRayleighParameters(scene.rayleigh);
        if (!scene.rayleigh.enabled) return;
        if (!scene.integrator.m1Surface || !scene.integrator.applySpectralRendering)
            throw std::invalid_argument("rayleigh.enabled requires m1Surface=true and applySpectralRendering=true");
        if (scene.integrator.type != "Path Tracing")
            throw std::invalid_argument("rayleigh requires integrator.type=Path Tracing");
        if (!scene.environment.file.empty() || scene.enableGroundRendering)
            throw std::invalid_argument("rayleigh requires environment.file empty and enableGroundRendering=false; it supplies its own spherical ground");
        for (const auto& object : scene.objects) {
            auto extension = std::filesystem::path(object.file).extension().string();
            for (auto& c : extension) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (object.type != "mesh" || extension == ".vdb" || extension == ".nvdb")
                throw std::invalid_argument("rayleigh supports only mesh objects, not VDB or other media: " + object.name);
        }
        for (const auto value : scene.camera.from)
            if (!std::isfinite(value)) throw std::invalid_argument("rayleigh camera position must be finite");
        const auto cameraHeight = m3::tracking::height(scene.rayleigh,
            m1::Vec3{scene.camera.from[0], scene.camera.from[1], scene.camera.from[2]});
        if (cameraHeight.status != m3::tracking::Status::Ok || !(cameraHeight.height_m > 0.0f))
            throw std::invalid_argument("rayleigh camera must start above the opaque planet surface (check groundY)");
    }

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
        if (!j.is_object()) throw std::invalid_argument("outputs must be an object");
        for (auto it = j.begin(); it != j.end(); ++it)
            if (it.key() != "resolution") throw std::invalid_argument("Unknown outputs field: " + it.key());
        if (!j.contains("resolution")) return;
        const auto& resolution = j["resolution"];
        if (!resolution.is_array() || resolution.size() != 2)
            throw std::invalid_argument("outputs.resolution must be [width, height]");
        for (size_t axis = 0; axis < 2; ++axis) {
            const auto& value = resolution[axis];
            if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() <= 0))
                throw std::invalid_argument("outputs.resolution dimensions must be positive integers");
            const auto number = value.get<uint64_t>();
            if (number == 0 || number > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("outputs.resolution dimension is outside the supported integer range");
            v.resolution[axis] = static_cast<int>(number);
        }
        const auto pixels = uint64_t(v.resolution[0]) * uint64_t(v.resolution[1]);
        if (pixels > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("outputs.resolution exceeds the framebuffer index range");
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
        if(j.contains("wavelengthMinNm"))v.wavelengthMinNm=m4rain::json_float(j["wavelengthMinNm"],"spectrum.wavelengthMinNm");
        if(j.contains("wavelengthMaxNm"))v.wavelengthMaxNm=m4rain::json_float(j["wavelengthMaxNm"],"spectrum.wavelengthMaxNm");
        if(!(v.wavelengthMinNm>0&&v.wavelengthMinNm<v.wavelengthMaxNm))
            throw std::invalid_argument("Invalid spectrum wavelength interval");
    }

    inline void to_json(nlohmann::ordered_json& j, const Spectrum& v){
        j = {
            {"upSampleBasisFile", v.upSampleBasisFile},
            {"xyzFuncFile", v.xyzFuncFile},
            {"D65File", v.D65File},
            {"m1AssetDirectory", v.m1AssetDirectory},
            {"wavelengthMinNm",v.wavelengthMinNm},{"wavelengthMaxNm",v.wavelengthMaxNm},
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
        if(j.contains("outputs"))       v.outputs       = j["outputs"].get<Outputs>();
        if(j.contains("camera"))        v.camera        = j["camera"].get<Camera>();
        if(j.contains("environment"))   v.environment   = j["environment"].get<Environment>();
        if(j.contains("integrator"))    v.integrator    = j["integrator"].get<Integrator>();
        if(j.contains("spectrum"))      v.spectrum      = j["spectrum"].get<Spectrum>();
        if(j.contains("display"))       v.display       = j["display"].get<Display>();
        if(j.contains("rayleigh"))      readRayleigh(j["rayleigh"], v.rayleigh);
        if(j.contains("rain"))          m4rain::read(j["rain"],v.rain);
        v.enableGroundRendering                         = j.value("enableGroundRendering", v.enableGroundRendering);
        validateRayleighScene(v);
        m4rain::validate_scene(v.rain,v.rayleigh,v.spectrum.wavelengthMinNm,v.spectrum.wavelengthMaxNm);
    }

    inline void to_json(nlohmann::ordered_json& j, const Scene& v) 
    {
        j = {
            {"objects", v.objects},
            {"outputs", v.outputs},
            {"camera",  v.camera},
            {"environment", v.environment},
            {"integrator", v.integrator},
            {"spectrum", v.spectrum},
            {"display", v.display},
            {"rayleigh", rayleighJson(v.rayleigh)},
            {"rain",m4rain::to_json(v.rain)},
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
        // JSON merge-patch treats null as deletion. Preserve the submitted M3
        // object so null/bad types cannot silently disable it or reset a value.
        if (j.contains("rayleigh")) merged["rayleigh"] = j["rayleigh"];
        if (j.contains("rain")) merged["rain"] = j["rain"];
        if (j.contains("outputs")) merged["outputs"] = j["outputs"];
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
