#include "sceneDescIO.hpp"
#include <chrono>
#include <limits>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void rejects(const nlohmann::ordered_json& input, const char* message) {
    bool rejected = false;
    try { (void)input.get<sceneIO::Scene>(); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "pass repository root");
        const std::filesystem::path root(argv[1]);
        for (const auto* name : {"sampleScene.json", "sampleScene2.json"}) {
            sceneIO::Scene scene;
            std::string error;
            require(sceneIO::loadOrCreate(root / "scene" / name, scene, &error, false), "legacy scene parse");
            require(!scene.integrator.m1Surface, "legacy input must not enable M1");
            require(!scene.rayleigh.enabled, "legacy input must not enable M3");
            require(scene.outputs.resolution == std::array<int,2>{1280,720}, "existing initial window size is preserved");
            require(scene.objects[0].materialProfile == "legacy", "legacy material profile default");
            require(scene.display.exposure == 1.1f && scene.display.white == 100.0f,
                    "existing scenes retain their display defaults");
        }
        for (const auto* name : {"m2_emerald_day.json", "m2_emerald_overview.json"}) {
            sceneIO::Scene scene;
            std::string error;
            require(sceneIO::loadOrCreate(root / "scene" / name, scene, &error, false), "M2 preset parse");
            require(scene.objects.size() == 1 && scene.objects[0].materialProfile == "orca_metal_rough",
                    "M2 explicitly selects the ORCA material convention");
            require(scene.integrator.m1Surface && scene.integrator.applySpectralRendering &&
                    scene.integrator.m1DirectMode == "mis", "M2 reuses the spectral MIS integrator");
            require(scene.environment.file.empty() && !scene.enableGroundRendering, "M2 vacuum configuration");
            require(!scene.rayleigh.enabled, "M2 input remains in vacuum");
            require(!scene.objects[0].normalizeScale && !scene.objects[0].placeOnGround &&
                    !scene.objects[0].placeCenterBBoxAtOrigin, "M2 preserves authored coordinates");
            const nlohmann::ordered_json serialized = scene;
            const auto copy = serialized.get<sceneIO::Scene>();
            require(copy.objects[0].materialProfile == "orca_metal_rough" &&
                    copy.display.exposure == 0.02f && copy.display.white == 100.0f,
                    "M2 profile and display survive serialization");
        }
        for (const auto* invalid : {
                 "{\"objects\":[{\"materialProfile\":\"unknown\"}]}",
                 "{\"display\":{\"exposure\":0}}",
                 "{\"display\":{\"white\":-1}}"}) {
            bool rejected = false;
            try { (void)nlohmann::ordered_json::parse(invalid).get<sceneIO::Scene>(); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "invalid material profile/display must not silently substitute defaults");
        }
        for (const auto* lighting : {"area", "sun", "combined"}) {
            for (const auto* mode : {"mis", "bsdf", "nee"}) {
                sceneIO::Scene scene;
                std::string error;
                const auto path = root / "scene" / (std::string("m1_") + lighting + '_' + mode + ".json");
                require(sceneIO::loadOrCreate(path, scene, &error, false), "M1 fixture parse");
                require(scene.integrator.m1Surface && scene.integrator.applySpectralRendering, "M1 spectral selection");
                require(scene.integrator.m1DirectMode == mode, "direct estimator mode selection");
                require(scene.environment.file.empty() && !scene.enableGroundRendering, "M1 vacuum configuration");
                require(!scene.rayleigh.enabled, "M1 input remains in vacuum");
                require(std::filesystem::is_directory(root / scene.spectrum.m1AssetDirectory), "spectral assets directory");
                for (const auto& object : scene.objects) {
                    require(!object.placeCenterBBoxAtOrigin && !object.placeOnGround, "fixture instance transforms");
                    require(std::filesystem::is_regular_file(root / "model/mesh" / object.file), "fixture mesh exists");
                }
                nlohmann::ordered_json serialized = scene;
                const auto roundTrip = serialized.get<sceneIO::Scene>();
                require(roundTrip.integrator.m1DirectMode == mode && roundTrip.integrator.m1Seed == 17,
                        "M1 settings survive JSON serialization");
            }
        }
        nlohmann::ordered_json skyJson, dayJson, vacuumJson;
        for (const auto* name : {"m3_rayleigh_sky.json", "m3_emerald_day.json", "m3_emerald_overview.json",
                                 "m3_emerald_sunset.json", "m3_emerald_vacuum.json"}) {
            sceneIO::Scene scene;
            std::string error;
            require(sceneIO::loadOrCreate(root / "scene" / name, scene, &error, false), "M3 preset parse");
            const bool sky = std::string(name) == "m3_rayleigh_sky.json";
            const bool vacuum = std::string(name) == "m3_emerald_vacuum.json";
            require(scene.rayleigh.enabled == 1, "M3 presets retain the spherical planet, including the vacuum control");
            require(vacuum ? scene.rayleigh.beta550 == 0.0 : scene.rayleigh.beta550 > 0.0,
                    "vacuum control changes only atmospheric scattering to zero");
            require(scene.integrator.m1Surface && scene.integrator.applySpectralRendering,
                    "M3 reuses the existing spectral path and assets");
            require(scene.environment.file.empty() && !scene.enableGroundRendering, "M3 must not select legacy sky/ground");
            require(scene.integrator.spp == 1, "M3 tracking presets start at one sample per launch");
            require(scene.rayleigh.groundY == -6.0 && scene.rayleigh.planetRadius == 6371000.0,
                    "M3 metre-based ground datum and radius are explicit");
            require(scene.rayleigh.areaNeeTransmittance == 1, "M3 default area MIS includes medium transmittance");
            require(scene.outputs.resolution == (sky ? std::array<int,2>{512,288} : std::array<int,2>{640,360}),
                    "M3 initial preview resolution is present in each scene");
            if (sky) require(scene.objects.empty(), "clean M3 sky has no external mesh dependency");
            else require(scene.objects.size() == 1 && scene.objects[0].materialProfile == "orca_metal_rough" &&
                         !scene.objects[0].normalizeScale && !scene.objects[0].placeOnGround &&
                         !scene.objects[0].placeCenterBBoxAtOrigin, "M3 Emerald retains authored coordinates");
            if (std::string(name) == "m3_emerald_sunset.json")
                require(scene.integrator.m1SunZenithDeg == 88.0f, "sunset preset has a low finite sun disk");
            const nlohmann::ordered_json serialized = scene;
            const auto copy = serialized.get<sceneIO::Scene>();
            require(sceneIO::rayleighJson(copy.rayleigh) == sceneIO::rayleighJson(scene.rayleigh) &&
                    copy.outputs.resolution == scene.outputs.resolution, "M3 physical/numerical settings survive serialization");
            if (sky) skyJson = serialized;
            if (std::string(name) == "m3_emerald_day.json") dayJson = serialized;
            if (vacuum) vacuumJson = serialized;
        }
        dayJson["rayleigh"]["beta550"] = 0.0;
        require(dayJson == vacuumJson, "M3 vacuum control changes only beta550; camera, planet, sun, and estimator stay fixed");

        // Every mutable parameter is changed from its default. A dropped field
        // must not pass a round-trip test by accidentally restoring its default.
        const auto customRayleigh = nlohmann::ordered_json::parse(R"({
            "enabled":true,"areaNeeTransmittance":false,
            "solver":"delta_ratio_fp32","wavelengthLaw":"lambda^-4",
            "planetRadius":6000001,"atmosphereHeight":90001,"scaleHeight":7501,"groundY":-5.75,
            "beta550":0.000025,"groundAlbedo":0.25,
            "ratioRrThreshold":0.03125,"maxTrackingEvents":98765
        })");
        auto custom = skyJson;
        custom["rayleigh"] = customRayleigh;
        custom["outputs"]["resolution"] = {320,180};
        const auto customScene = custom.get<sceneIO::Scene>();
        const nlohmann::ordered_json customSerialized = customScene;
        auto expectedRayleigh = customRayleigh;
        expectedRayleigh["beta550"] = static_cast<float>(0.000025);
        require(customSerialized["rayleigh"] == expectedRayleigh,
                "physical values round once to the FP32 launch format; every field is serialized");
        const auto customCopy = customSerialized.get<sceneIO::Scene>();
        require(sceneIO::rayleighJson(customCopy.rayleigh) == customSerialized["rayleigh"],
                "FP32 serialization is stable across repeated scene loads");
        require(customSerialized["outputs"]["resolution"] == custom["outputs"]["resolution"], "output resolution round trip");

        for (const auto& field : {"enabled", "areaNeeTransmittance"}) {
            auto bad = skyJson; bad["rayleigh"][field] = 1;
            rejects(bad, "M3 flags accept JSON booleans, not integers");
        }
        for (const auto& value : {nlohmann::ordered_json(-1), nlohmann::ordered_json(4294967296ULL),
                                  nlohmann::ordered_json(12.5)}) {
            auto bad = skyJson; bad["rayleigh"]["maxTrackingEvents"] = value;
            rejects(bad, "work budgets reject wrapped/fractional integers before uint32 conversion");
        }
        for (const auto* field : {"planetRadius", "atmosphereHeight", "scaleHeight", "maxTrackingEvents"}) {
            auto bad = skyJson; bad["rayleigh"][field] = 0;
            rejects(bad, "physical dimensions and work budget must be positive");
        }
        for (const auto value : {-0.01, 1.01}) {
            auto bad = skyJson; bad["rayleigh"]["ratioRrThreshold"] = value;
            rejects(bad, "ratio roulette threshold must be in [0,1]");
        }
        for (const auto value : {1e-100, 1e100}) {
            auto bad = skyJson; bad["rayleigh"]["beta550"] = value;
            rejects(bad, "FP32 conversion must not silently create zero density or infinity");
        }
        for (const auto& value : {nlohmann::ordered_json(true), nlohmann::ordered_json("8000"),
                                  nlohmann::ordered_json(nullptr)}) {
            auto bad = skyJson; bad["rayleigh"]["scaleHeight"] = value;
            rejects(bad, "physical parameters require numeric JSON values");
        }
        {
            auto noRoulette = skyJson; noRoulette["rayleigh"]["ratioRrThreshold"] = 0;
            require(noRoulette.get<sceneIO::Scene>().rayleigh.ratioRrThreshold == 0.0f,
                    "explicitly disabling ratio roulette is supported");
            auto old = skyJson; old["rayleigh"]["solver"] = "numerical_reference";
            rejects(old, "old solver tags require explicit scene migration");
            old = skyJson; old["rayleigh"]["tauAbsoluteTolerance"] = 1e-10;
            rejects(old, "obsolete quadrature options cannot be silently ignored");
        }
        {
            auto bad = skyJson; bad["rayleigh"]["scaleHieght"] = 8000;
            rejects(bad, "unknown M3 field must not silently use a default");
            bad = skyJson; bad["rayleigh"]["solver"] = "chapman";
            rejects(bad, "unimplemented attenuation solver is rejected");
            bad = skyJson; bad["rayleigh"]["wavelengthLaw"] = "constant";
            rejects(bad, "unimplemented wavelength law is rejected");
            bad = skyJson; bad["integrator"]["m1Surface"] = false;
            rejects(bad, "M3 must not silently select the legacy renderer");
            bad = skyJson; bad["integrator"]["applySpectralRendering"] = false;
            rejects(bad, "M3 requires explicit spectral rendering");
            bad = skyJson; bad["environment"]["file"] = "sky.hdr";
            rejects(bad, "M3 does not mix an environment map with its atmosphere");
            bad = skyJson; bad["enableGroundRendering"] = true;
            rejects(bad, "M3 does not mix the old ground renderer with its planet");
            bad = skyJson; bad["objects"] = nlohmann::ordered_json::array({{{"type","vdb"},{"file","cloud.vdb"}}});
            rejects(bad, "M3 does not accept VDB media");
            bad = skyJson; bad["camera"]["from"] = {0,-7,0};
            rejects(bad, "M3 camera cannot start inside the planet");
            bad = skyJson; bad["camera"]["from"] = {0,-6,0};
            rejects(bad, "M3 camera cannot start exactly on the planet boundary");
            bad = skyJson; bad["rayleigh"] = nullptr;
            rejects(bad, "null M3 settings must not silently disable the medium");
        }
        for (const auto* resolution : {"[0,288]", "[-1,288]", "[512.5,288]", "[512]", "[512,288,1]",
                                      "[4294967297,1]", "[65536,65536]"}) {
            auto bad = skyJson; bad["outputs"]["resolution"] = nlohmann::ordered_json::parse(resolution);
            rejects(bad, "output dimensions must be exactly two positive representable integers");
        }
        {
            auto invalid = customScene;
            invalid.rayleigh.beta550 = std::numeric_limits<double>::quiet_NaN();
            bool rejected = false;
            try { sceneIO::validateRayleighScene(invalid); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "programmatically constructed M3 scenes also validate before upload");
        }
        const auto temporary = std::filesystem::temp_directory_path() /
            ("neuralpf-m1-json-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(temporary);
        const auto path = temporary / "bad.json";
        sceneIO::Scene scene;
        std::string error;
        require(!sceneIO::loadOrCreate(path, scene, &error, false), "explicit missing file must fail");
        require(!std::filesystem::exists(path), "missing scene must not create default file");
        { std::ofstream output(path); output << "{invalid JSON"; }
        require(!sceneIO::loadOrCreate(path, scene, &error, false), "explicit malformed file must fail");
        require(!std::filesystem::exists(path.string() + ".bad"), "strict parse must not rename input");
        std::string contents;
        require(sceneIO::loadText(path, contents) && contents == "{invalid JSON", "strict parse must preserve input");
        for (const auto* nullInput : {"{\"rayleigh\":null}", "{\"rayleigh\":{\"enabled\":null}}",
                                      "{\"outputs\":{\"resolution\":null}}"}) {
            { std::ofstream output(path); output << nullInput; }
            bool rejected = false;
            try { rejected = !sceneIO::loadOrCreate(path, scene, &error, false); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "merge-patch null must not erase an invalid M3/output setting");
            require(sceneIO::loadText(path, contents) && contents == nullInput, "semantic error must preserve scene input");
        }
        std::filesystem::remove_all(temporary);
        std::cout << "Scene selection: 9 M1 fixtures + 2 M2 + 5 M3 presets + 2 legacy scenes; strict failures preserve inputs\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
