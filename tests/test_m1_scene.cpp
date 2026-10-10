#include "sceneDescIO.hpp"
#include <chrono>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
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
        std::filesystem::remove_all(temporary);
        std::cout << "M1 scene selection: 9 fixtures + 2 legacy scenes; strict failures preserve inputs\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
