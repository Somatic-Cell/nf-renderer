#include "sceneDescIO.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
static void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(int argc,char** argv){try{
    check(argc==2,"pass repository root");const std::filesystem::path root(argv[1]);
    nlohmann::ordered_json rain_scene;
    for(const char* file:{"m4_rain_sky.json","m4_emerald_rain.json","m4_emerald_rain_fp32.json","m4_emerald_no_rain.json","m4_emerald_rain_exponential.json"}){
        std::ifstream input(root/"scene"/file);nlohmann::ordered_json j;input>>j;
        const auto scene=j.get<sceneIO::Scene>();
        nlohmann::ordered_json encoded=scene;const auto copy=encoded.get<sceneIO::Scene>();
        check(copy.spectrum.wavelengthMinNm==380&&copy.spectrum.wavelengthMaxNm==830,"wavelength domain");
        check(m4rain::to_json(scene.rain)==m4rain::to_json(copy.rain),"rain round trip");rain_scene=j;
    }
    for(const char* file:{"m3_rayleigh_sky.json","m3_emerald_day.json"}){
        std::ifstream input(root/"scene"/file);nlohmann::ordered_json j;input>>j;
        const auto scene=j.get<sceneIO::Scene>();
        check(!scene.rain.parameters.enabled&&scene.spectrum.wavelengthMinNm==360,"M3 defaults changed");
    }
    auto rejected=[&](nlohmann::ordered_json j){try{(void)j.get<sceneIO::Scene>();}catch(const std::exception&){return;}throw std::runtime_error("invalid scene accepted");};
    auto j=rain_scene;j["rain"]["backend"]="wrong";rejected(j);
    j=rain_scene;j["rain"]["enabled"]=1;rejected(j);
    j=rain_scene;j["spectrum"]["wavelengthMinNm"]=360;rejected(j);
    j=rain_scene;j["rain"]["numberDensityAtMinY"]=-1;rejected(j);
    j=rain_scene;j["rayleigh"]["enabled"]=false;rejected(j);
    j=rain_scene;j["rain"]["boundsMin"]={-3000,-100,-3000};rejected(j);
    j=rain_scene;j["rain"]["unknownSetting"]=1;rejected(j);
    std::cout<<"5 rain scenes + 2 M3 defaults + 7 rejection cases PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
