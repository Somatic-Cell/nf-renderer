#include "optix8.hpp"
#include <stdexcept>
#include <stdio.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <assert.h>
#include <string>
#include <vector>
#include "sceneDescIO.hpp"

#include "application.hpp"



extern "C" int main(int ac, char **av)
  {
    try {

    // 引数から Json ファイルのパスを決定
    std::filesystem::path jsonPath = (ac > 1) ? std::filesystem::path(av[1]) : std::filesystem::path("scene.json");
    std::filesystem::path jsonAbsPath = sceneIO::getJsonAbsPath(jsonPath);
    std::cout << "[Scene] using: " << jsonAbsPath.u8string() << std::endl;

    // Json ファイルを読み込む
    sceneIO::Scene sceneDesc;
    std::string jerr;
    // Explicit scene files are read-only inputs: a typo or parse failure must
    // not rename the file or render a different default scene.
    if(!sceneIO::loadOrCreate(jsonAbsPath, sceneDesc, &jerr, ac <= 1)) {
      std::cerr << "FATAL: scene load failed: " << jerr << std::endl;
      return 1;
    }
    if(!jerr.empty()){
      std::cerr << "[Scene] : note: " << jerr << std::endl;
    }
    
  
      // モデルファイルの読み込み
      std::vector<std::string> modelFileNames;
      std::vector<std::unique_ptr<Model>> ownedModels;
      std::vector<const Model*> models;
      models.reserve(sceneDesc.objects.size());
      for(const auto& obj : sceneDesc.objects){
          if(obj.type != "mesh") {
            if (sceneDesc.integrator.m1Surface)
              throw std::invalid_argument("M1 objects must have type mesh: " + obj.name);
            continue;
          }
          std::cout << "Model File Name:" << obj.file << std::endl;
          
          if(auto* m = loadModel(obj.file, obj)){
            ownedModels.emplace_back(m);
            models.push_back(m);
          }else {
            if (sceneDesc.integrator.m1Surface)
              throw std::runtime_error("M1 model failed to load: " + obj.file);
            std::cerr << "WARNING: failed to load model: " << obj.file << std::endl;
          }
      }
      if(sceneDesc.integrator.m1Surface && models.empty())
        throw std::runtime_error("M1 scene has no loaded meshes");

      // 環境マップの指定
      std::string envMapFileName;
      if(!sceneDesc.environment.file.empty()){
        envMapFileName = sceneDesc.environment.file;
      } else {
        envMapFileName = "symmetrical_garden_4k.hdr";
      }

      Application application("Atmospheric RT", sceneDesc, models);
      application.enableFlyMode();
      application.run();
    
    } catch (const std::exception& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
