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

    // 引数から Json ファイルのパスを決定
    std::filesystem::path jsonPath = (ac > 1) ? std::filesystem::path(av[1]) : std::filesystem::path("scene.json");
    std::filesystem::path jsonAbsPath = sceneIO::getJsonAbsPath(jsonPath);
    std::cout << "[Scene] using: " << jsonAbsPath.u8string() << std::endl;

    // Json ファイルを読み込む
    sceneIO::Scene sceneDesc;
    std::string jerr;
    if(!sceneIO::loadOrCreate(jsonAbsPath, sceneDesc, &jerr)) {
      std::cerr << "FATAL: scene load failed: " << jerr << std::endl;
      return 1;
    }
    if(!jerr.empty()){
      std::cerr << "[Scene] : note: " << jerr << std::endl;
    }
    
  
    try {

      // モデルファイルの読み込み
      std::vector<std::string> modelFileNames;
      std::vector<const Model*> models;
      models.reserve(sceneDesc.objects.size());
      for(const auto& obj : sceneDesc.objects){
          if(obj.type != "mesh") continue;
          std::cout << "Model File Name:" << obj.file << std::endl;
          
          if(auto* m = loadModel(obj.file, obj)){
            models.push_back(m);
          }else {
            std::cerr << "WARNING: failed to load model: " << obj.file << std::endl;
          }
      }
      // if(models.empty()) {
      //   throw std::runtime_error("no model loaded");
      // }

      // 環境マップの指定
      std::string envMapFileName;
      if(!sceneDesc.environment.file.empty()){
        envMapFileName = sceneDesc.environment.file;
      } else {
        envMapFileName = "symmetrical_garden_4k.hdr";
      }

      Application *application = new Application("Atmospheric RT", sceneDesc, models);

      application->enableFlyMode();
      application->run();
    
    } catch (std::runtime_error& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }