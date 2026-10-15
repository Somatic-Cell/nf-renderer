#include "application.hpp"
#include <fstream>
#include <limits>


Application::~Application()
{
    cudaDestroySurfaceObject(m_surf);

}

void Application::key(int key, int mods) 
{
    switch(key){
        default:
            GLFWCameraWindow::key(key, mods);
            break;
    }
}

void Application::save(std::string fileName) 
{
    // レンダリング結果を CPU のメモリに転送
    m_renderer.downloadPixels(m_pixels.data());
    
    const std::filesystem::path stem = std::filesystem::path("output") / fileName;
    if (!stem.parent_path().empty()) std::filesystem::create_directories(stem.parent_path());
    const std::string saveFileName = stem.string() + ".png";

    // 反転
    std::vector<uint32_t> flipped = m_pixels;
    for(int y = 0; y < m_fbSize.y; y++){
        uint32_t* srcRow = &m_pixels[y * m_fbSize.x];
        uint32_t* dstRow = &flipped[(m_fbSize.y - 1 - y) * m_fbSize.x];
        std::memcpy(dstRow, srcRow, m_fbSize.x * sizeof(uint32_t));
    }

    // 書き込み
    std::vector<uint8_t> pixels;
    pixels.resize(m_pixels.size() * sizeof(uint32_t));
    std::memcpy(pixels.data(), flipped.data(), pixels.size());

    // 書き込み
    uint32_t sizeX = (uint32_t)m_fbSize.x;
    uint32_t sizeY = (uint32_t)m_fbSize.y;
    uint32_t channel = 4;

    if (!fpng::fpng_encode_image_to_file(saveFileName.c_str(), pixels.data(), sizeX, sizeY, channel))
        throw std::runtime_error("Could not write image: " + saveFileName);

    if (m_scene.integrator.m1Surface) {
        // The existing display/PNG path is tonemapped. Save the actual film
        // separately for MIS/BSDF/NEE comparisons, including negative RGB values.
        std::vector<float4> linear(m_pixels.size());
        m_renderer.downloadLinearPixels(linear.data());
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                      "PFM export requires IEEE-754 binary32");
        const uint32_t endianProbe = 1;
        const bool littleEndian = *reinterpret_cast<const unsigned char*>(&endianProbe) == 1;
        const std::string pfmPath = stem.string() + ".linear.pfm";
        std::ofstream pfm(pfmPath, std::ios::binary);
        if (!pfm) throw std::runtime_error("Could not open linear image: " + pfmPath);
        pfm << "PF\n" << sizeX << ' ' << sizeY << '\n' << (littleEndian ? "-1.0\n" : "1.0\n");
        // PFM rows run bottom to top; the existing framebuffer has that order.
        for (const auto& p : linear) {
            const float rgb[3] = {p.x, p.y, p.z};
            pfm.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
        pfm.close();
        if (!pfm) throw std::runtime_error("Could not finish linear image: " + pfmPath);

        const auto params = m_renderer.getLaunchParams();
        nlohmann::ordered_json info;
        info["schema"] = "NeuralPFRenderer-M1-film-v1";
        info["encoding"] = "float32 linear sRGB from raw CIE XYZ integrals";
        info["wavelength_nm"] = {params.spectral.wavelengthMin, params.spectral.wavelengthMax};
        info["wavelength_pdf"] = "uniform; compensated exactly once in the film";
        info["xyz_normalization"] = "none; no division by integral(ybar) and no 683 factor";
        info["width"] = sizeX;
        info["height"] = sizeY;
        info["requested_initial_resolution"] = m_scene.outputs.resolution;
        info["accumulated_frames"] = params.frame.frameID;
        info["spp_per_frame"] = params.frame.numPixelSamples;
        info["total_spp"] = uint64_t(params.frame.frameID) * uint64_t(params.frame.numPixelSamples);
        info["direct_mode"] = m_scene.integrator.m1DirectMode;
        info["seed"] = params.m1.seed;
        info["max_bounce"] = params.frame.maxBounce;
        info["rr_depth"] = params.m1.rrDepth;
        info["sun_scale"] = params.m1.sunScale;
        info["sun_enabled"] = bool(params.m1.sunEnabled);
        info["sun_angular_radius_rad"] = params.m1.sunAngularRadiusRad;
        info["sun_zenith_rad"] = params.sunParams.sunZenithRad;
        info["sun_azimuth_rad"] = params.sunParams.sunAzimuthRad;
        info["light_intensity_factor"] = params.light.lightIntensityFactor;
        info["ray_epsilon"] = params.m1.rayEpsilon;
        info["integrator_model"] = params.rayleigh.enabled ? "M3 spherical Rayleigh path tracing" : "M1/M2 spectral surface path tracing";
        info["rayleigh"] = sceneIO::rayleighJson(params.rayleigh);
        if(params.rain.parameters.enabled){
            const bool mie=params.rain.parameters.phaseModel==m4rain::PhaseModel::SphereMie1mm;
            info["integrator_model"]=mie?"M4 sphere Lorenz-Mie rain + M3 Rayleigh, component-labelled path tracing":
                                           "M4 oriented NF rain + M3 Rayleigh, component-labelled path tracing";
            auto rain=m_scene.rain;rain.parameters=params.rain.parameters;
            info["rain"]=m4rain::to_json(rain);
            if(mie) {
                info["rain"]["sphereMieSha256"]=m_renderer.getRainAssetSha256();
                info["rain"]["mieSolver"]="miepython 3.3.0, unpolarized";
            } else {
                info["rain"]["selectedStep"]=params.rain.selectedStep;
                info["rain"]["modelSha256"]=m_renderer.getRainAssetSha256();
            }
            info["rain"]["crossSections"]=mie?
                (params.rain.parameters.crossSectionMode==m4mie::CrossSection::PhysicalMie?
                    "Csca=pi*r^2*Qsca, Cext=pi*r^2*Qext": "Csca=Cext=pi*r^2 (geometric comparison)"):
                "Csca=Cext=Aproj; Cabs=0";
            info["rain"]["proposal"]=mie?"Mie p(dot(d,w)); sampling/evaluation from the same LUT":
                                               "P_NF(w|d); physical K uses ki=-w,ko=-d";
            if(!mie)info["rain"]["endpointExtension"]="g/C only; NN receives unchanged signed cosine";
            info["rain"]["componentSampling"]="independent competing free flights; conditional component NEE and q";
        }
        if (params.rayleigh.enabled) {
            info["rayleigh_model"] = {
                {"coordinates", "metres; +Y up"},
                {"planet_center_m", {0.0, static_cast<double>(params.rayleigh.groundY) - static_cast<double>(params.rayleigh.planetRadius), 0.0}},
                {"density", "exp(-altitude/scaleHeight), truncated at atmosphereHeight"},
                {"scattering", "scalar unpolarized Rayleigh; pure scattering, no absorption"},
                {"beta_wavelength", "beta550 * (550 / wavelength_nm)^4"},
                {"ground", "opaque, spectrally neutral Lambertian"},
                {"medium_area_light_nee", false},
                {"area_shadow_transmittance_omission", params.rayleigh.areaNeeTransmittance == 0},
                {"arithmetic", "FP32; compensated FP32 pairs for geometry and ray-parameter accumulation"},
                {"majorant", "wavelength-dependent global sea-level coefficient"},
                {"free_flight", "analog delta tracking; distance-sampling throughput is one"},
                {"shadow_transmittance", "ratio tracking with optional unbiased Russian roulette"},
                {"random_numbers", "PCG32 stream, 23-bit open-interval midpoint for tracking"},
                {"tracking_budget", "exhaustion rejects the frame; no accepted truncated product or flight"},
                {"numerical_reference", "FP64 Gauss-Kronrod / inverse CDF retained in CPU tests only"}
            };
        }
        info["display"] = {{"exposure", m_renderer.getExposure()},
                           {"white", m_renderer.getWhite()}};
        info["objects"] = m_scene.objects;
        info["camera"] = {
            {"from", {params.camera.position.x, params.camera.position.y, params.camera.position.z}},
            {"direction", {params.camera.direction.x, params.camera.direction.y, params.camera.direction.z}},
            {"mode", params.camera.cameraMode}, {"fov_degrees", params.camera.fov},
            {"focal_length_mm", params.camera.focalLength}, {"f_value", params.camera.fValue},
            {"focus_distance", params.camera.pintDist}, {"sensitivity", params.camera.sensitivity}
        };
        std::ofstream metadata(stem.string() + ".linear.json");
        metadata << std::setw(2) << info << '\n';
        metadata.close();
        if (!metadata) throw std::runtime_error("Could not write linear image metadata");
        std::cout << "Saved " << pfmPath << " (" << info["total_spp"] << " spp)" << std::endl;
    }
}

void Application::draw() {


    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    
    // m_renderer.downloadPixels(m_pixels.data());
    if (m_fbTexture == 0)
    {
        glGenTextures(1, &m_fbTexture);
        glBindTexture(GL_TEXTURE_2D, m_fbTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, m_fbSize.x, m_fbSize.y, 0, texFormat, texelType, nullptr);
    }

    auto stream = m_renderer.getStream();

    cudaGraphicsMapResources(1, &m_cudaTexResource, stream);

    cudaArray_t surfaceArray = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&surfaceArray, m_cudaTexResource, 0, 0);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = surfaceArray;

    cudaSurfaceObject_t surf = 0;
    cudaCreateSurfaceObject(&surf, &resDesc);

    copyBufferToSurface(stream, surf);
    cudaDestroySurfaceObject(surf);

    cudaGraphicsUnmapResources(1, &m_cudaTexResource, stream);

    glBindTexture(GL_TEXTURE_2D, m_fbTexture);
    
    glDisable(GL_LIGHTING);
    glColor3f(1.0f, 1.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, m_fbSize.x, m_fbSize.y);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)m_fbSize.x, 0.f, (float)m_fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
    
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)m_fbSize.y, 0.f);
    
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)m_fbSize.x, (float)m_fbSize.y, 0.f);
    
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)m_fbSize.x, 0.f, 0.f);
    }
    glEnd();

    glFlush();

    glGetError();
}

// ptx ファイルを読み込み，CUDA のモジュールを登録
void Application::createCUDAModule()
{
    const int numCudaModule = m_cudaModuleFileNames.size();
    m_cudaModule.resize(numCudaModule);
    // m_cudaFunction.resize();

    // ptx ファイルの読み出し
    for(size_t i = 0; i < m_cudaModuleFileNames.size(); i++){
        std::vector<char> ptxCode = readData(m_cudaModuleFileNames[i]);
        if (ptxCode.empty()) {
            throw std::runtime_error("PTX code is empty. Check file path: " + m_cudaModuleFileNames[i]);
        }
        
        std::filesystem::path exe_path;
        char path_buffer[MAX_PATH] = {};
#if defined(_WIN32)
        if(GetModuleFileNameA(NULL, path_buffer, MAX_PATH) == 0){
            std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
        }
        exe_path = std::filesystem::path(path_buffer);
#else
        ssize_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
        if(count == -1) {
            std::cerr << "ERROR: readlink() failed" << std::endl;
        }
#endif
        std::filesystem::path ptx_dir = exe_path.parent_path().parent_path() / "ptxes" / m_cudaModuleFileNames[i];

        std::cout << "Loading CUDA module..." << std::endl;
        CUDA_DRIVER_CHECK(cuModuleLoad(&m_cudaModule[i], ptx_dir.string().c_str()));
        std::cout << "Loaded CUDA module." << std::endl;
    }
}

// モジュールから CUDA カーネルを検出して登録
void Application::fetchCUDAFunction()
{
    
    std::cout << "Fetching kernel function..." << std::endl;
    
    // 登録
    m_cudaFunction.resize(static_cast<int>(ApplicationCudaKernelIdentifier::NUM_CUDA_KERNEL));

    CUDA_DRIVER_CHECK(cuModuleGetFunction(&m_cudaFunction[static_cast<int>(ApplicationCudaKernelIdentifier::CUDA_KERNEL_COPYBUFFER_TO_SURFACE)], m_cudaModule[static_cast<int>(ApplicationCudaModuleIdentifier::CUDA_MODULE_ID_COPYBUFFER_TO_SURFACE)], "copyBufferToSurfaceKernel"));
    std::cout << "Fetched kernel function." << std::endl;
}

// GPU 上にある uint32_t 型の描画結果を，GPU 上の OpenGL テクスチャにコピー
void Application::copyBufferToSurface(CUstream stream, cudaSurfaceObject_t surf)
{
    int2 fbSize = m_renderer.getLaunchParams().frame.size;
    int2 blockSize = make_int2(32);
    int2 numBlocks = make_int2(std::ceil((float)fbSize.x / (float)blockSize.x), std::ceil((float)fbSize.y / (float)blockSize.y));

    CUdeviceptr finalColorBufferPtr = m_renderer.getFinalColorBuffer().getDevicePointer();
    cudaSurfaceObject_t surface = surf;
    
    void* arg0 = &finalColorBufferPtr;
    void* arg1 = &surface;
    void* arg2 = &fbSize;

    void* args[] = {
        arg0, arg1, arg2
    };

    CUDA_DRIVER_CHECK(
        cuLaunchKernel(
            m_cudaFunction[static_cast<int>(ApplicationCudaKernelIdentifier::CUDA_KERNEL_COPYBUFFER_TO_SURFACE)],
            numBlocks.x, numBlocks.y, 1,    // スレッドのブロック数
            blockSize.x, blockSize.y, 1,    // 各ブロック内のスレッド数
            0,
            stream,
            args,
            nullptr 
        )
    );
    CUDA_SYNC_CHECK();
}
