#include "application.hpp"


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
    
    const std::string saveFileName = "output\\" + fileName + std::string(".png");

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

    fpng::fpng_encode_image_to_file(saveFileName.c_str(), pixels.data(), (uint32_t)m_fbSize.x, (uint32_t)m_fbSize.y, (uint32_t)4);
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