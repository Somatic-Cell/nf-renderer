#ifndef APPLICATION_HPP_
#define APPLICATION_HPP_

#include "model.h"
#include "renderer.hpp"
#include "glfw_window.hpp"
#include "ptx_data_reader.hpp"
#include "sceneDescIO.hpp"
#include <gl/GL.h>

#define FPNG_IMPLEMENTATION
#include "../ext/fpng/src/fpng.h"


// #if defined(_WIN32)
// #include <windows.h>
// #ifndef NOMINMAX
// #define NOMINMAX
// #endif
// #else
// #include <unistd.h>
// #endif


enum class ApplicationCudaModuleIdentifier{
    CUDA_MODULE_ID_COPYBUFFER_TO_SURFACE=0,
    NUM_CUDA_MODULE_IDENTIFIERS
};

enum class ApplicationCudaKernelIdentifier{
    CUDA_KERNEL_COPYBUFFER_TO_SURFACE=0,
    NUM_CUDA_KERNEL
};

// Renderer でレンダリングを行い，結果を Window に表示する
class Application : public GLFWCameraWindow
{
public:
    Application(
        const std::string &title,
        const sceneIO::Scene scene,
        std::vector<const Model*> models
    ) 
    : GLFWCameraWindow(
        title, 
        make_float3(scene.camera.from[0], scene.camera.from[1], scene.camera.from[2]), 
        make_float3(scene.camera.at[0], scene.camera.at[1], scene.camera.at[2]), 
        make_float3(scene.camera.up[0], scene.camera.up[1], scene.camera.up[2]),
        scene.camera.focalLength,
        scene.camera.fValue,
        scene.camera.fov,
        scene.camera.sensitivity,
        scene.camera.pintDist,
        models.empty() ? 1.0f : length(models[0]->bounds.getSpan())
    ), m_renderer(models, scene)
    {
        Camera camera;
      
        camera.setExtrinsics(
            make_float3(scene.camera.from[0], scene.camera.from[1], scene.camera.from[2]),
            make_float3(scene.camera.at[0], scene.camera.at[1], scene.camera.at[2]),
            make_float3(scene.camera.up[0], scene.camera.up[1], scene.camera.up[2])
        );

        camera.setIntrinsics(
            scene.camera.focalLength,
            scene.camera.fValue,
            scene.camera.fov,
            scene.camera.pintDist,
            scene.camera.sensitivity
        );

        m_renderer.setCamera(camera);
        m_renderer.setEnvMap(scene.environment.file);

        m_scene = scene;


        // 結果を描画するテクスチャの作成
        GLenum texFormat = GL_RGBA;
        GLenum texelType = GL_UNSIGNED_BYTE;
        
        glGenTextures(1, &m_fbTexture);
        glBindTexture(GL_TEXTURE_2D, m_fbTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, m_fbSize.x, m_fbSize.y, 0, texFormat, texelType, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // CUDA に OpenGL のテクスチャを登録
        cudaGraphicsGLRegisterImage(
            &m_cudaTexResource,                         // CUDA 側のリソースハンドル
            m_fbTexture,                                // OpenGL テクスチャ
            GL_TEXTURE_2D,                              // 対象が 2D テクスチャであることを指定
            cudaGraphicsRegisterFlagsSurfaceLoadStore   // 書き込みを有効にするフラグ
        );

        // Surface Object 用のメモリを GPU 上に確保
        // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        // cudaMallocArray(&m_surfaceArray, &channelDesc, m_fbSize.x, m_fbSize.y, cudaArraySurfaceLoadStore);




        // CUDA カーネルの読み込みと登録
        m_cudaModuleFileNames.resize(static_cast<int>(ApplicationCudaModuleIdentifier::NUM_CUDA_MODULE_IDENTIFIERS));
        m_cudaModuleFileNames[static_cast<int>(ApplicationCudaModuleIdentifier::CUDA_MODULE_ID_COPYBUFFER_TO_SURFACE)] = std::string("copy_buffer_to_surface.ptx");

        createCUDAModule();
        fetchCUDAFunction();

        fpng::fpng_init();

    }

    ~Application() override;

    virtual void render() override{

        // GUI からカメラ情報の変更があれば修正
        if(m_cameraFrame.getIsTransformDirty()){
            m_renderer.setCamera(
                Camera{
                    m_cameraFrame.getFrom(),
                    m_cameraFrame.getAt(),
                    m_cameraFrame.getUp(),
                    m_cameraFrame.getFocalLength(),
                    m_cameraFrame.getFValue(),
                    m_cameraFrame.getFov(),
                    m_cameraFrame.getPintDist(),
                    m_cameraFrame.getSensitivity()
                }
            );
            m_cameraFrame.setIsTransformDirty(false);
        }

        m_renderer.render();

    }

    virtual void draw() override;
    inline void drawImGuiContents()
    {
        {
            static int mode = LENS_TYPE_PINHOLE;
            ImGui::Begin("Camera Intrinsic");

            // ImGui::Checkbox("Enable DoF", &show_demo_window);      // Edit bools storing our window open/close state
            if(ImGui::RadioButton("PINHOLE", &mode, LENS_TYPE_PINHOLE)){
                m_cameraFrame.setIsTransformDirty(true);
            } 
            ImGui::SameLine(); 
            if(ImGui::RadioButton("THIN LENS", &mode, LENS_TYPE_THIN_LENS)){
                m_cameraFrame.setIsTransformDirty(true);
            }
            
            ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            if (ImGui::TreeNode("lens parameters")) {
                if (mode == LENS_TYPE_PINHOLE) {
                    m_renderer.setCameraModel(LENS_TYPE_PINHOLE);
                    ImGui::Text("PINHOLE Parameters:");

                    float fov = m_cameraFrame.getFov();
                    if(ImGui::SliderFloat("FOV (degree)", &fov, 10.0f, 80.0f)){
                        m_cameraFrame.setFov(fov);
                        m_cameraFrame.setIsTransformDirty(true);
                    }
                }
                else if (mode == LENS_TYPE_THIN_LENS) {
                    m_renderer.setCameraModel(LENS_TYPE_THIN_LENS);
                    ImGui::Text("Thin lens parameters:");
                    float focalLength = m_cameraFrame.getFocalLength();
                    if(ImGui::SliderFloat("Focal length  (mm)", &focalLength, 20.0f, 600.0f)){
                        m_cameraFrame.setFocalLength(focalLength);
                        m_cameraFrame.setIsTransformDirty(true);
                    }

                    float pintDist = m_cameraFrame.getPintDist(); 
                    if(ImGui::SliderFloat("Pint dist  (m)", &pintDist, 0.0001f, 20.0f)){
                        m_cameraFrame.setPintDist(pintDist);
                        m_cameraFrame.setIsTransformDirty(true);
                    }

                    float fValue = m_cameraFrame.getFValue();
                    if(ImGui::SliderFloat("F value", &fValue, 1.4f, 20.0f)){
                        m_cameraFrame.setFValue(fValue);
                        m_cameraFrame.setIsTransformDirty(true);
                    }
                    float sensitivity = m_cameraFrame.getSensitivity();
                    if(ImGui::SliderFloat("Sensitivity (ISO)", &sensitivity, 0.1f, 20.0f)){
                        m_cameraFrame.setSensitivity(sensitivity);
                        m_cameraFrame.setIsTransformDirty(true);
                    }
                }
                ImGui::TreePop();
            }

            ImGui::End();
        }

        {
            static int mode = COLOR;
            static char* suffix = "";
            ImGui::Begin("Render Mode");
            ImGui::Text("Display buffer:");
            ImGui::RadioButton("COLOR", &mode, COLOR); ImGui::SameLine(); 
            ImGui::RadioButton("NORMAL", &mode, NORMAL); ImGui::SameLine();
            ImGui::RadioButton("ALBEDO", &mode, ALBEDO);
            if(mode == COLOR){
                m_renderer.setRenderBufferType(COLOR);
                suffix = "_COLOR";
            }
            if(mode == NORMAL){
                m_renderer.setRenderBufferType(NORMAL);
                suffix = "_NORMAL";

            } 
            if(mode == ALBEDO){
                m_renderer.setRenderBufferType(ALBEDO);
                suffix = "_ALBEDO";

            }  
            ImGui::Text("Tonemap:");
            float white = m_renderer.getWhite();
            float exposure = m_renderer.getExposure();
            if(ImGui::SliderFloat("Exposure", &exposure, 0.01f, 50.0f)){
                m_renderer.setExposure(exposure);
            }
            if(ImGui::SliderFloat("White", &white, 1.f, 100.0f)){
                m_renderer.setWhite(white);
            }

            ImGui::Text("Save image:");
            static char text[256] = "filename";
            ImGui::InputText("file name:", text, sizeof(text));
            if (ImGui::Button("save as .jpg")) {
                save(std::string(text) + std::string(suffix));
	        }
            ImGui::End();
        }

        {
            ImGui::Begin("Sky params");
            ImGui::Text("Sun direction:");
            float zenithRad = m_renderer.getZenith() / M_PI * 180.0f;  
            float azimuthRad = m_renderer.getAzimuth() / M_PI * 180.0f;  
            if(ImGui::SliderFloat("Zenith angle", &zenithRad, 0.0f, 100.0f)){
                m_renderer.setZenith(zenithRad / 180.0f * M_PI);
                m_cameraFrame.setIsTransformDirty(true);
            }
            if(ImGui::SliderFloat("Azimuth angle", &azimuthRad, 0.f, 360.0f)){
                m_renderer.setAzimuth(azimuthRad / 180.0f * M_PI);
                m_cameraFrame.setIsTransformDirty(true);
            }
            ImGui::End();
        }

        if(m_scene.integrator.applySpectralRendering){
            ImGui::Begin("For Spectral Rendering");
            ImGui::Text("Spectrum Data:");
            ImGui::Text("XYZ CMF: %s\n", m_scene.spectrum.xyzFuncFile.c_str());
            ImGui::Text("Basis function for reflectance: %s\n",  m_scene.spectrum.upSampleBasisFile.c_str());
            ImGui::Text("Illumination: %s\n", m_scene.spectrum.D65File.c_str());
            ImGui::Text("Spectrum Data:");
            ImGui::Text("Wavelength range:");
            ImGui::Text("Min: %f nm, ", m_renderer.getWavelengthMin()); ImGui::SameLine();
            ImGui::Text("Max: %f nm", m_renderer.getWavelengthMax()); 
            
            ImGui::End();
        }

        // {
        //     float density = m_renderer.getDensityScale();
        //     ImGui::Begin("For Volume Rendering");
        //     ImGui::Text("Density:");
        //     if(ImGui::SliderFloat("Scale", &density, 0.01f, 10.0f)){
        //         m_renderer.setDensityScale(density);
        //         m_cameraFrame.setIsTransformDirty(true);
        //     }
        //     ImGui::End();
        // }

        GLFWCameraWindow::drawImGuiContents();

    }

    virtual void save(std::string fileName) override;
         

    virtual void resize(const int2 &newSize) 
    {
        m_fbSize = newSize;
        m_pixels.resize(newSize.x * newSize.y);
        m_renderer.resize(newSize);
        
    }

    virtual void key(int key, int mods) override;
    void createCUDAModule();
    void fetchCUDAFunction();
    void copyBufferToSurface(CUstream stream, cudaSurfaceObject_t surf);

protected:
    int2                    m_fbSize            {make_int2(1280, 720)};
    GLuint                  m_fbTexture         {0};        // レンダリング結果を表示する OpenGL テクスチャ 
    cudaGraphicsResource_t  m_cudaTexResource   {nullptr};
    Renderer                m_renderer;
    std::vector<uint32_t>   m_pixels;

    std::vector<CUmodule>       m_cudaModule;
    std::vector<CUfunction>     m_cudaFunction;
    std::vector<std::string>    m_cudaModuleFileNames;     // cuda 用の.ptx のコード一覧

    cudaSurfaceObject_t     m_surf;
    cudaArray_t             m_surfaceArray;

    sceneIO::Scene          m_scene;
};

#endif // APPLICATION_HPP_