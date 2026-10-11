#ifndef GLFW_WINDOW_HPP_
#define GLFW_WINDOW_HPP_

#include <string>
#include <iostream>
#include <memory>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_math.h"
#include "my_math.hpp"

#include "imgui.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

// OpenGL のヘッダファイルを自動でインクルードさせない
// 
#define GLFW_INCLUDE_NONE

class MyGLFWWindow{

public:
    MyGLFWWindow(const std::string &title, int initialWidth = 1280, int initialHeight = 720);
    virtual ~MyGLFWWindow();

    virtual void draw()
    {}

    virtual void resize(const int2 &newSize)
    {}

    virtual void key(int key, int mods)
    {}

    virtual void mouseMotion(const int2 &newPos)
    {}

    virtual void mouseButton(int button, int action, int mods)
    {}

    inline int2 getMousePos() const
    {
        double x, y;
        glfwGetCursorPos(m_handle, &x, &y);
        return make_int2((int)x, (int)y);
    }

    virtual void render()
    {}

    virtual void save(std::string fileName)
    {}

    virtual void run()
    {}

    ImGuiIO* getImGuiIO() 
    {
        return m_io;
    }

protected:
    ImGuiIO     *m_io           {nullptr};
    GLFWwindow  *m_handle       {nullptr};
    ImVec4      m_clearColor    {ImVec4(0.45f, 0.55f, 0.60f, 1.00f)};
};

// --------------------------------------------------------
// カメラの外部・内部パラメータの状態を保持，制御するためのクラス
// --------------------------------------------------------
class CameraFrame{

public:
    CameraFrame(const float worldScale) : m_motionSpeed(worldScale)
    {}


    float3 getPointOfInterest() const 
    {
        return m_position - m_poiDistance * m_zAxis;
    }

    // 強制的に upVector をカメラの Y 軸方向に設定
    void forceUpFrame()
    {
        // 既に upVector がカメラの Y 軸方向と一致していたら，そのまま
        if(fabsf(dot(m_yAxis, m_upVector)) < 1e-6f){
                return;
        }

        m_xAxis = normalize(cross(m_upVector, m_zAxis));
        m_yAxis = normalize(cross(m_zAxis, m_xAxis));
        m_isTransformDirty = true;
    }

    // カメラの外部パラメータを設定
    void setOrientation( 
        const float3 &origin,
        const float3 &interest,
        const float3 &up
    )
    {
        m_position = origin;
        m_upVector = up;
        m_zAxis = (interest == origin) ? make_float3(0.0f, 0.0f, 1.0f) : -1.0f * normalize(interest - origin);
        m_xAxis = cross(m_upVector, m_zAxis);
        if(dot(m_xAxis, m_xAxis) < 1e-8f){
            m_xAxis = make_float3(0.0f, 1.0f, 0.0f);
        } else {
            m_xAxis = normalize(m_xAxis);
        }
        m_yAxis = normalize(cross(m_zAxis, m_xAxis));
        m_poiDistance = length(interest - origin);
        forceUpFrame();
    }

    void setUpVector(const float3 &up)
    {
        m_upVector = up;
        forceUpFrame();
    }
    float getFocalLength() const {
        return m_focalLength;
    }

    void setFocalLength(const float focalLength) {
        m_focalLength = focalLength;
    }

    float getFValue() const {
        return m_fValue;
    }

    void setFValue(const float fValue) {
        m_fValue = fValue;
    }

    float getFov() const {
        return m_fov;
    }

    void setFov(const float fov) {
        m_fov = fov;
    }
    float getPintDist() const {
        return m_pintDist;
    }

    void setPintDist(const float pintDist) {
        m_pintDist = pintDist;
    }
    void setIsTransformDirty(const bool isTransformDirty)
    {
        m_isTransformDirty = isTransformDirty;
    }

    bool getIsTransformDirty() const 
    {
        return m_isTransformDirty;
    }

    inline float getSensitivity() const {return m_sensitivity; }
    
    void setSensitivity(float sensitivity) {
        m_sensitivity = sensitivity; 
    }

    void setMotionSpeed(float motionSpeed){
        m_motionSpeed = motionSpeed;
    }

    void setPosition(const float3 position)
    {
        m_position = position;
    }

    void setPoiDistance(const float poiDistance)
    {
        m_poiDistance = poiDistance;
    }

    float getPoiDistance() const 
    {
        return m_poiDistance;
    }

    void setXAxis(const float3 xAxis){
        m_xAxis = xAxis;
    }

    void setYAxis(const float3 yAxis){
        m_yAxis = yAxis;
    }

    void setZAxis(const float3 zAxis){
        m_zAxis = zAxis;
    }

    inline float getMotionSpeed()   const { return m_motionSpeed; }
    inline bool getForceUp()        const { return m_forceUp; }

    inline float3 getFrom()         const { return m_position; }
    inline float3 getAt()           const { return getPointOfInterest();}
    inline float3 getUp()           const { return m_upVector; }
    inline float3 getXAxis()        const { return m_xAxis; }
    inline float3 getYAxis()        const { return m_yAxis; }
    inline float3 getZAxis()        const { return m_zAxis; }

protected:
    inline float computeStableEpsilon(const float f) const{
        return fabsf(f) * float(1./(1<<21));
    }

    inline float computeStableEpsilon(const float3 v) const{
        return fmaxf( fmaxf(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
    }

private:
    // カメラの外部パラメータ
    float3 m_xAxis  {make_float3(1.0f, 0.0, 0.0f)};
    float3 m_yAxis  {make_float3(0.0f, 1.0, 0.0f)};
    float3 m_zAxis  {make_float3(0.0f, 0.0, 1.0f)};

    float3 m_position       {make_float3(0.0f, -1.0f, 0.0f)};
    float  m_poiDistance    {1.0f};
    float3 m_upVector       {make_float3(0.0f, 1.0, 0.0f)};

    // カメラの内部パラメータ
    float m_focalLength   {50.0f};    // mm
    float m_fValue        {20.0f};    // 
    float m_sensorSize    {35.0f};    // mm
    float m_sensitivity   {1.0f};
    float m_pintDist      {100.0f};   // m
    float m_fov           {50.0f};    // degree
    
    bool  m_forceUp       {true};
    float m_motionSpeed   {1.0f};
    bool  m_isTransformDirty      {true};

};

// --------------------------------------------------------
// ユーザの入力を通して CameraFrame の内部状態を変更するクラス
// --------------------------------------------------------
class CameraFrameManip {

public:

    CameraFrameManip(CameraFrame *cameraFrame) : m_cameraFrame(cameraFrame)
    {}
    // キーボード入力に従ってカメラの情報を変更
    virtual void key(const int key, const int mods)
    {
        CameraFrame &cf = *m_cameraFrame;

        switch(key) {
            // motionSpeed を変更
            case '+':
            case '=':
                cf.setMotionSpeed(cf.getMotionSpeed() * 1.5f);
                std::cout << "now motion speed is " << cf.getMotionSpeed() << std::endl;
                break;
            case '-':
            case '_':
                cf.setMotionSpeed(cf.getMotionSpeed() / 1.5f);
                std::cout << "now motion speed is " << cf.getMotionSpeed() << std::endl;
                break;
            
            // カメラの外部パラメータを表示
            case 'c':
            case 'C':
                displayCameraExtrinsicParam();
                break;

            // X, -X, Y, -Y, Z, -Z 軸方向にカメラの V 方向をセット
            case 'x':
            case 'X':
                if(length(cf.getUp() - make_float3(1.0f, 0.0f, 0.0f)) < 1e-4f){
                    cf.setUpVector(make_float3(-1.0f, 0.0f, 0.0f)); 
                } else {
                    cf.setUpVector(make_float3( 1.0f, 0.0f, 0.0f));
                }
                break;
            case 'y':
            case 'Y':
                if(length(cf.getUp() - make_float3(0.0f, 1.0f, 0.0f)) < 1e-4f){
                    cf.setUpVector(make_float3(0.0f, -1.0f, 0.0f)); 
                } else {
                    cf.setUpVector(make_float3(0.0f,  1.0f, 0.0f));
                }
                break;
            case 'z':
            case 'Z':
                if(length(cf.getUp() - make_float3(0.0f, 0.0f, 1.0f)) < 1e-4f){
                    cf.setUpVector(make_float3(0.0f, 0.0f, -1.0f)); 
                } else {
                    cf.setUpVector(make_float3(0.0f, 0.0f,  1.0f));
                }
                break;
            default:
                break;
        }
        
    }

    // 3次元空間上でのカメラの平行移動
    // ワールド座標で平行移動
    virtual void strafe(const float3 &howMuch)
    {
        const float3 newPos = m_cameraFrame->getFrom() + howMuch;
        m_cameraFrame->setPosition(newPos);
        m_cameraFrame->setIsTransformDirty(true);
    }

    // 2次元空間上でのカメラの平行移動
    // カメラ座標に沿ってカメラ位置を移動
    virtual void strafe(const float2 &howMuch)
    {
        const float3  xAxis = m_cameraFrame->getXAxis();
        const float3  yAxis = m_cameraFrame->getYAxis();
        strafe (+howMuch.x * xAxis - howMuch.y * yAxis);
    }

    // 必ずサブクラスで実装する関数
    virtual void move(const float step) = 0;
    virtual void rotate(const float dx, const float dy) = 0;

    // マウス左ボタンのドラッグでは，カメラを回転
    virtual void mouseDragLeft(const float2 &delta)
    {
        rotate(delta.x * m_degreesPerDragFraction, delta.y * m_degreesPerDragFraction);
        if(m_verbose){
            displayCameraExtrinsicParam();
        }
    }

    // マウス中ボタンのドラッグでは，カメラを上下左右に移動
    virtual void mouseDragMiddle(const float2 &delta)
    {
        strafe(delta * m_pixelPerMove * m_cameraFrame->getMotionSpeed());
        if(m_verbose){
            displayCameraExtrinsicParam();
        }
    }

    // マウス右ボタンのドラッグでは，カメラを前後に移動
    virtual void mouseDragRight(const float2 &delta)
    {
        move(delta.y * m_pixelPerMove * m_cameraFrame->getMotionSpeed());
        if(m_verbose){
            displayCameraExtrinsicParam();
        }
    }

protected:
    void displayCameraExtrinsicParam() const 
    {
        std::cout << "Current camera: " << std::endl;
        const float3 from = m_cameraFrame->getFrom();
        const float3 poi = m_cameraFrame->getPointOfInterest();
        const float3 up = m_cameraFrame->getUp();

        std::cout << "- from x:" << from.x << ", y:" << from.y << ", z:" << from.z << std::endl;
        std::cout << "-  poi x:" << poi.x  << ", y:" << poi.y  << ", z:" << poi.z  << std::endl;
        std::cout << "-   up x:" << up.x   << ", y:" << up.y   << ", z:" << up.z   << std::endl;
    }

    CameraFrame *m_cameraFrame;
    const float m_keyboardRotateDegrees   {10.0f};
    const float m_degreesPerDragFraction  {15.0f};
    const float m_pixelPerMove            {10.0f};

    bool m_verbose = false; // デバッグ時に true 
};

// --------------------------------------------------------
// GLFW を使ってウィンドウとマウス入力を受け取り， カメラを操作するクラス
// --------------------------------------------------------
class GLFWCameraWindow : public MyGLFWWindow {

public:
    GLFWCameraWindow(
        const std::string &title,
        const float3 & cameraFrom,
        const float3 & cameraAt,
        const float3 & cameraUp,
        const float focalLength,
        const float fValue,
        const float fov,
        const float sensitivity,
        const float pintDist,
        const float worldScale,
        const int initialWidth = 1280,
        const int initialHeight = 720
    ) : MyGLFWWindow(title, initialWidth, initialHeight),
    m_cameraFrame(worldScale)
    {
        m_cameraFrame.setOrientation(cameraFrom, cameraAt, cameraUp);
        m_cameraFrame.setFocalLength(focalLength);
        m_cameraFrame.setFValue(fValue);
        m_cameraFrame.setFov(fov);
        m_cameraFrame.setSensitivity(sensitivity);
        m_cameraFrame.setPintDist(pintDist);
        
        enableInspectMode();
        enableFlyMode();
    }

    virtual void run() override;

    void enableFlyMode();
    void enableInspectMode();

    virtual void drawImGuiContents();

    virtual void key(int key, int mods) override{
        if(!m_io->WantCaptureKeyboard){
            switch(key)
            {
                case 'f':
                case 'F':
                    std::cout << "Entering 'fly' mode" << std::endl;
                    if(m_flyModeManip) m_cameraFrameManip = m_flyModeManip;
                    break;
                case 'i':
                case 'I':
                    std::cout << "Entering 'inspect' mode" << std::endl;
                    if(m_inspectModeManip) m_cameraFrameManip = m_inspectModeManip;
                    break;
                default:
                    if(m_cameraFrameManip){
                        m_cameraFrameManip->key(key, mods);
                    }
                    break;
            }
        }
    }
    
    // マウスが動いたときに，方向と移動量に合わせてカメラを操作
    virtual void mouseMotion(const int2 &newPos) override{
        if(!m_io->WantCaptureMouse){
            // windowのサイズを取得
            int2 windowSize;
            glfwGetWindowSize(m_handle, &windowSize.x, &windowSize.y);

            float2 delta = make_float2(
                (float)(newPos.x - m_lastMousePos.x) / (float)windowSize.x, 
                (float)(newPos.y - m_lastMousePos.y) / (float)windowSize.y 
            );

            if(m_isPressed.leftButton && m_cameraFrameManip){
                m_cameraFrameManip->mouseDragLeft(delta);
            }

            if(m_isPressed.middleButton && m_cameraFrameManip){
                m_cameraFrameManip->mouseDragMiddle(delta);
            }

            if(m_isPressed.rightButton && m_cameraFrameManip){
                m_cameraFrameManip->mouseDragRight(delta);
            }

            m_lastMousePos = newPos;
        }

    }

    // マウスボタンが押されたときに，ボタンの状態を記録
    virtual void mouseButton(int button, int action, int mods) override
    {
        const bool pressed = (action == GLFW_PRESS);

        m_io->AddMouseButtonEvent(button, pressed);
        if(!m_io->WantCaptureMouse){
            switch (button){
                case GLFW_MOUSE_BUTTON_LEFT:
                    m_isPressed.leftButton = pressed;
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    m_isPressed.middleButton = pressed;
                    break;
                case GLFW_MOUSE_BUTTON_RIGHT:
                    m_isPressed.rightButton = pressed;
                    break;
                default:
                    break;
            }
            m_lastMousePos = getMousePos();
        }
    }

    CameraFrame& getCameraFramePtr()
    {
        return m_cameraFrame;
    }

protected:

    struct {
        bool leftButton     {false};
        bool middleButton   {false};
        bool rightButton    {false};
    } m_isPressed;
    int2 m_lastMousePos     {make_int2(-1, -1)};
    CameraFrame m_cameraFrame;

    friend struct CameraFrameManip;
    std::shared_ptr<CameraFrameManip> m_cameraFrameManip;
    std::shared_ptr<CameraFrameManip> m_inspectModeManip;
    std::shared_ptr<CameraFrameManip> m_flyModeManip;
};


// --------------------------------------------------------
// 注目点を中心に視点移動するカメラの動きを実装したクラス
// --------------------------------------------------------
class InspectModeManip : public CameraFrameManip{

public:

    InspectModeManip(CameraFrame* cameraFrame)
    : CameraFrameManip(cameraFrame)
    {}

private:
    virtual void rotate(const float deg_u, const float deg_v) override {
        float rad_u = M_PI / 180.0f * deg_u;
        float rad_v = -M_PI / 180.0f * deg_v;

        const float3 poi = m_cameraFrame->getPointOfInterest();

        float3 xAxis = m_cameraFrame->getXAxis();
        float3 yAxis = m_cameraFrame->getYAxis();
        float3 zAxis = m_cameraFrame->getZAxis();

        mymath::rotate(xAxis, rad_v, xAxis, yAxis, zAxis);
        mymath::rotate(yAxis, rad_u, xAxis, yAxis, zAxis);

        m_cameraFrame->setXAxis(xAxis);
        m_cameraFrame->setYAxis(yAxis);
        m_cameraFrame->setZAxis(zAxis);

        if(m_cameraFrame->getForceUp()){
            m_cameraFrame->forceUpFrame();
        }

        m_cameraFrame->setPosition(poi + m_cameraFrame->getPoiDistance() * m_cameraFrame->getZAxis());
        m_cameraFrame->setIsTransformDirty(true);
    }

    virtual void move(const float step) override
    {
        const float3 poi = m_cameraFrame->getPointOfInterest();
        const float minReqDistance = 0.1f * m_cameraFrame->getMotionSpeed();
        
        m_cameraFrame->setPoiDistance(fmaxf(minReqDistance, m_cameraFrame->getPoiDistance() - step));
        m_cameraFrame->setPosition(poi + m_cameraFrame->getPoiDistance() * m_cameraFrame->getZAxis());
        m_cameraFrame->setIsTransformDirty(true);
    }

};

// --------------------------------------------------------
// 一人称視点でカメラを移動するクラス
// --------------------------------------------------------
class FlyModeManip : public CameraFrameManip{

public:

    FlyModeManip(CameraFrame* cameraFrame)
    : CameraFrameManip(cameraFrame)
    {}

private:

    virtual void rotate(const float deg_u, const float deg_v) override {
        float rad_u = M_PI / 180.0f * deg_u;
        float rad_v = -M_PI / 180.0f * deg_v;

        const float3 poi = m_cameraFrame->getPointOfInterest();

        float3 xAxis = m_cameraFrame->getXAxis();
        float3 yAxis = m_cameraFrame->getYAxis();
        float3 zAxis = m_cameraFrame->getZAxis();

        mymath::rotate(xAxis, rad_v, xAxis, yAxis, zAxis);
        mymath::rotate(yAxis, rad_u, xAxis, yAxis, zAxis);

        m_cameraFrame->setXAxis(xAxis);
        m_cameraFrame->setYAxis(yAxis);
        m_cameraFrame->setZAxis(zAxis);
        
        if(m_cameraFrame->getForceUp()){
            m_cameraFrame->forceUpFrame();
        }

        m_cameraFrame->setIsTransformDirty(true);
    }

    virtual void move(const float step) override
    {
        const float3 position = m_cameraFrame->getFrom();
        
        m_cameraFrame->setPosition(position + step * m_cameraFrame->getZAxis());
        m_cameraFrame->setIsTransformDirty(true);
    }

};

inline void GLFWCameraWindow::enableInspectMode()
{
    m_inspectModeManip  = std::make_shared<InspectModeManip>(&m_cameraFrame);
    m_cameraFrameManip  = m_inspectModeManip;
}

inline void GLFWCameraWindow::enableFlyMode()
{
    m_flyModeManip = std::make_shared<FlyModeManip>(&m_cameraFrame);
    m_cameraFrameManip  = m_flyModeManip;
}

inline void GLFWCameraWindow::drawImGuiContents()
    {
        CameraFrame &cf = getCameraFramePtr();
        ImGui::Begin("Camera extrinsic");                          // Create a window called "Hello, world!" and append into it.

        // ImGui::Checkbox("Enable DoF", &show_demo_window);      // Edit bools storing our window open/close state
        
        ImGui::Text("Extrinsics");

        ImGui::Text("From:");
        float3 position = cf.getFrom();
        ImGui::Text("x %f, ", position.x);
        ImGui::SameLine();
        ImGui::Text("y %f, ", position.y);
        ImGui::SameLine();
        ImGui::Text("z %f", position.z);
        
        float3 upVector = cf.getUp();
        ImGui::Text("Up");
        ImGui::Text("x %f, ", upVector.x);
        ImGui::SameLine();
        ImGui::Text("y %f, ", upVector.y);
        ImGui::SameLine();
        ImGui::Text("z %f", upVector.z);

        float3 at = cf.getAt();
        ImGui::Text("At");
        ImGui::Text("x %f, ", at.x);
        ImGui::SameLine();
        ImGui::Text("y %f, ", at.y);
        ImGui::SameLine();
        ImGui::Text("z %f", at.z);
        
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / m_io->Framerate, m_io->Framerate);
        ImGui::End();
    }

#endif // GLFW_WINDOW_HPP_
