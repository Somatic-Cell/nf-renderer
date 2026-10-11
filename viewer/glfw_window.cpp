#include "glfw_window.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <chrono>
#include <iostream>
#include <stdexcept>



// --------------------------------------------
// コンストラクタ
// --------------------------------------------
MyGLFWWindow::MyGLFWWindow(const std::string & title, int initialWidth, int initialHeight)
{
    if (initialWidth <= 0 || initialHeight <= 0)
        throw std::invalid_argument("Initial window dimensions must be positive");
    glfwSetErrorCallback(
        [](int error, const char* description) -> void {
            fprintf(stderr, "Error: %s\n", description);
        }
    );
    if(!glfwInit()){
        exit(EXIT_FAILURE);
    }

#if defined(IMGUI_IMPL_OPENGL_E32)
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
#elif defined(__APPLE__)
        const char* glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
#else
        const char* glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

#endif
    m_handle = glfwCreateWindow(initialWidth, initialHeight, title.c_str(), NULL, NULL);
    if(!m_handle){
        glfwTerminate();
        exit(EXIT_FAILURE);
    }


    glfwSetWindowUserPointer(m_handle, this);
    glfwMakeContextCurrent(m_handle);
    glfwSwapInterval( 1 ); // ティアリング防止

    // ImGui のセットアップ
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    m_io = &ImGui::GetIO();
    (void)m_io;
    m_io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    m_io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    // io->WantCaptureMouse = true;

    // ImGui style
    ImGui::StyleColorsDark();

    // binding
    ImGui_ImplGlfw_InitForOpenGL(m_handle, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // --------------------------------------------
    // GLFW のイベントを MyGLFWWindow に渡すコールバック関数群
    // ImGUI による入力の補足への考慮も行う
    // --------------------------------------------
    // ウィンドウサイズが変更されたときに呼ばれるコールバック関数
    // クラス側の resize() を通じて描画領域などを更新
    glfwSetFramebufferSizeCallback(
        m_handle, 
        [](GLFWwindow* window, const int width, const int height){
            MyGLFWWindow* gw = static_cast<MyGLFWWindow*>(glfwGetWindowUserPointer(window));
            // assert(gw);
            if (!gw) return; // release ビルド時は assert() ではなくこちらがいい？
            gw->resize(make_int2(width, height));
        }
    );

    // マウスボタンが押された・離されたときに呼ばれるコールバック関数
    // ImGui がマウスを捕捉していない場合のみ、クラス側の mouseButton() を呼び出す
    glfwSetMouseButtonCallback(
        m_handle, 
        [](GLFWwindow* window, int button, int action, int mods){
            ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
            MyGLFWWindow * gw = static_cast<MyGLFWWindow*>(glfwGetWindowUserPointer(window));
            // assert(gw);
            if (!gw) return; // release ビルド時は assert() ではなくこちらがいい？
            gw->mouseButton(button, action, mods);
        }
    );

    
    // キーボード入力があったときに呼ばれるコールバック関数
    // ImGui がキーボードを捕捉していない場合のみ、クラス側の key() を呼び出す
    glfwSetKeyCallback(
        m_handle, 
        [](GLFWwindow* window, const int key, const int scanmode, const int action, const int mods){
            ImGui_ImplGlfw_KeyCallback(window, key, scanmode, action, mods);
            MyGLFWWindow* gw = static_cast<MyGLFWWindow*>(glfwGetWindowUserPointer(window));
            // assert(gw);
            if (!gw) return; // release ビルド時は assert() ではなくこちらがいい？
            if(action == GLFW_PRESS){
                gw->key(key, mods);
            }
        }
    );

    // マウスカーソルが移動したときに呼ばれるコールバック関数
    // ImGui がマウスを捕捉していない場合のみ、クラス側の mouseMotion() を呼び出す
    glfwSetCursorPosCallback(
        m_handle, 
        [](GLFWwindow* window, double x, double y){
            ImGui_ImplGlfw_CursorPosCallback(window, x, y);
            MyGLFWWindow * gw = static_cast<MyGLFWWindow*>(glfwGetWindowUserPointer(window));
            // assert(gw);
            if (!gw) return; // release ビルド時は assert() ではなくこちらがいい？
            gw->mouseMotion(make_int2((int)x, (int)y));
        }
    );
}

// --------------------------------------------
// デストラクタ
// --------------------------------------------
MyGLFWWindow::~MyGLFWWindow()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_handle);
    glfwTerminate();

}


void GLFWCameraWindow::run()
{
    int width, height;
    glfwGetFramebufferSize(m_handle, &width, &height);
    resize(make_int2(width, height));

    while(!glfwWindowShouldClose(m_handle)){
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        render();
        // save("smoke");
        const auto t1 = clock::now();

        // const auto sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        // std::cout << "render() wall time: " << sec << " s (" << (sec / 60.0) << " min)\n";

        glfwPollEvents();
        if(glfwGetWindowAttrib(m_handle, GLFW_ICONIFIED) != 0){
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start ImGUI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawImGuiContents();
        ImGui::Render();
            
        draw();        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        // glfwTerminate();
        glfwSwapBuffers(m_handle);
    }
}

