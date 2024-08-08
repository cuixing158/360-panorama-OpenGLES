#ifndef PANORAMARENDERER_H
#define PANORAMARENDERER_H

#include <GLES3/gl3.h>
#include "opencv2/opencv.hpp"
#include "Sphere.h"

// 全景视频
#include <thread>
#include <atomic>
#include <mutex>

// 安卓& log
#ifdef __ANDROID__
// android
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>
#define LOG_TAG "PanoramaRenderer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <stdio.h>
#define LOGE(FORMAT, ...) printf(FORMAT, ##__VA_ARGS__)
#endif

// C++
#include<iostream>
#include<string>
#include <utility>

// glm高性能数学库
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

class PanoramaRenderer {
public:
    PanoramaRenderer(AAssetManager* assetManager,std::string filepath);
    ~PanoramaRenderer();

    void onSurfaceCreated();
    void onDrawFrame();
    void onSurfaceChanged(int width, int height);

    void handleTouchDrag(float deltaX, float deltaY);
    void handlePinchZoom(float scaleFactor);

private:
    GLuint loadShader(GLenum type, const char *shaderSrc);
    GLuint createProgram(const char *vertexSrc, const char *fragmentSrc);

    // 全景图像需要的函数
    GLuint loadTexture(const char *assetPath);

    // 全景视频需要的函数
    void videoDecodingLoop();

    GLuint shaderProgram;
    GLuint texture;
    GLuint vao, vboVertices, vboTexCoords,vboIndices;
    SphereData* sphereData;
    AAssetManager* assetManager;
    std::string sharePath;

    glm::mat4 projection;
    glm::mat4 view;
    float rotationX;
    float rotationY;
    float zoom;

    // 屏幕宽和高尺寸
    int widthScreen;
    int heightScreen;

    //视频解码(opencv 支持)
    std::thread videoThread;
    std::atomic<bool> stopVideo;
    cv::VideoCapture videoCapture;
    cv::Mat frame;
    GLuint videoTexture;
    std::mutex textureMutex;
};

#endif // PANORAMARENDERER_H
