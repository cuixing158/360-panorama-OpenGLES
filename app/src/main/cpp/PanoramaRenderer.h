#ifndef PANORAMARENDERER_H
#define PANORAMARENDERER_H

#include <GLES2/gl2.h>
#include "opencv2/opencv.hpp"
#include "Sphere.h"

// FFmpeg，音视频解码，同步，播放
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/channel_layout.h>
#include <libavutil/time.h>
}

// 全景视频，多线程处理
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include<iostream>
#include<string>
#include <utility>

// 安卓& log
#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

// Android AudioTrack
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#include <pthread.h>
#include <sched.h>

#define LOG_TAG "PanoramaRenderer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#else
#include <stdio.h>
#define LOGE(FORMAT, ...) printf(FORMAT, ##__VA_ARGS__)
#endif


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

    void processFrame(const cv::Mat& rgbFrame);

private:
    GLuint loadShader(GLenum type, const char *shaderSrc);
    GLuint createProgram(const char *vertexSrc, const char *fragmentSrc);

    // 全景图像需要的函数
    GLuint loadTexture(const char *assetPath);

    // 全景图片和视频渲染
    GLuint shaderProgram;
    GLuint texture; // 全景图片纹理
    GLuint vao, vboVertices, vboTexCoords,vboIndices;
    SphereData* sphereData;
    cv::Mat frame;  // 全景视频帧
    int frameWidth,frameHeight; //全景视频帧的宽和高
    GLuint videoTexture; // 全景视频帧纹理
    cv::VideoCapture videoCapture; // 使用OpenCV解码

    std::mutex textureMutex; // 纹理线程锁
    AAssetManager* assetManager;
    std::string sharePath; // 共享文件夹，具有读写权限, JNI传入
    glm::mat4 projection;
    glm::mat4 view;
    float rotationX,rotationY,zoom;

    // 播放屏幕宽和高尺寸
    int widthScreen;
    int heightScreen;

};

#endif // PANORAMARENDERER_H
