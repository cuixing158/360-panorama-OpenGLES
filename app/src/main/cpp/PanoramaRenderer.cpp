#include "PanoramaRenderer.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <jni.h>
//#include <unistd.h> // pwd 使用，不用可注释掉
#include<iostream>
#include<string>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#define LOG_TAG "PanoramaRenderer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

PanoramaRenderer::PanoramaRenderer(AAssetManager *assetManager,std::string filepath)
    : shaderProgram(0), texture(0), vboVertices(0), vboTexCoords(0), vboIndices(0),
    sphereData(new SphereData(1.0f, 32, 32)), assetManager(assetManager),
    sharePath(std::move(filepath)),rotationX(0.0f), rotationY(0.0f), zoom(1.0f) {
    //sharePath = std::move(filepath);
}

PanoramaRenderer::~PanoramaRenderer() {
    delete sphereData;
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &texture);
    glDeleteBuffers(1, &vboVertices);
    glDeleteBuffers(1, &vboTexCoords);
    glDeleteBuffers(1, &vboIndices);
    glDeleteVertexArrays(1, &vao);
}

GLuint PanoramaRenderer::loadShader(GLenum type, const char *shaderSrc) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &shaderSrc, nullptr);
    glCompileShader(shader);

    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char *infoLog = new char[infoLen];
            glGetShaderInfoLog(shader, infoLen, nullptr, infoLog);
            LOGE("Error compiling shader:\n%s\n", infoLog);
            delete[] infoLog;
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint PanoramaRenderer::createProgram(const char *vertexSrc, const char *fragmentSrc) {
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexSrc);
    if (!vertexShader) {
        LOGE("Failed to load vertex shader");
        return 0;
    }

    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentSrc);
    if (!fragmentShader) {
        LOGE("Failed to load fragment shader");
        glDeleteShader(vertexShader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint infoLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char *infoLog = new char[infoLen];
            glGetProgramInfoLog(program, infoLen, nullptr, infoLog);
            LOGE("Error linking program:\n%s\n", infoLog);
            delete[] infoLog;
        }
        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

GLuint PanoramaRenderer::loadTexture(const char *assetPath) {
    AAsset *asset = AAssetManager_open(assetManager, assetPath, AASSET_MODE_STREAMING);
    if (!asset) {
        LOGE("Failed to open asset %s", assetPath);
        return 0;
    }

    off_t assetLength = AAsset_getLength(asset);
    std::vector<unsigned char> fileData(assetLength);

    AAsset_read(asset, fileData.data(), assetLength);
    AAsset_close(asset);

    int width, height, nrChannels;
    // Decode the image using OpenCV
    cv::Mat img = cv::imdecode(fileData, cv::IMREAD_COLOR);
    if (img.empty()) {
        LOGE("Failed to load texture image from asset");
        return 0;
    }

    // Convert the image to RGB format if it is BGR
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    // Flip the image vertically
    cv::flip(img, img, 0);
    width = img.cols;
    height  = img.rows;
    //nrChannels = img.channels();
    //char cwd[1024];
    //std::string currPath;
    //if (getcwd(cwd, sizeof(cwd)) != nullptr)
        //__android_log_print(ANDROID_LOG_INFO, "", cwd);
        //currPath = std::string(cwd);
    bool issave = cv::imwrite(sharePath+"/AAA.jpg",img);
    LOGE("IS save:%d\n",issave);

    //stbi_uc *data = stbi_load_from_memory(fileData.data(), fileData.size(), &width, &height, &nrChannels, 0);
    //if (!data) {
    //    LOGE("Failed to load texture image from asset");
    //    return 0;
    //}

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //stbi_image_free(data);
    return texture;
}

void PanoramaRenderer::onSurfaceCreated() {
    const char *vertexShaderSource =
        "#version 300 es\n"
        "layout(location = 0) in vec3 aPos;\n"
        "layout(location = 1) in vec2 aTexCoord;\n"
        "out vec2 TexCoord;\n"
        "uniform mat4 projection;\n"
        "uniform mat4 view;\n"
        "void main()\n"
        "{\n"
        "   TexCoord = aTexCoord;\n"
        "   gl_Position = projection * view * vec4(aPos, 1.0);\n"
        "}\n";

    const char *fragmentShaderSource =
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec2 TexCoord;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D texture1;\n"
        "void main()\n"
        "{\n"
        "FragColor = texture(texture1, TexCoord);\n"
        "}\n";

    shaderProgram = createProgram(vertexShaderSource, fragmentShaderSource);
    if (!shaderProgram) {
        LOGE("Failed to create shader program");
        return;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vboVertices);
    glGenBuffers(1, &vboIndices);
    glGenBuffers(1, &vboTexCoords);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
    glBufferData(GL_ARRAY_BUFFER, sphereData->getNumVertices() * sizeof(GLfloat), sphereData->getVertices(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr); // 第5个参数stride也可以设置为0来让OpenGL决定具体步长是多少（只有当数值是紧密排列时才可用）
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vboTexCoords);
    glBufferData(GL_ARRAY_BUFFER, sphereData->getNumTexs() * sizeof(GLfloat), sphereData->getTexCoords(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*sizeof(GLfloat), nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboIndices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereData->getNumIndices() * sizeof(GLushort), sphereData->getIndices(), GL_STATIC_DRAW);

    texture = loadTexture("360panorama.jpg");
    if (!texture) {
        LOGE("Failed to load texture");
        return;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);// 解绑 VBO
    glBindVertexArray(0); // 解绑VAO

    glEnable(GL_DEPTH_TEST);
}

void PanoramaRenderer::onDrawFrame() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);

    projection = glm::perspective(glm::radians(55.0f), (float)800 / (float)800, 0.1f, 100.0f);
    // 小行星
    // 计算相机的位置，假设以 (0, 0, 0) 为中心点，围绕Y轴和X轴旋转
    // 使用球面坐标来实现环绕效果
//    float radius = 5.0f; // 距离，从球心到相机的距离
//
//    // 计算相机位置
//    float x = radius * sin(glm::radians(rotationY)) * cos(glm::radians(rotationX));
//    float y = radius * sin(glm::radians(rotationX));
//    float z = radius * cos(glm::radians(rotationY)) * cos(glm::radians(rotationX));

    view = glm::lookAt(glm::vec3(0.0f,0.0f,0.0f) , glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    view = glm::rotate(view, glm::radians(rotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));

    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawElements(GL_TRIANGLES, sphereData->getNumIndices(), GL_UNSIGNED_SHORT, 0);
    glBindVertexArray(0);// 解绑 VAO
}

void PanoramaRenderer::onSurfaceChanged(int width, int height) {
    GLsizei len = std::min({width,height});
    glViewport(0, 0, len,len);
}

void PanoramaRenderer::handleTouchDrag(float deltaX, float deltaY) {
    rotationY += deltaX * 0.1f;
    rotationX += deltaY * 0.1f;
}

void PanoramaRenderer::handlePinchZoom(float scaleFactor) {
    zoom *= scaleFactor;
}

// JNI Interfaces
extern "C" {
JNIEXPORT jlong JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeCreateRenderer(JNIEnv *env, jobject obj, jobject assetManager,jstring path) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    // Returns a pointer to an array of bytes representing the string
// in modified UTF-8 encoding. This array is valid until it is released
// by ReleaseStringUTFChars().
    const char * temp = env->GetStringUTFChars(path, NULL);
    return reinterpret_cast<jlong>(new PanoramaRenderer(mgr,std::string(temp)));
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeOnSurfaceCreated(JNIEnv *env, jobject obj, jlong rendererPtr) {
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    renderer->onSurfaceCreated();
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeOnDrawFrame(JNIEnv *env, jobject obj, jlong rendererPtr) {
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    renderer->onDrawFrame();
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeOnSurfaceChanged(JNIEnv *env, jobject obj, jlong rendererPtr, jint width, jint height) {
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    renderer->onSurfaceChanged(width, height);
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeHandleTouchDrag(JNIEnv *env, jobject obj, jlong rendererPtr, jfloat deltaX ,jfloat deltaY) {
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    renderer->handleTouchDrag(deltaX, deltaY);
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeHandlePinchZoom(JNIEnv *env, jobject obj, jlong rendererPtr, jfloat scaleFactor) {
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    renderer->handlePinchZoom(scaleFactor);
}
}
