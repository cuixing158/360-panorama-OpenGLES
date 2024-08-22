#include "PanoramaRenderer.h"


// 钩子函数，从ff_ffplay.c中执行
 cv::Mat PanoramaRenderer::frame;
 std::mutex PanoramaRenderer::textureMutex;

void  processDecodedFrame(AVFrame* avFrame){
    PanoramaRenderer::processDecodedFrameImpl( avFrame);
}

 void  PanoramaRenderer::processDecodedFrameImpl(AVFrame* avFrame) {
// Check if the frame is in a format OpenCV understands
     if (avFrame->format != AV_PIX_FMT_BGR24 && avFrame->format != AV_PIX_FMT_RGB24) {
         // Convert frame to BGR24 or RGB24 format using sws_scale
         struct SwsContext* img_convert_ctx = sws_getContext(
                 avFrame->width, avFrame->height, (AVPixelFormat)avFrame->format,
                 avFrame->width, avFrame->height, AV_PIX_FMT_BGR24, SWS_BICUBIC,
                 nullptr, nullptr, nullptr);

         AVFrame* pFrameBGR = av_frame_alloc();
         int numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, avFrame->width, avFrame->height, 1);
         uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

         av_image_fill_arrays(pFrameBGR->data, pFrameBGR->linesize, buffer,
                              AV_PIX_FMT_BGR24, avFrame->width, avFrame->height, 1);

         sws_scale(img_convert_ctx, avFrame->data, avFrame->linesize, 0, avFrame->height,
                   pFrameBGR->data, pFrameBGR->linesize);

         // Now pFrameBGR contains the BGR24 formatted data.
         // Convert this frame to OpenCV Mat
         cv::Mat img(cv::Size(avFrame->width, avFrame->height), CV_8UC3, pFrameBGR->data[0], pFrameBGR->linesize[0]);
         {
             // Lock the textureMutex and update the frame
             std::lock_guard<std::mutex> lock(textureMutex);
             cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
             cv::flip(img,img,0);
             frame = img.clone();
         }

         // Free resources
         av_free(buffer);
         av_frame_free(&pFrameBGR);
         sws_freeContext(img_convert_ctx);
     } else {
         // If the frame is already in BGR24 or RGB24 format, directly create the Mat
         cv::Mat img(cv::Size(avFrame->width, avFrame->height), CV_8UC3, avFrame->data[0], avFrame->linesize[0]);
         {
             // Lock the textureMutex and update the frame
             std::lock_guard<std::mutex> lock(textureMutex);
             cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
             cv::flip(img,img,0);
             frame = img.clone();
         }
     }
}

PanoramaRenderer::PanoramaRenderer(AAssetManager *assetManager,std::string filepath)
    : shaderProgram(0), texture(0), videoTexture(0),vboVertices(0), vboTexCoords(0), vboIndices(0),
    sphereData(new SphereData(1.0f, 50, 50)), assetManager(assetManager),
    sharePath(std::move(filepath)),rotationX(0.0f), rotationY(0.0f), zoom(1.0f) ,
    widthScreen(800),heightScreen(800),ahrs(1.0f/60.0f){

    // Open the input file
    //std::string mp4File = sharePath+"/360panorama.mp4"; // 360panorama.mp4
    //videoCapture.open(mp4File);
}

PanoramaRenderer::~PanoramaRenderer() {

    delete sphereData;
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &texture);
    glDeleteTextures(1,&videoTexture);
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

// 全景图像需要的函数
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
//    bool issave = cv::imwrite(sharePath+"/AAA.jpg",img);
//    LOGE("IS save:%d\n",issave);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture;
}



void PanoramaRenderer::onSurfaceCreated() {
    LOGI("onSurfaceCreated have successfully Initialized.\n");
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

    ////////////////// 360全景图像////////////////////////////////////////////
    // texture = loadTexture("360panorama.jpg");
    // if (!texture) {
    //     LOGE("Failed to load texture");
    //     return;
    // }
    // glBindBuffer(GL_ARRAY_BUFFER, 0);// 解绑 VBO,360全景图像最好需要
    // glBindVertexArray(0); // 解绑VAO,360全景图像最好需要
    // glEnable(GL_DEPTH_TEST);//360全景图像最好需要
    //////////////////end of 360全景图像////////////////////////////////////////////

    glGenTextures(1, &videoTexture);
    glBindTexture(GL_TEXTURE_2D, videoTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameWidth, frameHeight,
                 0, GL_RGB, GL_UNSIGNED_BYTE,nullptr);
    //glGenerateMipmap(GL_TEXTURE_2D); //全景图像使用，但是视频渲染不使用 glGenerateMipmap,较少性能开销

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 启用深度测试，防止遮挡影响
    glEnable(GL_DEPTH_TEST);
    // 设置深度测试函数
    glDepthFunc(GL_LESS);
    LOGI("onSurfaceCreated have successfully run.\n");
}

//void PanoramaRenderer::videoDecodingLoop() {
//////////////////////////之前使用OpenCV解码视频播放/////////////////////////////////////
//    while (!stopPlayback) {
//        cv::Mat tempFrame;
//        if (!videoCapture.read(tempFrame)) {
//            LOGE("Failed to read frame from video");
//            break;
//        }
//
//        cv::flip(tempFrame,frame,0);
//        //cv::cvtColor(tempFrame, tempFrame, cv::COLOR_BGR2RGB);
//        // std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Adjust frame rate if needed
//        {
//            std::lock_guard<std::mutex> lock(textureMutex);
//            if (!frame.empty()) {
//                frameReady = true;
//                frameReadyCondition.notify_one();  // Notify the rendering thread
//                LOGI("videoDecodingLoop: buffer copied and notified.");
//            }
//        }
//    }
/////////////////////end of 之前使用OpenCV解码视频播放/////////////////////////////////////
   // LOGI("videoDecodingLoop has successfully run.\n");
//}

void PanoramaRenderer::onDrawFrame() {
    updateVideoFrame();

    unsigned codecVer = avcodec_version();
    const char *ffmpegVersion = av_version_info(); // 获取FFmpeg版本信息

    LOGI("FFmpeg version is: %s, avcodec version is: %d\n", ffmpegVersion, codecVer);

    LOGI("onDrawFrame have successfully initialized.\n");
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);

    float fovDeg = 60*zoom;
    if (fovDeg<30)
        fovDeg = 30;
    if (fovDeg>160)
        fovDeg = 160;
    projection = glm::perspective(glm::radians(fovDeg), (float)widthScreen / (float)heightScreen, 0.1f, 100.0f);
    view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f) , glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    view = glm::rotate(view, glm::radians(rotationX), glm::vec3(0.0f, 0.0f, 1.0f));
    view = glm::rotate(view, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));

    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    glBindVertexArray(vao);
    // 视频渲染
//  frame = cv::imread(sharePath+"/dst_videoDecodingLoop.jpg");
    {
        std::unique_lock<std::mutex> lock(textureMutex);
            LOGI("onDrawFrame, frame.empty():%d\n", frame.empty());
            if (!frame.empty()) {
                //cv::imwrite(sharePath + "/dst_ondrawFrame.jpg", frame);
                glBindTexture(GL_TEXTURE_2D, videoTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
            }
    }

    //    glActiveTexture(GL_TEXTURE0);  // 全景图像使用
//    glBindTexture(GL_TEXTURE_2D, texture);// 全景图像使用
    glDrawElements(GL_TRIANGLES, sphereData->getNumIndices(), GL_UNSIGNED_SHORT, 0);
    glBindVertexArray(0);  // Unbind VAO
    LOGI("onDrawFrame have successfully run.\n");
}

void PanoramaRenderer::onSurfaceChanged(int width, int height) {
    LOGI("onSurfaceChanged have successfully initialized.\n");
//    GLsizei len = std::min({width,height});
    widthScreen = width;
    heightScreen = height;
    glViewport(0, 0, width,height);
    LOGI("onSurfaceChanged have successfully run.\n");
}

void PanoramaRenderer::handleTouchDrag(float deltaX, float deltaY) {
    rotationY += deltaX * 0.1f;
    rotationX += deltaY * 0.1f;
}

void PanoramaRenderer::handlePinchZoom(float scaleFactor) {
    zoom *= scaleFactor;
}

void PanoramaRenderer::onGyroAccUpdate(float gyroX, float gyroY, float gyroZ,float accX,float accY,float accZ){

    glm::vec3 gyro(gyroX, gyroY, gyroZ);  // Gyroscope values in rad/s
    glm::vec3 acc(accX, accY, accZ);      // Accelerometer values in m/s²

    // Check if all components of gyro and acc are finite (not NaN or Inf)
    bool gyroValid1 = std::isfinite(gyro.x) && std::isfinite(gyro.y) && std::isfinite(gyro.z);
    bool gyroValid2 = !((fabs(gyro.x) < 1e-6)&&(fabs(gyro.y) < 1e-6)&&(fabs(gyro.z) < 1e-6));
    bool accValid1 = std::isfinite(acc.x) && std::isfinite(acc.y) && std::isfinite(acc.z);
    bool accValid2 = !((fabs(acc.x) < 1e-6)&&(fabs(acc.y) < 1e-6)&&(fabs(acc.z) < 1e-6));
    bool gyroValid = gyroValid1&&gyroValid2;
    bool accValid = accValid1 && accValid2;

    static glm::vec3 gyroData,accData;
    if (gyroValid)
    {
        gyroData = gyro;
    }
    if(accValid)
    {
        accData = acc;
    }

    bool gyroDataValid = !((fabs(gyroData.x) < 1e-6)&&(fabs(gyroData.y) < 1e-6)&&(fabs(gyroData.z) < 1e-6));
    bool accDataValid = !((fabs(accData.x) < 1e-6)&&(fabs(accData.y) < 1e-6)&&(fabs(accData.z) < 1e-6));
    if(gyroDataValid&&accDataValid){
        LOGI("gyro and acc data: gyro=(%.6f, %.6f, %.6f), acc=(%.6f, %.6f, %.6f)",
             gyroData.x, gyroData.y, gyroData.z, accData.x, accData.y, accData.z);
        ahrs.Update(gyroData, accData);
    }

//    if (gyroValid && accValid) {
//        LOGI("gyro or acc data: gyro=(%.6f, %.6f, %.6f), acc=(%.6f, %.6f, %.6f)",
//             gyro.x, gyro.y, gyro.z, acc.x, acc.y, acc.z);
////        ahrs.Update(gyroData, accData);
//    } else {
//        LOGE("Invalid gyro or acc data: gyro=(%.6f, %.6f, %.6f), acc=(%.6f, %.6f, %.6f)",
//             gyro.x, gyro.y, gyro.z, acc.x, acc.y, acc.z);
//    }

    glm::vec3 eulerAngles = ahrs.getEulerAngles();
    float azimuth = eulerAngles.y;  // Yaw
    float elevation = eulerAngles.x;  // Pitch
    // Adjust the rotation based on the gyroscope's rate of rotation
    rotationX = azimuth*180.0f/3.141592653f;
    rotationY = elevation*180.0f/3.141592653f;

    LOGI("rotationX:%.2f,rotationY:%.2f\n",rotationX,rotationY);

    // Clamp rotationX and rotationY to prevent excessive rotation
    // rotationX could be clamped to something like [-90, 90] degrees for a "look up/down" effect
//    rotationX = glm::clamp(rotationX, -90.0f, 90.0f);
    // rotationY could be cyclic, so no clamping is needed unless you want to restrict movement

}

// Create external texture for video frames
GLuint PanoramaRenderer::createExternalTexture() {
    GLuint textureId;
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, textureId);

    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Unbind texture
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);

    return textureId;
}

// Update video frame
void PanoramaRenderer::updateVideoFrame() {
    std::lock_guard<std::mutex> lock(textureMutex);  // Lock for thread safety

    if (frame.empty()) {
        return;  // No frame to update
    }

    // Update the OpenGL texture with the current video frame
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, videoTexture);

    // Upload frame data to texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

    glBindTexture(GL_TEXTURE_2D, 0);  // Unbind texture
}



// JNI Interfaces
extern "C" {
JNIEXPORT jlong JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeCreateRenderer(JNIEnv *env, jobject obj, jobject assetManager,jstring path) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    // Returns a pointer to an array of bytes representing the string
// in modified UTF-8 encoding. This array is valid until it is released
// by ReleaseStringUTFChars().
    const char * temp = env->GetStringUTFChars(path, nullptr);
    return reinterpret_cast<jlong>(new PanoramaRenderer(mgr,std::string(temp)));
}

JNIEXPORT jint JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeCreateExternalTexture(JNIEnv *env, jobject instance, jlong rendererPtr) {
    // Cast the renderer pointer to your PanoramaRenderer class
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    // Call the createExternalTexture function and return the texture ID
    return renderer->createExternalTexture();
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeOnDrawFrame(JNIEnv *env, jobject instance, jlong rendererPtr) {
    // Cast the renderer pointer to your PanoramaRenderer class
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);

    // Call the onDrawFrame function which should include the call to updateVideoFrame()
    renderer->onDrawFrame();
}

JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeOnSurfaceCreated(JNIEnv *env, jobject obj, jlong rendererPtr) {
    PanoramaRenderer *renderer = reinterpret_cast<PanoramaRenderer *>(rendererPtr);
    renderer->onSurfaceCreated();
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

// 处理从 Java 传递来的陀螺仪数据
JNIEXPORT void JNICALL
Java_com_example_my360panorama_MainActivity_00024PanoramaRenderer_nativeOnGyroAccUpdate(JNIEnv *env, jobject obj, jlong rendererPtr, jfloat gyroX, jfloat gyroY, jfloat gyroZ,
                                                                                        jfloat accX,jfloat accY,jfloat accZ) {
    PanoramaRenderer* renderer = reinterpret_cast<PanoramaRenderer*>(rendererPtr);
    if (renderer != nullptr) {
        renderer->onGyroAccUpdate(gyroX, gyroY, gyroZ,accX,accY,accZ);
    }
}

}
