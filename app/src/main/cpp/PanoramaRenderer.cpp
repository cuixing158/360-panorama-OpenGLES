#include "PanoramaRenderer.h"

PanoramaRenderer::PanoramaRenderer(AAssetManager *assetManager,std::string filepath)
    : shaderProgram(0), texture(0), videoTexture(0),vboVertices(0), vboTexCoords(0), vboIndices(0),
    sphereData(new SphereData(1.0f, 50, 50)), assetManager(assetManager),
    sharePath(std::move(filepath)),rotationX(0.0f), rotationY(0.0f), zoom(1.0f) ,
    widthScreen(800),heightScreen(800),frameReady(false),
    formatCtx(nullptr), videoCodecCtx(nullptr), // ffmpeg 音视频
    audioCodecCtx(nullptr), swrCtx(nullptr), stopPlayback(false),
    videoStreamIndex(-1), audioStreamIndex(-1),
    engineObject(nullptr), engineEngine(nullptr),
    outputMixObject(nullptr), playerObject(nullptr),audioBufferReady(false),
    playerPlay(nullptr), bufferQueue(nullptr), audioBuffer(nullptr),
    audioBufferSize(0){

    avformat_network_init();

    // Open the input file
    std::string mp4File = sharePath+"/360panodemo.mp4";
    videoCapture.open(mp4File);
    if (avformat_open_input(&formatCtx, mp4File.c_str(), nullptr, nullptr) != 0) {
        LOGE("Could not open input file");
        return;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        LOGE("Could not find stream information");
        return;
    }

    // Find the video and audio streams
    for (unsigned int i = 0; i < formatCtx->nb_streams; i++) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
        } else if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
        }
    }

    if (videoStreamIndex == -1 || audioStreamIndex == -1) {
        LOGE("Could not find video or audio stream");
        return;
    }

    // Find the video decoder
    videoStream = formatCtx->streams[videoStreamIndex];
    const AVCodec* videoCodec = avcodec_find_decoder(videoStream->codecpar->codec_id);
    if (!videoCodec) {
        LOGE("Could not find video codec");
        return;
    }

    videoCodecCtx = avcodec_alloc_context3(videoCodec);
    avcodec_parameters_to_context(videoCodecCtx, videoStream->codecpar);
    if (avcodec_open2(videoCodecCtx, videoCodec, nullptr) < 0) {
        LOGE("Could not open video codec");
        return;
    }

    frameWidth = videoCodecCtx->width;
    frameHeight = videoCodecCtx->height;

    // 创建音频解码上下文
    audioStream = formatCtx->streams[audioStreamIndex];
    const AVCodec* audioCodec = avcodec_find_decoder(audioStream->codecpar->codec_id);
    if (!audioCodec) {
        LOGE("Failed to find audio codec");
        return;
    }

    audioCodecCtx = avcodec_alloc_context3(audioCodec);
    if (avcodec_parameters_to_context(audioCodecCtx, audioStream->codecpar) < 0) {
        LOGE("Failed to copy audio codec parameters to context");
        return;
    }

    if (avcodec_open2(audioCodecCtx, audioCodec, nullptr) < 0) {
        LOGE("Failed to open audio codec");
        return;
    }

    // 初始化 SwrContext 用于音频重采样
    swrCtx = swr_alloc();

    // 使用 av_channel_layout_copy 设置输入和输出通道布局
    AVChannelLayout in_ch_layout;
    AVChannelLayout out_ch_layout;

    if (av_channel_layout_copy(&in_ch_layout, &audioStream->codecpar->ch_layout) < 0 ||
        av_channel_layout_copy(&out_ch_layout, &audioStream->codecpar->ch_layout) < 0) {
        LOGE("Failed to copy channel layout");
        return;
    }

    av_opt_set_chlayout(swrCtx, "in_chlayout", &in_ch_layout, 0);
    av_opt_set_chlayout(swrCtx, "out_chlayout", &out_ch_layout, 0);

    av_opt_set_int(swrCtx, "in_sample_rate", audioCodecCtx->sample_rate, 0);
    av_opt_set_int(swrCtx, "out_sample_rate", audioCodecCtx->sample_rate, 0);

    av_opt_set_sample_fmt(swrCtx, "in_sample_fmt", audioCodecCtx->sample_fmt, 0);
    av_opt_set_sample_fmt(swrCtx, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);

    if (swr_init(swrCtx) < 0) {
        LOGE("Failed to initialize the resampling context");
        return;
    }

    // 释放通道布局
    av_channel_layout_uninit(&in_ch_layout);
    av_channel_layout_uninit(&out_ch_layout);

    // Initialize OpenSL ES for audio playback
    LOGI("Initializing OpenSL ES for audio playback");
    initializeAudio();
    LOGI("OpenSL ES initialized successfully");

    // Start video and audio decoding threads
    LOGI("Starting video and audio decoding threads");
    try {
        videoThread = std::thread(&PanoramaRenderer::videoDecodingLoop, this);
        audioThread = std::thread(&PanoramaRenderer::audioDecodingLoop, this);
        LOGI("Threads started successfully");
    } catch (const std::exception &e) {
        LOGE("Error starting threads: %s", e.what());
    } catch (...) {
        LOGE("Unknown error starting threads");
    }
}

PanoramaRenderer::~PanoramaRenderer() {
    stopPlayback = true;
    if (videoThread.joinable()) {
        videoThread.join();
    }
    if (audioThread.joinable()) {
        audioThread.join();
    }
    shutdownAudio();

    if (videoCodecCtx) {
        avcodec_free_context(&videoCodecCtx);
    }
    if (audioCodecCtx) {
        avcodec_free_context(&audioCodecCtx);
    }
    if (formatCtx) {
        avformat_close_input(&formatCtx);
    }
    if (swrCtx) {
        swr_free(&swrCtx);
    }

    // Release OpenSL ES resources
    (*playerObject)->Destroy(playerObject);
    (*outputMixObject)->Destroy(outputMixObject);
    (*engineObject)->Destroy(engineObject);

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
    LOGI("onSurfaceCreated have successfully run.\n");
}

void PanoramaRenderer::videoDecodingLoop() {
    //LOGI("videoDecodingLoop has successfully initialized.\n");
    AVPacket packet;
    AVFrame* vFrame = av_frame_alloc();

    if (!vFrame) {
        LOGE("videoDecodingLoop: Failed to allocate AVFrame.\n");
        return;
    }

    SwsContext* swsCtx = sws_getContext(
            videoCodecCtx->width, videoCodecCtx->height, videoCodecCtx->pix_fmt,
            videoCodecCtx->width, videoCodecCtx->height, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!swsCtx) {
        LOGE("videoDecodingLoop: Failed to create SwsContext.\n");
        av_frame_free(&vFrame);
        return;
    }

    // Allocate buffer and RGB frame outside the loop to avoid frequent allocations
    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, videoCodecCtx->width, videoCodecCtx->height, 32);
    uint8_t* buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

    if (!buffer) {
        LOGE("videoDecodingLoop: Failed to allocate buffer for RGB conversion.\n");
        av_frame_free(&vFrame);
        sws_freeContext(swsCtx);
        return;
    }

    AVFrame* rgbFrame = av_frame_alloc();
    if (!rgbFrame) {
        LOGE("videoDecodingLoop: Failed to allocate RGB frame.\n");
        av_free(buffer);
        av_frame_free(&vFrame);
        sws_freeContext(swsCtx);
        return;
    }

    av_image_fill_arrays(rgbFrame->data, rgbFrame->linesize, buffer, AV_PIX_FMT_RGB24, videoCodecCtx->width, videoCodecCtx->height, 1);

    double lastPts = 0.0; // 控制播放速度
    while (!stopPlayback) {
        //LOGI("videoDecodingLoop: reading frame...");
        int ret = av_read_frame(formatCtx, &packet);

        if (ret < 0) {  // If we couldn't read, it's either an error or EOF
            if (ret == AVERROR_EOF) {
                //LOGI("videoDecodingLoop: End of file reached.\n");
                break;  // Exit the loop if EOF is reached
            }
            char errBuf[128];
            av_strerror(ret, errBuf, sizeof(errBuf));
            LOGE("videoDecodingLoop: Failed to read frame: %s\n", errBuf);
            continue;  // Try to read again
        }

        if (packet.stream_index == videoStreamIndex) {
            ret = avcodec_send_packet(videoCodecCtx, &packet);
            if (ret < 0) {
                char errBuf[128];
                av_strerror(ret, errBuf, sizeof(errBuf));
                LOGE("videoDecodingLoop: Failed to send packet to decoder: %s\n", errBuf);
                av_packet_unref(&packet);
                continue;  // Skip this packet and continue to the next
            }

            while (true) {
                ret = avcodec_receive_frame(videoCodecCtx, vFrame);
                if (ret == 0) {
                    // Convert the frame from YUV to RGB
                    sws_scale(swsCtx, vFrame->data, vFrame->linesize, 0, videoCodecCtx->height, rgbFrame->data, rgbFrame->linesize);

                    // 控制播放速度，通过帧间间隔时间休眠,与mp4文件固有帧率一致
                    double currentPts = (double)(vFrame->best_effort_timestamp) * av_q2d(videoStream->time_base);
                    if (lastPts != 0.0) {
                        double frameInterval = currentPts - lastPts;

                        if (frameInterval > 0) {
                            std::this_thread::sleep_for(std::chrono::duration<double>(frameInterval));
                        }
                    }
                    lastPts = currentPts;

                    {
                        std::lock_guard<std::mutex> lock(textureMutex);
                        cv::Mat matFrame = cv::Mat(videoCodecCtx->height, videoCodecCtx->width, CV_8UC3, rgbFrame->data[0], rgbFrame->linesize[0]);
                        cv::flip(matFrame, frame, 0);

                        if (!frame.empty()) {
                            currentVideoPts = currentPts;

                            frameReady = true;
                            frameReadyCondition.notify_one();  // Notify the rendering thread
                            LOGI("videoDecodingLoop: buffer copied and notified.");
                        }
                    }

                } else if (ret == AVERROR(EAGAIN)) {
                    //LOGI("videoDecodingLoop: No frame available yet, continuing...\n");
                    break;  // No more frames available yet
                } else {
                    char errBuf[128];
                    av_strerror(ret, errBuf, sizeof(errBuf));
                    LOGE("videoDecodingLoop: Failed to receive frame from video decoder: %s\n", errBuf);
                    break;  // Handle errors appropriately
                }
            }
        }
        av_packet_unref(&packet);  // Unreference the packet after processing
    }

    av_free(buffer);
    av_frame_free(&rgbFrame);
    av_frame_free(&vFrame);
    sws_freeContext(swsCtx);

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
}

//void PanoramaRenderer::audioDecodingLoop() {
//    LOGI("audioDecodingLoop has successfully initialized.\n");
//
//    AVPacket packet;
//    AVFrame* audioFrame = av_frame_alloc();
//    uint8_t* outBuffer = nullptr;
//    int maxBufferSize = 0;
//
//    double audioClock = 0.0;  // 用于追踪音频时间
//    double lastAudioPts = 0.0;  // 上一次的音频PTS
//    double startTime = av_gettime_relative() / 1000000.0;  // 开始的系统时间
//
//    while (!stopPlayback) {
//        if (av_read_frame(formatCtx, &packet) >= 0) {
//            if (packet.stream_index == audioStreamIndex) {
//                if (avcodec_send_packet(audioCodecCtx, &packet) >= 0) {
//                    while (avcodec_receive_frame(audioCodecCtx, audioFrame) >= 0) {
//                        int outSamples = av_rescale_rnd(
//                                swr_get_delay(swrCtx, audioCodecCtx->sample_rate) + audioFrame->nb_samples,
//                                44100, audioCodecCtx->sample_rate, AV_ROUND_UP);
//
//                        int outBufferSize = av_samples_alloc(&outBuffer, nullptr, 2, outSamples, AV_SAMPLE_FMT_S16, 0);
//
//                        outSamples = swr_convert(
//                                swrCtx, &outBuffer, outSamples,
//                                (const uint8_t**)audioFrame->data, audioFrame->nb_samples);
//
//                        if (outSamples > 0) {
//                            audioBufferSize = av_samples_get_buffer_size(
//                                    nullptr, 2, outSamples, AV_SAMPLE_FMT_S16, 1);
//
//                            if (audioBuffer == nullptr || audioBufferSize > maxBufferSize) {
//                                maxBufferSize = audioBufferSize;
//                                audioBuffer = (uint8_t*)av_malloc(maxBufferSize);
//                                if (!audioBuffer) {
//                                    LOGE("Failed to allocate audioBuffer.");
//                                    break;
//                                }
//                            }
//
//                            if (audioBuffer != nullptr) {
//                                std::memcpy(audioBuffer, outBuffer, audioBufferSize);
//
//                                // 获取当前音频帧的 PTS
//                                double audioPts = (double)(audioFrame->best_effort_timestamp) * av_q2d(audioStream->time_base);
//
//                                // 将当前的系统时间转换为相对音频的播放时间
//                                double currentTime = (av_gettime_relative() / 1000000.0) - startTime;
//
//                                // 音频时间（PTS）与当前系统时间进行比较，以同步播放
//                                if (audioPts > currentTime) {
//                                    double sleepTime = audioPts - currentTime;
//                                    if (sleepTime > 0.001) {
//                                        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
//                                    }
//                                }
//                                audioClock = audioPts;
//
//                                audioBufferReady = true;
//
//                                // 将音频数据放入缓冲区，准备播放
//                                (*bufferQueue)->Enqueue(bufferQueue, audioBuffer, audioBufferSize);
//                                LOGI("audioDecodingLoop: buffer copied and audio played.");
//                            }
//                        }
//
//                        av_freep(&outBuffer);
//                    }
//                }
//                av_packet_unref(&packet);
//            }
//        } else {
//            LOGI("audioDecodingLoop: end of stream.");
//            break;
//        }
//    }
//
//    av_frame_free(&audioFrame);
//    if (audioBuffer) {
//        av_free(audioBuffer);
//    }
//    LOGI("audioDecodingLoop has successfully run.\n");
//}

void PanoramaRenderer::audioDecodingLoop() {
    LOGI("audioDecodingLoop has successfully initialized.\n");

    AVPacket packet;
    AVFrame* audioFrame = av_frame_alloc();
    uint8_t* outBuffer = nullptr;
    int maxBufferSize = 0;

    while (!stopPlayback) {
        // Read audio packets
        if (av_read_frame(formatCtx, &packet) >= 0) {
            if (packet.stream_index == audioStreamIndex) {
                if (avcodec_send_packet(audioCodecCtx, &packet) >= 0) {
                    while (avcodec_receive_frame(audioCodecCtx, audioFrame) >= 0) {
                        // Convert audio frame to desired format
                        int outSamples = av_rescale_rnd(
                                swr_get_delay(swrCtx, audioCodecCtx->sample_rate) + audioFrame->nb_samples,
                                44100, audioCodecCtx->sample_rate, AV_ROUND_UP);

                        int outBufferSize = av_samples_alloc(&outBuffer, nullptr, 2, outSamples, AV_SAMPLE_FMT_S16, 0);

                        outSamples = swr_convert(
                                swrCtx, &outBuffer, outSamples,
                                (const uint8_t**)audioFrame->data, audioFrame->nb_samples);

                        if (outSamples > 0) {
                            audioBufferSize = av_samples_get_buffer_size(
                                    nullptr, 2, outSamples, AV_SAMPLE_FMT_S16, 1);

                            if (audioBuffer == nullptr || audioBufferSize > maxBufferSize) {
                                maxBufferSize = audioBufferSize;
                                audioBuffer = (uint8_t*)av_malloc(maxBufferSize);
                                if (!audioBuffer) {
                                    LOGE("Failed to allocate audioBuffer.");
                                    break;
                                }
                            }

                            if (audioBuffer != nullptr) {
                                std::memcpy(audioBuffer, outBuffer, audioBufferSize);

                                // Sync audio playback with video PTS (Presentation Time Stamp)
                                double audioPts = (double)(audioFrame->best_effort_timestamp) * av_q2d(audioStream->time_base);

                                {
                                    std::unique_lock<std::mutex> lock(audioMutex);
                                    while (audioPts > currentVideoPts && !stopPlayback) {
                                        audioCondVar.wait(lock);  // Wait until video is caught up
                                        // 在音频数据准备好后通知其他线程
                                        audioBufferReady = true;
                                        audioCondVar.notify_all();
                                    }

                                }
                                // Enqueue audio data to OpenSL ES
                                (*bufferQueue)->Enqueue(bufferQueue, audioBuffer, audioBufferSize);
                                LOGI("audioDecodingLoop: buffer copied and audio played.");
                            }
                        }

                        av_freep(&outBuffer);
                    }
                }
                av_packet_unref(&packet);
            }
        } else {
            LOGI("audioDecodingLoop: end of stream.");
            break;
        }
    }

    av_frame_free(&audioFrame);
    if (audioBuffer) {
        av_free(audioBuffer);
    }
    LOGI("audioDecodingLoop has successfully run.\n");
}

void PanoramaRenderer::initializeAudio() {
    LOGI("initializeAudio has successfully initialized.\n");

    // Create OpenSL ES engine and output mix
    slCreateEngine(&engineObject, 0, nullptr, 0, nullptr, nullptr);
    (*engineObject)->Realize(engineObject, SL_BOOLEAN_FALSE);
    (*engineObject)->GetInterface(engineObject, SL_IID_ENGINE, &engineEngine);
    (*engineEngine)->CreateOutputMix(engineEngine, &outputMixObject, 0, nullptr, nullptr);
    (*outputMixObject)->Realize(outputMixObject, SL_BOOLEAN_FALSE);

    // Create the audio player
    SLDataLocator_AndroidSimpleBufferQueue loc_bufq = {SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE, 1};

    // Set up the channel count
    int channels = (audioCodecCtx->ch_layout.nb_channels > 0) ? audioCodecCtx->ch_layout.nb_channels : 1;

    // Set OpenSL ES audio format
    SLDataFormat_PCM format_pcm = {
            SL_DATAFORMAT_PCM,
            (SLuint32)channels,
            SL_SAMPLINGRATE_44_1,
            SL_PCMSAMPLEFORMAT_FIXED_16,
            SL_PCMSAMPLEFORMAT_FIXED_16,
            (channels == 2) ? SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT : SL_SPEAKER_FRONT_CENTER,
            SL_BYTEORDER_LITTLEENDIAN
    };

    SLDataSource audioSrc = {&loc_bufq, &format_pcm};
    SLDataLocator_OutputMix loc_outmix = {SL_DATALOCATOR_OUTPUTMIX, outputMixObject};
    SLDataSink audioSnk = {&loc_outmix, nullptr};

    const SLInterfaceID ids[1] = {SL_IID_ANDROIDSIMPLEBUFFERQUEUE};
    const SLboolean req[1] = {SL_BOOLEAN_TRUE};
    (*engineEngine)->CreateAudioPlayer(engineEngine, &playerObject, &audioSrc, &audioSnk, 1, ids, req);
    (*playerObject)->Realize(playerObject, SL_BOOLEAN_FALSE);
    (*playerObject)->GetInterface(playerObject, SL_IID_PLAY, &playerPlay);
    (*playerObject)->GetInterface(playerObject, SL_IID_ANDROIDSIMPLEBUFFERQUEUE, &bufferQueue);

    // Register callback for buffer queue
    (*bufferQueue)->RegisterCallback(bufferQueue, [](SLAndroidSimpleBufferQueueItf bq, void* context) {
        auto* renderer = static_cast<PanoramaRenderer*>(context);

        // 使用线程锁的形式？// 在音频初始化完成后通知等待线程
        std::unique_lock<std::mutex> lock(renderer->audioMutex);
        renderer->audioBufferReady = true;
        renderer->audioCondVar.notify_all();
//        if (renderer->audioBuffer && renderer->audioBufferSize > 0) {
//            (*bq)->Enqueue(bq, renderer->audioBuffer, renderer->audioBufferSize);
//        }

        // 使用flag形式，引入变量audioBufferReady？
//        if (renderer->audioBufferReady) {
//            (*bq)->Enqueue(bq, renderer->audioBuffer, renderer->audioBufferSize);
//            renderer->audioBufferReady = false;  // Reset the flag
//        }
    }, this);

    // Start playing audio
    (*playerPlay)->SetPlayState(playerPlay, SL_PLAYSTATE_PLAYING);
    LOGI("initializeAudio has successfully run.\n");
}



void PanoramaRenderer::shutdownAudio() {
    LOGI("shutdownAudio have successfully initialized.\n");
    if (playerObject) {
        (*playerPlay)->SetPlayState(playerPlay, SL_PLAYSTATE_STOPPED);
        (*playerObject)->Destroy(playerObject);
    }
    if (outputMixObject) {
        (*outputMixObject)->Destroy(outputMixObject);
    }
    if (engineObject) {
        (*engineObject)->Destroy(engineObject);
    }

    if (audioBuffer) {
        free(audioBuffer);
    }
    LOGI("shutdownAudio have successfully run.\n");
}

void PanoramaRenderer::onDrawFrame() {
    LOGI("onDrawFrame have successfully initialized.\n");
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);

    projection = glm::perspective(glm::radians(55.0f), (float)widthScreen / (float)heightScreen, 0.1f, 100.0f);
    view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f) * zoom, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    view = glm::rotate(view, glm::radians(rotationX), glm::vec3(0.0f, 0.0f, 1.0f));
    view = glm::rotate(view, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));

    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");

    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    glBindVertexArray(vao);
    // 视频渲染
    {
        std::unique_lock<std::mutex> lock(textureMutex);
        if (frameReadyCondition.wait_for(lock, std::chrono::milliseconds(100), [this] { return frameReady; })) {
            LOGI("onDrawFrame, frame.empty():%d\n", frame.empty());
            if (!frame.empty()) {
                cv::imwrite(sharePath + "/dst_ondrawFrame.jpg", frame);
                glBindTexture(GL_TEXTURE_2D, videoTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
            }
            frameReady = false;
        } else {
            LOGI("onDrawFrame: No new frame ready, skipping draw.");
        }
    }

    // 音频同步
    {
        std::unique_lock<std::mutex> lock(audioMutex);
        // 等待音频缓冲区准备好
        audioCondVar.wait_for(lock,std::chrono::milliseconds(100), [this]() { return audioBufferReady; });
        if (audioBufferReady) {
            // Enqueue the audio buffer for playback
            (*bufferQueue)->Enqueue(bufferQueue, audioBuffer, audioBufferSize);
        }
        audioBufferReady = false;  // Reset the flag
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
