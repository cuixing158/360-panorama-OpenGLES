# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("my360panorama")

set (CMAKE_CXX_STANDARD 11)
add_definitions(-DNDEBUG)

set(common_sources
        ijkmedia/ijkj4a/j4a/j4a_allclasses.c
        ijkmedia/ijkj4a/j4a/j4a_base.c
        ijkmedia/ijkj4a/j4a/class/android/media/AudioTrack.c
        ijkmedia/ijkj4a/j4a/class/android/media/MediaCodec.c
        ijkmedia/ijkj4a/j4a/class/android/media/MediaFormat.c
        ijkmedia/ijkj4a/j4a/class/android/media/PlaybackParams.c
        ijkmedia/ijkj4a/j4a/class/android/os/Build.c
        ijkmedia/ijkj4a/j4a/class/android/os/Bundle.c
        ijkmedia/ijkj4a/j4a/class/java/nio/Buffer.c
        ijkmedia/ijkj4a/j4a/class/java/nio/ByteBuffer.c
        ijkmedia/ijkj4a/j4a/class/java/util/ArrayList.c
        ijkmedia/ijkj4a/j4a/class/tv/danmaku/ijk/media/player/misc/IMediaDataSource.c
        ijkmedia/ijkj4a/j4a/class/tv/danmaku/ijk/media/player/misc/IAndroidIO.c
        ijkmedia/ijkj4a/j4a/class/tv/danmaku/ijk/media/player/IjkMediaPlayer.c
        ijkmedia/ijkj4a/j4au/class/android/media/AudioTrack.util.c
        ijkmedia/ijkj4a/j4au/class/java/nio/ByteBuffer.util.c

        #ijkmedia/ijksdl/
        ijkmedia/ijksdl/ijksdl_aout.c
        ijkmedia/ijksdl/ijksdl_audio.c
        ijkmedia/ijksdl/ijksdl_egl.c
        ijkmedia/ijksdl/ijksdl_error.c
        ijkmedia/ijksdl/ijksdl_mutex.c
        ijkmedia/ijksdl/ijksdl_stdinc.c
        ijkmedia/ijksdl/ijksdl_thread.c
        ijkmedia/ijksdl/ijksdl_timer.c
        ijkmedia/ijksdl/ijksdl_vout.c
        ijkmedia/ijksdl/ijksdl_extra_log.c
        ijkmedia/ijksdl/gles2/color.c
        ijkmedia/ijksdl/gles2/common.c
        ijkmedia/ijksdl/gles2/renderer.c
        ijkmedia/ijksdl/gles2/renderer_rgb.c
        ijkmedia/ijksdl/gles2/renderer_yuv420p.c
        ijkmedia/ijksdl/gles2/renderer_yuv444p10le.c
        ijkmedia/ijksdl/gles2/shader.c
        ijkmedia/ijksdl/gles2/fsh/rgb.fsh.c
        ijkmedia/ijksdl/gles2/fsh/yuv420p.fsh.c
        ijkmedia/ijksdl/gles2/fsh/yuv444p10le.fsh.c
        ijkmedia/ijksdl/gles2/vsh/mvp.vsh.c
        ijkmedia/ijksdl/dummy/ijksdl_vout_dummy.c
        ijkmedia/ijksdl/ffmpeg/ijksdl_vout_overlay_ffmpeg.c
        ijkmedia/ijksdl/ffmpeg/abi_all/image_convert.c
        ijkmedia/ijksdl/android/android_audiotrack.c
        ijkmedia/ijksdl/android/android_nativewindow.c
        ijkmedia/ijksdl/android/ijksdl_android_jni.c
        ijkmedia/ijksdl/android/ijksdl_aout_android_audiotrack.c
        ijkmedia/ijksdl/android/ijksdl_aout_android_opensles.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediacodec_dummy.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediacodec_internal.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediacodec_java.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediacodec.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediadef.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediaformat_java.c
        ijkmedia/ijksdl/android/ijksdl_codec_android_mediaformat.c
        ijkmedia/ijksdl/android/ijksdl_vout_android_nativewindow.c
        ijkmedia/ijksdl/android/ijksdl_vout_android_surface.c
        ijkmedia/ijksdl/android/ijksdl_vout_overlay_android_mediacodec.c

        #ijkmedia/ijksoundtouch/
        ijkmedia/ijksoundtouch/soundtouch-jni.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/AAFilter.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/FIFOSampleBuffer.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/FIRFilter.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/cpu_detect_x86.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/sse_optimized.cpp
        ijkmedia/ijksoundtouch/source/SoundStretch/WavFile.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/RateTransposer.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/SoundTouch.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/InterpolateCubic.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/InterpolateLinear.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/InterpolateShannon.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/TDStretch.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/BPMDetect.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/PeakFinder.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/SoundTouch.cpp
        ijkmedia/ijksoundtouch/source/SoundTouch/mmx_optimized.cpp
        ijkmedia/ijksoundtouch/ijksoundtouch_wrap.cpp

        #ijkmedia/ijkyuv
        ijkmedia/ijkyuv/source/compare.cc
        ijkmedia/ijkyuv/source/compare_common.cc
        ijkmedia/ijkyuv/source/compare_posix.cc
        ijkmedia/ijkyuv/source/convert.cc
        ijkmedia/ijkyuv/source/convert_argb.cc
        ijkmedia/ijkyuv/source/convert_from.cc
        ijkmedia/ijkyuv/source/convert_from_argb.cc
        ijkmedia/ijkyuv/source/convert_to_argb.cc
        ijkmedia/ijkyuv/source/convert_to_i420.cc
        ijkmedia/ijkyuv/source/cpu_id.cc
        ijkmedia/ijkyuv/source/format_conversion.cc
        ijkmedia/ijkyuv/source/planar_functions.cc
        ijkmedia/ijkyuv/source/rotate.cc
        ijkmedia/ijkyuv/source/rotate_argb.cc
        ijkmedia/ijkyuv/source/rotate_mips.cc
        ijkmedia/ijkyuv/source/row_any.cc
        ijkmedia/ijkyuv/source/row_common.cc
        ijkmedia/ijkyuv/source/row_mips.cc
        ijkmedia/ijkyuv/source/row_posix.cc
        ijkmedia/ijkyuv/source/scale.cc
        ijkmedia/ijkyuv/source/scale_argb.cc
        ijkmedia/ijkyuv/source/scale_common.cc
        ijkmedia/ijkyuv/source/scale_mips.cc
        ijkmedia/ijkyuv/source/scale_posix.cc
        ijkmedia/ijkyuv/source/video_common.cc

        #ijkplayer
        ijkmedia/ijkplayer/ff_cmdutils.c
        ijkmedia/ijkplayer/ff_ffplay.c
        ijkmedia/ijkplayer/ff_ffpipeline.c
        ijkmedia/ijkplayer/ff_ffpipenode.c
        ijkmedia/ijkplayer/ijkmeta.c
        ijkmedia/ijkplayer/ijkplayer.c
        ijkmedia/ijkplayer/pipeline/ffpipeline_ffplay.c
        ijkmedia/ijkplayer/pipeline/ffpipenode_ffplay_vdec.c
        ijkmedia/ijkplayer/android/ffmpeg_api_jni.c
        ijkmedia/ijkplayer/android/ijkplayer_android.c
        ijkmedia/ijkplayer/android/ijkplayer_jni.c
        ijkmedia/ijkplayer/android/pipeline/ffpipeline_android.c
        ijkmedia/ijkplayer/android/pipeline/ffpipenode_android_mediacodec_vdec.c
        ijkmedia/ijkplayer/ijkavformat/allformats.c
        ijkmedia/ijkplayer/ijkavformat/ijklivehook.c
        ijkmedia/ijkplayer/ijkavformat/ijkmediadatasource.c
        ijkmedia/ijkplayer/ijkavformat/ijkio.c
        ijkmedia/ijkplayer/ijkavformat/ijkiomanager.c
        ijkmedia/ijkplayer/ijkavformat/ijkiocache.c
        ijkmedia/ijkplayer/ijkavformat/ijkioffio.c
        ijkmedia/ijkplayer/ijkavformat/ijkioandroidio.c
        ijkmedia/ijkplayer/ijkavformat/ijkioprotocol.c
        ijkmedia/ijkplayer/ijkavformat/ijkioapplication.c
        ijkmedia/ijkplayer/ijkavformat/ijkiourlhook.c
        ijkmedia/ijkplayer/ijkavformat/ijkasync.c
        ijkmedia/ijkplayer/ijkavformat/ijkurlhook.c
        ijkmedia/ijkplayer/ijkavformat/ijklongurl.c
        ijkmedia/ijkplayer/ijkavformat/ijksegment.c
        ijkmedia/ijkplayer/ijkavutil/ijkdict.c
        ijkmedia/ijkplayer/ijkavutil/ijkutils.c
        ijkmedia/ijkplayer/ijkavutil/ijkthreadpool.c
        ijkmedia/ijkplayer/ijkavutil/ijktree.c
        ijkmedia/ijkplayer/ijkavutil/ijkfifo.c
        ijkmedia/ijkplayer/ijkavutil/ijkstl.cpp

        prof.c
)

set(armeabi_v7a_sources
        ijkmedia/ijkyuv/source/compare_neon.cc
        ijkmedia/ijkyuv/source/rotate_neon.cc
        ijkmedia/ijkyuv/source/row_neon.cc
        ijkmedia/ijkyuv/source/scale_neon.cc
)

set(arm64_v8a_sources
        ijkmedia/ijkyuv/source/compare_neon64.cc
        ijkmedia/ijkyuv/source/rotate_neon64.cc
        ijkmedia/ijkyuv/source/row_neon64.cc
        ijkmedia/ijkyuv/source/scale_neon64.cc
)

# 参考：https://blog.csdn.net/qq_37857689/article/details/126548915
set(OpenCV_STATIC ON)
#set(OpenCV_DIR "E:/softwares/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native/jni")
#find_package(OpenCV REQUIRED)
add_library(opencv
        SHARED
        IMPORTED)
set_target_properties(opencv
        PROPERTIES IMPORTED_LOCATION
        ../../../../libs/${CMAKE_ANDROID_ARCH_ABI}/libopencv_java4.so)

find_library(log-lib log)
find_library(m-lib m)
find_library(z-lib z)
find_library(android-lib android)
find_library(EGL-lib EGL)
find_library(jnigraphics-lib jnigraphics)
find_library(GLES-lib GLESv2)
find_library(OpenSLES-lib OpenSLES)

if (${ANDROID_ABI} STREQUAL armeabi-v7a)
    add_library( # Sets the name of the library.
            ijkplayer
            SHARED
            ${common_sources}
            ${armeabi_v7a_sources}
            PanoramaRenderer.cpp
            Sphere.cpp
    )
else ()
    add_library( # Sets the name of the library.
            ijkplayer
            SHARED
            ${common_sources}
            ${arm64_v8a_sources}
            PanoramaRenderer.cpp
            Sphere.cpp
    )
endif ()


# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
#configure_file(${Project_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libijkffmpeg.so ${Project_BINARY_DIR}/libijkffmpeg.so COPYONLY)


#ffmpeg
add_library(ijkffmpeg-lib
        SHARED
        IMPORTED)
set_target_properties(ijkffmpeg-lib
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libijkffmpeg.so)
#sdl
add_library(ijksdl-lib
        SHARED
        IMPORTED)
set_target_properties(ijksdl-lib
        PROPERTIES IMPORTED_LOCATION
        libijksdl.so)


# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        ijkplayer

        # Links the target library to the log library
        # included in the NDK.
        ijkffmpeg-lib ${log-lib} ${m-lib} ${z-lib} ${android-lib} ${EGL-lib} ${GLES-lib} ${jnigraphics-lib} ${OpenSLES-lib}
        opencv
        EGL
        GLESv3)
target_include_directories(
        ijkplayer PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/ffmpeginclude
        ${CMAKE_SOURCE_DIR}/ffmpeginclude/${ANDROID_ABI}
        ${CMAKE_SOURCE_DIR}/ijkmedia
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkplayer
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkj4a
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijksdl
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijksoundtouch
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijksoundtouch/include
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkyuv
        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkyuv/include
        ${CMAKE_SOURCE_DIR}/include  # opencv
)
#include_directories(
#        ${CMAKE_SOURCE_DIR}
#        ${CMAKE_SOURCE_DIR}/ffmpeginclude
#        ${CMAKE_SOURCE_DIR}/ffmpeginclude/${ANDROID_ABI}
#        ${CMAKE_SOURCE_DIR}/ijkmedia
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkplayer
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkj4a
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijksdl
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijksoundtouch
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijksoundtouch/include
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkyuv
#        ${CMAKE_SOURCE_DIR}/ijkmedia/ijkyuv/include
#        ${CMAKE_SOURCE_DIR}/include  # opencv
#)



find_library(log-lib log)
find_library(android-lib android)
# 查找 OpenSL ES 库
find_library(OpenSLES_LIB OpenSLES)


#include_directories("E:/androidStudio_works/my360panorama/opencv/native/jni/include")
## ffmpeg prebuilt libraries
#include_directories(${CMAKE_SOURCE_DIR}/include)


#add_library(ffmpeg
#        SHARED
#        IMPORTED)
#set_target_properties(ffmpeg
#        PROPERTIES IMPORTED_LOCATION
#        ../../../../libs/${CMAKE_ANDROID_ARCH_ABI}/libffmpeg.so)

#if (${CMAKE_ANDROID_ARCH_ABI} MATCHES "armeabi-v7a")
#    include_directories(include/armeabi-v7a/libavcodec)
#    include_directories(include/armeabi-v7a/libavutil)
#    message("This is armeabi-v7a")
#elseif (${CMAKE_ANDROID_ARCH_ABI} MATCHES "arm64-v8a")
#    include_directories(include/arm64-v8a/libavcodec)
#    include_directories(include/arm64-v8a/libavutil)
#    message("This is arm64-v8a")
#endif ()




# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        stb_image.h
        PanoramaRenderer.cpp Sphere.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})


# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        #${OpenCV_LIBS}
        EGL
        GLESv3
        # List libraries link to the target library
        android
        log
        # ffmpeg libraries
#        ffmpeg
        ijkplayer
        opencv
        ${OpenSLES_LIB}
)