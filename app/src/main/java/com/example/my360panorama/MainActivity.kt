package com.example.my360panorama

import android.content.res.AssetManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.view.MotionEvent
import android.view.Surface
import androidx.appcompat.app.AppCompatActivity
import tv.danmaku.ijk.media.player.IjkMediaPlayer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10


class MainActivity : AppCompatActivity() {
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: PanoramaRenderer

    private var previousX = 0f
    private var previousY = 0f
    private var previousDistance = 0f

    private lateinit var ijkMediaPlayer: IjkMediaPlayer
    private var surfaceTexture: SurfaceTexture? = null
    private var surface: Surface? = null
    private var textureID: Int = 0

    companion object {
        init {
            System.loadLibrary("my360panorama")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        glSurfaceView = GLSurfaceView(this)
        glSurfaceView.setEGLContextClientVersion(3)

        renderer = PanoramaRenderer(assets, filesDir.absolutePath)
        glSurfaceView.setRenderer(renderer)

        glSurfaceView.setOnTouchListener { _, event ->
            when (event.pointerCount) {
                1 -> handleDrag(event)
                2 -> handlePinch(event)
            }
            true
        }

        setContentView(glSurfaceView)

        ijkMediaPlayer = IjkMediaPlayer()
    }

    override fun onResume() {
        super.onResume()
        try {
            // Initialize the SurfaceTexture and Surface
            surfaceTexture = renderer.createSurfaceTexture()
            surface = Surface(surfaceTexture)

            // Set the Surface to the IJKPlayer
            ijkMediaPlayer.setSurface(surface)
            ijkMediaPlayer.dataSource = filesDir.absolutePath + "/360panorama.mp4" // Your video file path
            ijkMediaPlayer.setOnPreparedListener { iMediaPlayer -> iMediaPlayer.start() }
            ijkMediaPlayer.prepareAsync()

            // Update the SurfaceTexture in the onDrawFrame method
            glSurfaceView.queueEvent {
                surfaceTexture?.setOnFrameAvailableListener {
                    glSurfaceView.requestRender()
                }
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onPause() {
        super.onPause()
        ijkMediaPlayer.release()
        surfaceTexture?.release()
        surface?.release()
    }

    private fun handleDrag(event: MotionEvent) {
        if (event.action == MotionEvent.ACTION_MOVE) {
            val deltaX = event.x - previousX
            val deltaY = event.y - previousY
            renderer.nativeHandleTouchDrag(renderer.nativeRendererPtr, deltaX, deltaY)
            previousX = event.x
            previousY = event.y
        } else if (event.action == MotionEvent.ACTION_DOWN) {
            previousX = event.x
            previousY = event.y
        }
    }

    private fun handlePinch(event: MotionEvent) {
        if (event.actionMasked == MotionEvent.ACTION_MOVE) {
            val dx = event.getX(0) - event.getX(1)
            val dy = event.getY(0) - event.getY(1)
            val distance = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()

            if (previousDistance != 0f) {
                val scaleFactor = distance / previousDistance
                if (scaleFactor != 0f) {
                    renderer.nativeHandlePinchZoom(renderer.nativeRendererPtr, 1 / scaleFactor)
                }
            }
            previousDistance = distance
        } else if (event.actionMasked == MotionEvent.ACTION_DOWN || event.actionMasked == MotionEvent.ACTION_POINTER_DOWN) {
            previousDistance = 0f
        }
    }

    private class PanoramaRenderer(assetManager: AssetManager, private val path: String) :
        GLSurfaceView.Renderer {

        var nativeRendererPtr: Long
        private var textureID: Int = 0
        private var surfaceTexture: SurfaceTexture? = null // Add this member variable

        init {
            nativeRendererPtr = nativeCreateRenderer(assetManager, path)
            if (nativeRendererPtr == 0L) {
                throw RuntimeException("Failed to create native renderer")
            }
            // Initialize the external texture (ID will be returned by native function)
            textureID = nativeCreateExternalTexture(nativeRendererPtr) // 生成一个外部纹理ID给到kt这边来
            surfaceTexture = SurfaceTexture(textureID) //
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            // Bind the texture and set parameters for external texture
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureID)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

            // Call native function to set up OpenGL
            nativeOnSurfaceCreated(nativeRendererPtr)
        }

        override fun onDrawFrame(gl: GL10?) {
            // Update the SurfaceTexture with the new video frame
            surfaceTexture?.updateTexImage() // Ensure this updates the texture

            // Draw the frame using the native renderer
            nativeOnDrawFrame(nativeRendererPtr)
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            nativeOnSurfaceChanged(nativeRendererPtr, width, height)
        }

        fun createSurfaceTexture(): SurfaceTexture {
            return surfaceTexture ?: throw RuntimeException("SurfaceTexture is not initialized")
        }

        // Native functions
        private external fun nativeCreateRenderer(assetManager: AssetManager, path: String): Long
        private external fun nativeCreateExternalTexture(rendererPtr: Long): Int
        private external fun nativeOnSurfaceCreated(rendererPtr: Long)
        private external fun nativeOnDrawFrame(rendererPtr: Long)
        private external fun nativeOnSurfaceChanged(rendererPtr: Long, width: Int, height: Int)
        external fun nativeHandleTouchDrag(rendererPtr: Long, deltaX: Float, deltaY: Float)
        external fun nativeHandlePinchZoom(rendererPtr: Long, scaleFactor: Float)
        external fun nativeProcessFrame(rendererPtr: Long, yuvData: ByteArray, width: Int, height: Int)
    }
}
