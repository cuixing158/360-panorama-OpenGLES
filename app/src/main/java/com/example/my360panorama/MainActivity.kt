package com.example.my360panorama

import android.content.res.AssetManager
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import android.view.MotionEvent
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10


class MainActivity : AppCompatActivity() {
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: PanoramaRenderer

    private var previousX = 0f
    private var previousY = 0f
    private var previousDistance = 0f

    companion object {
        init {
            System.loadLibrary("my360panorama")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        glSurfaceView = GLSurfaceView(this)
        glSurfaceView.setEGLContextClientVersion(3)

        renderer = PanoramaRenderer(assets,filesDir.absolutePath)
        glSurfaceView.setRenderer(renderer)

        glSurfaceView.setOnTouchListener { _, event ->
            when (event.pointerCount) {
                1 -> handleDrag(event)
                2 -> handlePinch(event)
            }
            true
        }

        setContentView(glSurfaceView)
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
                if (scaleFactor!=0f) {
                    renderer.nativeHandlePinchZoom(renderer.nativeRendererPtr, 1/scaleFactor)
                }
            }
            previousDistance = distance
        } else if (event.actionMasked == MotionEvent.ACTION_DOWN || event.actionMasked == MotionEvent.ACTION_POINTER_DOWN) {
            previousDistance = 0f
        }
    }


    private class PanoramaRenderer(assetManager: AssetManager,private val path: String) : GLSurfaceView.Renderer {
        var nativeRendererPtr: Long

        init {
            nativeRendererPtr = nativeCreateRenderer(assetManager,path)
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            nativeOnSurfaceCreated(nativeRendererPtr)
        }

        override fun onDrawFrame(gl: GL10?) {
            nativeOnDrawFrame(nativeRendererPtr)
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            nativeOnSurfaceChanged(nativeRendererPtr, width, height)
        }


        private external fun nativeCreateRenderer(assetManager: AssetManager, path: String): Long
        private external fun nativeOnSurfaceCreated(rendererPtr: Long)
        private external fun nativeOnDrawFrame(rendererPtr: Long)
        private external fun nativeOnSurfaceChanged(rendererPtr: Long, width: Int, height: Int)
        external fun nativeHandleTouchDrag(rendererPtr: Long, deltaX: Float, deltaY: Float)
        external fun nativeHandlePinchZoom(rendererPtr: Long, scaleFactor: Float)
    }
}