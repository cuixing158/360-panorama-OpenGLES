package com.example.my360panorama

import android.content.Context
import android.content.res.AssetManager
import android.graphics.SurfaceTexture
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.MotionEvent
import android.view.Surface
import androidx.appcompat.app.AppCompatActivity
import tv.danmaku.ijk.media.player.IjkMediaPlayer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

import android.Manifest
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import androidx.core.app.ActivityCompat
import android.graphics.ImageFormat
import java.nio.ByteBuffer
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat
import android.widget.Toast


class MainActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: PanoramaRenderer

    private var previousX = 0f
    private var previousY = 0f
    private var previousDistance = 0f

    private lateinit var ijkMediaPlayer: IjkMediaPlayer
    private var surfaceTexture: SurfaceTexture? = null
    private var surface: Surface? = null
    private var textureID: Int = 0

    private lateinit var sensorManager: SensorManager
    private var gyroSensor: Sensor? = null
    private var accSensor: Sensor? = null
    private var gameRotationVectorSensor: Sensor? = null // Add this line
    private var rotationSensor: Sensor? = null

    private var gyroX = 0f
    private var gyroY = 0f
    private var gyroZ = 0f
    private var accX = 0f
    private var accY = 0f
    private var accZ = 0f

    // android 前后2个摄像头
    private lateinit var cameraManager: CameraManager
    private lateinit var backCameraId: String
    private lateinit var frontCameraId: String
    private lateinit var backImageReader: ImageReader
    private lateinit var frontImageReader: ImageReader
    private lateinit var cameraDevice: CameraDevice
    private var frontImage: Image? = null
    private var backImage: Image? = null
    private val CAMERA_PERMISSION_REQUEST_CODE = 100

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

        // Initialize sensor manager and sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gameRotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR) // Add this line
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        // Initialize camera manager
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        // 在打开摄像头之前检查权限
        if (checkCameraPermission()) {
            // 如果权限已授予，打开摄像头
            openCameraPermission()
        } else {
            // 如果权限未授予，请求权限
            requestCameraPermission()
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)  // 确保调用父类的实现

        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // 权限被授予，打开摄像头
                openCamera()
            } else {
                // 权限被拒绝，显示提示信息
                Toast.makeText(this, "Camera permission is required to access the camera", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun openCameraPermission() {
        try {
            // 在调用 openCamera 前再次检查权限
            if (checkCameraPermission()) {
                // 获取后摄像头ID
                val cameraId = getBackCameraId()

                // 打开摄像头
                cameraManager.openCamera(cameraId, stateCallback, null)
            } else {
                // 如果没有权限，显示提示信息
                Toast.makeText(this, "Camera permission is required to open the camera", Toast.LENGTH_SHORT).show()
            }
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: SecurityException) {
            // 处理 SecurityException
            Toast.makeText(this, "Camera access denied. Please grant camera permission.", Toast.LENGTH_SHORT).show()
        }
    }

    // 定义 CameraDevice.StateCallback
    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            // 摄像头打开时的操作
            cameraDevice = camera
            // 你可以在这里开始设置摄像头的预览，或者捕获会话
        }

        override fun onDisconnected(camera: CameraDevice) {
            // 摄像头断开连接时的操作
            cameraDevice?.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            // 摄像头打开出错时的操作
            Toast.makeText(this@MainActivity, "Camera error: $error", Toast.LENGTH_SHORT).show()
        }
    }


    override fun onResume() {
        super.onResume()
        // Register the sensor listeners
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        accSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gameRotationVectorSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) } // Add this line
        rotationSensor?.let {sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }

        try {
            // Release old SurfaceTexture and Surface if they exist
            surface?.release()
            surfaceTexture?.release()

            // Initialize the SurfaceTexture and Surface
            surfaceTexture = renderer.createSurfaceTexture()
            surface = Surface(surfaceTexture)

//            // IjkMediaPlayer 时启用硬件解码器,避免警告 YUV420P转RGB，但会影响图像显示，实测下面3句没有图像显示
//            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_PLAYER, "mediacodec", 1)
//            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_PLAYER, "mediacodec-auto-rotate", 1)
//            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_PLAYER, "mediacodec-handle-resolution-change", 1)

            // Set the Surface to the IJKPlayer
            ijkMediaPlayer.setSurface(surface)
            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_FORMAT, "rtmp_buffer", 1000)
            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_FORMAT, "rtmp_live", 1)
            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_FORMAT, "timeout", 3000000) // 设置连接超时
            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_PLAYER, "packet-buffering", 0)
            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_FORMAT, "fflags", "nobuffer")
            ijkMediaPlayer.setOption(IjkMediaPlayer.OPT_CATEGORY_PLAYER, "loglevel", "verbose")


            ijkMediaPlayer.dataSource = filesDir.absolutePath + "/dualfish1920_960.mp4" // Your video file path
//            ijkMediaPlayer.dataSource = "rtmp://172.17.10.76:1935/live/streamkey"//"rtmp://192.168.2.22/live/test"
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

        // Open front and back camera
        //openCamera()
    }

    private fun openCamera() {
        try {
            backCameraId = getBackCameraId()
            frontCameraId = getFrontCameraId()

            // 创建ImageReader
            backImageReader = ImageReader.newInstance(1920, 1080, ImageFormat.YUV_420_888, 2)
            frontImageReader = ImageReader.newInstance(1920, 1080, ImageFormat.YUV_420_888, 2)

            // 设置前摄像头的ImageReader监听器
            frontImageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                frontImage = image // 更新前摄像头图像
                if (backImage != null) {
                    processImage(frontImage!!, backImage!!) // 同时处理前后图像
                }
            }, null)

            // 设置后摄像头的ImageReader监听器
            backImageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                backImage = image // 更新后摄像头图像
                if (frontImage != null) {
                    processImage(frontImage!!, backImage!!) // 同时处理前后图像
                }
            }, null)

            // 打开前后摄像头并设置CameraDevice的回调
            openCameraDevice(frontCameraId)
            openCameraDevice(backCameraId)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // 打开摄像头设备
    private fun openCameraDevice(cameraId: String) {
        try {
            val cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId)
            val cameraDirection = cameraCharacteristics.get(CameraCharacteristics.LENS_FACING)
            val isFrontCamera = cameraDirection == CameraCharacteristics.LENS_FACING_FRONT

            val imageReader = if (isFrontCamera) frontImageReader else backImageReader

            val stateCallback = object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    // 在摄像头打开时，开始创建会话并启动图像捕获
                    val captureRequestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    captureRequestBuilder.addTarget(imageReader!!.surface)

                    // 创建一个 CameraCaptureSession 来开始捕获图像
                    camera.createCaptureSession(listOf(imageReader.surface), object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            if (cameraDevice != null) {
                                val captureRequest = captureRequestBuilder.build()
                                session.setRepeatingRequest(captureRequest, null, null)
                            }
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            // 处理配置失败
                        }
                    }, null)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                }
            }

            cameraManager.openCamera(cameraId, stateCallback, null)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun getFrontCameraId(): String {
        // Get the front camera ID
        return cameraManager.cameraIdList.first {
            val characteristics = cameraManager.getCameraCharacteristics(it)
            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT
        }
    }

    private fun getBackCameraId(): String {
        // Get the back camera ID
        return cameraManager.cameraIdList.first {
            val characteristics = cameraManager.getCameraCharacteristics(it)
            characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }
    }

    private fun processImage(frontImage: Image, backImage: Image) {
        // 获取图像的宽度和高度
        val frontWidth = frontImage.width
        val frontHeight = frontImage.height
        val backWidth = backImage.width
        val backHeight = backImage.height

        // Ensure front and back images have the same dimensions
        if (frontWidth != backWidth || frontHeight != backHeight) {
            throw IllegalArgumentException("Front and back images must have the same dimensions.")
        }

        val frontBuffer: ByteBuffer = frontImage.planes[0].buffer
        val backBuffer: ByteBuffer = backImage.planes[0].buffer

        // Convert image data to ByteArray
        val frontImageData = ByteArray(frontBuffer.remaining())
        frontBuffer.get(frontImageData)

        val backImageData = ByteArray(backBuffer.remaining())
        backBuffer.get(backImageData)

        // Call JNI method to pass image data
        renderer.nativeProcessImage(renderer.nativeRendererPtr, frontImageData, backImageData,frontWidth,frontHeight)

        // Close the images to release resources
        frontImage.close()
        backImage.close()
    }

    override fun onPause() {
        super.onPause()
        ijkMediaPlayer.release()
        surfaceTexture?.release()
        surface?.release()
        // Unregister the sensor listeners
        sensorManager.unregisterListener(this)
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

    // Implement sensor event listener methods
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                gyroX = event.values[0]
                gyroY = event.values[1]
                gyroZ = event.values[2]
                // Call native function to pass gyro data
                //renderer.nativeOnGyroAccUpdate(renderer.nativeRendererPtr, gyroX, gyroY, gyroZ, 0f, 0f, 0f)
            }
            Sensor.TYPE_ACCELEROMETER -> {
                accX = event.values[0]
                accY = event.values[1]
                accZ = event.values[2]
                // Call native function to pass accelerometer data
                //renderer.nativeOnGyroAccUpdate(renderer.nativeRendererPtr,0f, 0f, 0f, accX, accY, accZ)
            }
            Sensor.TYPE_ROTATION_VECTOR -> {  // TYPE_GAME_ROTATION_VECTOR 这个类型没有地磁
                val rotationVector = event.values
                val quaternion = FloatArray(4)
                SensorManager.getQuaternionFromVector(quaternion, rotationVector)

                val w = quaternion[0]
                val x = quaternion[1]
                val y = quaternion[2]
                val z = quaternion[3]

                // Call native function to pass quaternion data
                renderer.nativeOnGameRotationUpdate(renderer.nativeRendererPtr, w, x, y, z,accX,accY,accZ)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle sensor accuracy changes if needed
    }

    private class PanoramaRenderer(assetManager: AssetManager, private val path: String) :
        GLSurfaceView.Renderer {

        var nativeRendererPtr: Long
        private var textureID: Int = 0
        private var surfaceTexture: SurfaceTexture? = null

        init {
            nativeRendererPtr = nativeCreateRenderer(assetManager, path)
            if (nativeRendererPtr == 0L) {
                throw RuntimeException("Failed to create native renderer")
            }
            textureID = nativeCreateExternalTexture(nativeRendererPtr)
            surfaceTexture = SurfaceTexture(textureID)
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureID)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

            nativeOnSurfaceCreated(nativeRendererPtr)
        }

        override fun onDrawFrame(gl: GL10?) {
            if (surfaceTexture != null) {
                try {
                    surfaceTexture?.updateTexImage()
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
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
        external fun nativeOnGyroAccUpdate(rendererPtr: Long,gyroX: Float, gyroY: Float, gyroZ: Float, accX: Float, accY: Float, accZ: Float)
        external fun nativeOnGameRotationUpdate(rendererPtr: Long, w: Float, x: Float, y: Float, z: Float,accX: Float, accY: Float, accZ: Float) // Add this line
        external fun nativeProcessImage(rendererPtr: Long, frontImageData: ByteArray, backImageData: ByteArray,frontWidth:Int, frontHeight:Int)
    }
}
