package com.example.expression

import android.Manifest
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.expression.databinding.ActivityMainBinding
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var binding: ActivityMainBinding
    private val baseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when(status) {
                LoaderCallbackInterface.SUCCESS -> {
                    binding.cameraView.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }
    private var isFront  = 1
    private lateinit var mat1: Mat
    private var facialExpressionRecognition: FacialExpressionRecognition? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initDebug()
        binding = ActivityMainBinding.inflate(layoutInflater)

        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)== PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 111)
        }else{
            binding.cameraView.setCameraPermissionGranted()
        }

        setContentView(binding.root)

        binding.cameraView.visibility = CameraBridgeViewBase.VISIBLE
        binding.cameraView.setCvCameraViewListener(this)
        binding.cameraView.setCameraIndex(isFront)

        try {
            facialExpressionRecognition = FacialExpressionRecognition(assets, this, "model.tflite")
        } catch (e: Exception) {
            e.printStackTrace()
        }

        binding.flipCamera.setOnClickListener {
            isFront = if(isFront == 1) 0 else 1
            binding.cameraView.disableView()
            binding.cameraView.setCameraIndex(isFront)
            binding.cameraView.enableView()
        }

    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if(requestCode == 111) {
            if (grantResults.isNotEmpty()
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                binding.cameraView.setCameraPermissionGranted()
            } else {
                Toast.makeText(this, "Camera permission required.", Toast.LENGTH_LONG).show()
                this.finish()
            }
        } else{
            super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    override fun onResume() {
        super.onResume()
        if(OpenCVLoader.initDebug()) {
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }else{
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this, baseLoaderCallback)
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mat1 = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        mat1.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        mat1 = inputFrame!!.rgba()
        if(isFront == 1) {
            Core.rotate(mat1, mat1, Core.ROTATE_180)
            Core.flip(mat1, mat1,0)
        }
        mat1 = facialExpressionRecognition?.recognizeFaces(mat1)!!
        return mat1
    }

    override fun onPause() {
        super.onPause()
        binding.cameraView.disableView()
    }

}