package com.example.expression

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.opencv.core.Core

import org.opencv.core.Mat


class FacialExpressionRecognition(assetManager: AssetManager, context: Context, modelPath: String) {

    private var interpreter: Interpreter
    private var height = 0
    private var width = 0
    private var gpuDelegate: GpuDelegate
    private var cascadeClassifier: CascadeClassifier? = null

    init {
        val options = Interpreter.Options()
        gpuDelegate = GpuDelegate()
        options.addDelegate(gpuDelegate)
        options.setNumThreads(4)
        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
        try {
            val inputStream = context.resources.openRawResource(R.raw.haarcascade_frontalface_alt)
            val cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE)
            val mCascadeFile = File(cascadeDir, "face_detector.xml")
            val os = FileOutputStream(mCascadeFile)
            val buffer = ByteArray(4096)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                os.write(buffer, 0, bytesRead)
            }
            inputStream.close()
            os.close()
            cascadeClassifier = CascadeClassifier(mCascadeFile.absolutePath)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val assetFileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun recognizeFaces(mat: Mat): Mat {
        val a: Mat = mat.t()
        Core.flip(a, mat, 1)
        a.release()
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2RGB)
        height = gray.height()
        width = gray.width()
        val faces = MatOfRect()
        cascadeClassifier?.detectMultiScale(gray, faces, 1.2, 2,2, Size(height * 0.1, height * 0.1), Size())
        for(face in faces.toArray()) {
            Imgproc.rectangle(mat, face, Scalar(0.0,255.0,0.0,255.0), 2,Imgproc.LINE_AA,0)
            val crop = Rect(face.tl().x.toInt(), face.tl().y.toInt(), ((face.br().x - face.tl().x).toInt()), ((face.br().y - face.tl().y).toInt()))
            val cropMat = Mat(mat, crop)
            val bitmap = Bitmap.createBitmap(cropMat.cols(), cropMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(cropMat, bitmap)
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 48,48,false)
            val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
            val emotion = Array(1) { FloatArray(1) }
            Log.d("AshishSah",interpreter.toString())
            interpreter.run(byteBuffer, emotion)
            val emotionValue = emotion[0][0]
            val emotionText = getEmotionText(emotionValue)
            Imgproc.putText(mat, "$emotionText ( $emotionValue )", Point(face.tl().x + 10, face.tl().y + 20),1,1.5,
                Scalar(0.0,0.0,255.0,150.0),2
            )
        }
        val b: Mat = mat.t()
        Core.flip(b, mat, 0)
        b.release()
        return mat
    }

    private fun getEmotionText(emotionValue: Float): String {
        return if(emotionValue >= 0 && emotionValue < 0.5) {
            "Surprise"
        } else if(emotionValue >= 0.5 && emotionValue < 1.5) {
            "Fear"
        } else if(emotionValue >= 1.5 && emotionValue < 2.5) {
            "Angry"
        } else if(emotionValue >= 2.5 && emotionValue < 3.5) {
            "Neutral"
        } else if(emotionValue >= 3.5 && emotionValue < 4.5) {
            "Sad"
        } else if(emotionValue >= 4.5 && emotionValue < 5.5) {
            "Disgust"
        } else {
            "Happy"
        }
    }

    private fun convertBitmapToByteBuffer(scaledBitmap: Bitmap): ByteBuffer {
        val byteBuffer: ByteBuffer
        val sizeImage = 48
        byteBuffer = ByteBuffer.allocateDirect(4 * 1 * sizeImage * sizeImage * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(sizeImage * sizeImage)
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        var pixel = 0
        for (i in 0 until sizeImage) {
            for (j in 0 until sizeImage) {
                val value = intValues[pixel++]
                byteBuffer.putFloat((value shr 16 and 0xFF) / 255.0f)
                byteBuffer.putFloat((value shr 8 and 0xFF) / 255.0f)
                byteBuffer.putFloat((value and 0xFF) / 255.0f)
            }
        }
        return byteBuffer
    }

}