/*
 * MIT License
 *
 * Copyright (c) 2024 Shubham Panchal
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package android.example.clip

import android.clip.cpp.CLIPAndroid
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import org.junit.Before
import java.nio.ByteBuffer

/**
 *
 *
 * @author Shubham Panchal (github.com/shubham0204)
 */
@RunWith(AndroidJUnit4::class)
class CLIPAndroidInstrumentedTest {

    private val clipAndroid = CLIPAndroid()
    private val modelPath = "/data/local/tmp/clip_model.gguf"
    private val imagePaths = listOf(
        "/data/local/tmp/sample.png",
        "/data/local/tmp/sample_2.png"
    )

    @Before
    fun setup() {
        clipAndroid.load(modelPath, 1)
    }

    @Test
    fun getVisionHyperParameters_works() {
        val visionHyperParameters = clipAndroid.visionHyperParameters
        assertNotNull(visionHyperParameters)
    }

    @Test
    fun getTextHyperParameters_works() {
        val textHyperParameters = clipAndroid.textHyperParameters
        assertNotNull(textHyperParameters)
    }

    @Test
    fun imageEncode_works() {
        val imageBitmap = BitmapFactory.decodeFile(imagePaths[0])
        val width = imageBitmap.width
        val height = imageBitmap.height
        val imageBuffer = bitmapToByteBuffer(imageBitmap)
        val numThreads = 4
        val vectorDims = clipAndroid.visionHyperParameters.projectionDim
        val normalize = true
        val imageEmbedding = clipAndroid.encodeImage(imageBuffer, width, height, numThreads, vectorDims, normalize)
        assertNotNull(imageEmbedding)

        // If normalize = true, magnitude of embedding should be 1.0
        val magnitude = imageEmbedding.fold(0.0) { sqr, value -> sqr + value * value }
        assertEquals(1.0, magnitude, 0.0001)
    }

    @Test
    fun batchImageEncode_works() {
        val widths = IntArray(imagePaths.size)
        val heights = IntArray(imagePaths.size)
        val imageBuffers = imagePaths.mapIndexed { index, imagePath ->
            val imageBitmap = BitmapFactory.decodeFile(imagePath)
            widths[index] = imageBitmap.width
            heights[index] = imageBitmap.height
            bitmapToByteBuffer(imageBitmap)
        }
        val numThreads = 4
        val vectorDims = clipAndroid.visionHyperParameters.projectionDim
        val normalize = true
        val imageEmbeddings = clipAndroid.encodeImage(imageBuffers.toTypedArray(), widths, heights, numThreads, vectorDims, normalize)
        assertNotNull(imageEmbeddings)
        assertEquals(2, imageEmbeddings.size)

        // If normalize = true, magnitude of each embedding should be 1.0
        imageEmbeddings.forEach { embedding ->
            val magnitude = embedding.fold(0.0) { sqr, value -> sqr + value * value }
            assertEquals(1.0, magnitude, 0.0001)
        }
    }

    @Test
    fun getSimilarityScore_equalDims_works() {
        val vec1 = floatArrayOf(0.1f, 0.2f, 0.3f)
        val vec2 = floatArrayOf(0.2f, 0.3f, 0.4f)
        val similarityScore = clipAndroid.getSimilarityScore(vec1, vec2)
        assertEquals(0.20000002f, similarityScore)
    }

    @Test
    fun getSimilarityScore_unequalDims_throws() {
        val vec1 = floatArrayOf(0.1f, 0.2f, 0.3f)
        val vec2 = floatArrayOf(0.2f, 0.3f, 0.4f, 0.5f)
        assertThrows(IllegalArgumentException::class.java) {
            clipAndroid.getSimilarityScore(vec1, vec2)
        }
    }

    @Test
    fun textEncode_works() {
        val text = "a photo of a tiny dog"
        val numThreads = 4
        val vectorDims = clipAndroid.textHyperParameters.projectionDim
        val normalize = true
        val textEmbedding = clipAndroid.encodeText(text, numThreads, vectorDims, normalize)
        assertNotNull(textEmbedding)
    }

    @After
    fun clean() {
        clipAndroid.close()
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val imageBuffer = ByteBuffer.allocateDirect(width * height * 3)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                imageBuffer.put((pixel shr 16 and 0xFF).toByte())
                imageBuffer.put((pixel shr 8 and 0xFF).toByte())
                imageBuffer.put((pixel and 0xFF).toByte())
            }
        }
        return imageBuffer
    }

}