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
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer

class MainActivityViewModel: ViewModel() {

    val selectedImageState = mutableStateOf<Bitmap?>(null)
    val descriptionState = mutableStateOf("")
    val isLoadingModelState = mutableStateOf(true)
    val isInferenceRunning = mutableStateOf(false)
    val isShowingModelInfoDialogState = mutableStateOf(false)
    val similarityScoreState = mutableStateOf<Float?>(null)
    private val clipAndroid = CLIPAndroid()
    var visionHyperParameters: CLIPAndroid.CLIPVisionHyperParameters? = null
    var textHyperParameters: CLIPAndroid.CLIPTextHyperParameters? = null

    private val MODEL_PATH = "/data/local/tmp/clip_model_fp16.gguf"
    private val NUM_THREADS = 4
    private val VERBOSITY = 1

    init {
        CoroutineScope(Dispatchers.IO).launch {
            mainScope { isLoadingModelState.value = true }
            clipAndroid.load(MODEL_PATH, VERBOSITY)
            visionHyperParameters = clipAndroid.visionHyperParameters
            textHyperParameters = clipAndroid.textHyperParameters
            mainScope { isLoadingModelState.value = false }
        }
    }

    fun compare() {
        if (selectedImageState.value != null && descriptionState.value.isNotEmpty()) {
            CoroutineScope(Dispatchers.Default).launch {
                mainScope { isInferenceRunning.value = true }
                val textEmbedding = clipAndroid.encodeText(
                    descriptionState.value,
                    NUM_THREADS,
                    textHyperParameters?.projectionDim ?: 512,
                    true
                )
                val imageBuffer = bitmapToByteBuffer(selectedImageState.value!!)
                val imageEmbedding = clipAndroid.encodeImage(
                    imageBuffer,
                    selectedImageState.value!!.width,
                    selectedImageState.value!!.height,
                    NUM_THREADS,
                    visionHyperParameters?.projectionDim ?: 512,
                    true
                )
                mainScope {
                    similarityScoreState.value = clipAndroid.getSimilarityScore(textEmbedding, imageEmbedding)
                    isInferenceRunning.value = false
                }
            }
        }
    }

    fun showModelInfo() {
        isShowingModelInfoDialogState.value = true
    }

    fun reset() {
        selectedImageState.value = null
        descriptionState.value = ""
        similarityScoreState.value = null
        isInferenceRunning.value = false
    }

    override fun onCleared() {
        super.onCleared()
        clipAndroid.close()
    }

    private suspend fun mainScope(action: () -> Unit) {
        withContext(Dispatchers.Main) {
            action()
        }
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