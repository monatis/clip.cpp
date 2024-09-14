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

package android.clip.cpp;

import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


/**
 * CLIPAndroid provides an interface to the functions present in the clip.cpp library using JNI
 * for Android applications. It allows loading the CLIP model, encoding images and text, and
 * calculating the similarity score between two vectors.
 *
 * @author Shubham Panchal (github.com/shubham0204)
 */
public class CLIPAndroid {

    private long contextPtr; // holds clip_ctx* pointer

    static {
        System.loadLibrary("clip-android");
    }

    /**
     * Load the CLIP model from the specified file path.
     *
     * @param filePath Absolute path to the CLIP model file (.gguf file)
     * @param verbosity Verbosity level of the model loading process. 0: silent, 1: progress, 2: debug
     */
    public void load(String filePath, int verbosity) {
        if (!Paths.get(filePath).toFile().exists()) {
            throw new IllegalArgumentException("File not found: " + filePath);
        }
        long ptr = clipModelLoad(filePath, verbosity);
        if (ptr == 0) {
            throw new RuntimeException("Failed to load the model from " + filePath);
        } else {
            contextPtr = ptr;
        }
    }

    /**
     * Get the vision hyper-parameters of the loaded CLIP model.
     *
     * @return An instance of CLIPVisionHyperParameters
     */
    public CLIPVisionHyperParameters getVisionHyperParameters() {
        return clipGetVisionHyperParameters(contextPtr);
    }

    /**
     * Get the text hyper-parameters of the loaded CLIP model.
     *
     * @return An instance of CLIPTextHyperParameters
     */
    public CLIPTextHyperParameters getTextHyperParameters() {
        return clipGetTextHyperParameters(contextPtr);
    }

    /**
     * Encode an image into a vector using the CLIP model
     *
     * @param image      a `ByteBuffer` containing the image data in RGBRGB... format where each component is a byte
     *                   and the image is of size `width` x `height`
     * @param width      the width of the image
     * @param height     the height of the image
     * @param numThreads the number of threads to use for encoding
     * @param vectorDims the dimension of the output vector from CLIPVisionHyperParameters.projectionDim
     * @param normalize  whether to normalize the output vector
     * @return a float array (embedding) of size `vectorDims` containing the encoded image
     */
    public float[] encodeImage(ByteBuffer image, int width, int height, int numThreads, int vectorDims, boolean normalize) {
        return clipImageEncode(contextPtr, image, width, height, numThreads, vectorDims, normalize);
    }

    /**
     * Encode a batch of images into vectors using the CLIP model
     *
     * @param images an array of `ByteBuffer` containing the image data in RGBRGB... format where each component is a byte
     * @param widths an array containing the width of each image in `images`
     * @param heights an array containing the height of each image in `images`
     * @param numThreads the number of threads to use for encoding
     * @param vectorDims the dimension of the output vector from CLIPVisionHyperParameters.projectionDim
     * @param normalize whether to normalize the output vector
     * @return a list of float arrays (embeddings) of size `vectorDims` containing the encoded images
     */
    public List<float[]> encodeImage(ByteBuffer[] images, int[] widths, int[] heights, int numThreads, int vectorDims, boolean normalize) {
        if (images.length != widths.length || images.length != heights.length) {
            throw new IllegalArgumentException("images, widths, and heights must have the same length. Got "
                    + images.length + ", " + widths.length + ", " + heights.length);
        }
        float[] vectors = clipBatchImageEncode(contextPtr, images, widths, heights, numThreads, vectorDims, normalize);
        ArrayList<float[]> vectorsList = new ArrayList<>();
        for (int i = 0; i < vectors.length / vectorDims; i++) {
            float[] vec = new float[vectorDims];
            System.arraycopy(vectors, i * vectorDims, vec, 0, vectorDims);
            vectorsList.add(vec);
        }
        return vectorsList;
    }

    /**
     * Encode a text into a vector using the CLIP model
     *
     * @param text text to encode
     * @param numThreads number of threads to use for encoding
     * @param vectorDims the dimension of the output vector from CLIPTextHyperParameters.projectionDim
     * @param normalize whether to normalize the output vector
     * @return a float array (embedding) of size `vectorDims` containing the encoded text
     */
    public float[] encodeText(String text, int numThreads, int vectorDims, boolean normalize) {
        return clipTextEncode(contextPtr, text, numThreads, vectorDims, normalize);
    }

    /**
     * Calculate the similarity score between two vectors
     *
     * @param vec1 first vector
     * @param vec2 second vector
     * @return similarity score (cosine similarity) between the two vectors
     * @throws IllegalArgumentException if the vectors have different lengths
     */
    public float getSimilarityScore(float[] vec1, float[] vec2) {
        if (vec1.length != vec2.length) {
            throw new IllegalArgumentException("Vectors must have the same length. Got " + vec1.length + ", " + vec2.length);
        }
        return clipSimilarityScore(vec1, vec2);
    }

    /**
     * Releases the resources acquired by the CLIP model
     */
    public void close() {
        clipModelRelease(contextPtr);
    }

    private native long clipModelLoad(String filePath, int verbosity);

    private native void clipModelRelease(long model);

    private native CLIPVisionHyperParameters clipGetVisionHyperParameters(long contextPtr);

    private native CLIPTextHyperParameters clipGetTextHyperParameters(long contextPtr);

    private native float[] clipImageEncode(long contextPtr, ByteBuffer imageBuffer, int width, int height, int numThreads, int vectorDims, boolean normalize);

    private native float[] clipBatchImageEncode(long contextPtr, ByteBuffer[] imageBuffers, int[] widths, int[] heights, int numThreads, int vectorDims, boolean normalize);

    private native float[] clipTextEncode(long contextPtr, String text, int numThreads, int vectorDims, boolean normalize);

    private native float clipSimilarityScore(float[] vec1, float[] vec2);

    public static class CLIPVisionHyperParameters {
        public final int imageSize;
        public final int patchSize;
        public final int hiddenSize;
        public final int projectionDim;
        public final int nIntermediate;
        public final int nHead;
        public final int nLayer;

        public CLIPVisionHyperParameters(int imageSize, int patchSize, int hiddenSize, int projectionDim, int nIntermediate, int nHead, int nLayer) {
            this.imageSize = imageSize;
            this.patchSize = patchSize;
            this.hiddenSize = hiddenSize;
            this.projectionDim = projectionDim;
            this.nIntermediate = nIntermediate;
            this.nHead = nHead;
            this.nLayer = nLayer;
        }
    }

    public static class CLIPTextHyperParameters {
        public final int nVocab;
        public final int numPositions;
        public final int hiddenSize;
        public final int projectionDim;
        public final int nIntermediate;
        public final int nHead;
        public final int nLayer;

        public CLIPTextHyperParameters(int nVocab, int numPositions, int hiddenSize, int projectionDim, int nIntermediate, int nHead, int nLayer) {
            this.nVocab = nVocab;
            this.numPositions = numPositions;
            this.hiddenSize = hiddenSize;
            this.projectionDim = projectionDim;
            this.nIntermediate = nIntermediate;
            this.nHead = nHead;
            this.nLayer = nLayer;
        }
    }

}
