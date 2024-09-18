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

#include <jni.h>
#include <android/log.h>
#include "clip.h"

#define TAG "clip-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" JNIEXPORT jlong JNICALL
Java_android_clip_cpp_CLIPAndroid_clipModelLoad__Ljava_lang_String_2I(
    JNIEnv *env,
    jobject,
    jstring file_path,
    jint verbosity
) {
    const char* file_path_chars = env -> GetStringUTFChars(file_path, nullptr);
    LOGi("Loading the model from %s", file_path_chars);
    const clip_ctx* ctx = clip_model_load(file_path_chars, verbosity);

    if (!ctx) {
        LOGe("Failed to load the model from %s", file_path_chars);
        return 0;
    }

    env -> ReleaseStringUTFChars(file_path, file_path_chars);
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_android_clip_cpp_CLIPAndroid_clipModelRelease__J(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr
) {
    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);
    clip_free(ctx);
}

extern "C" JNIEXPORT jobject JNICALL
Java_android_clip_cpp_CLIPAndroid_clipGetVisionHyperParameters__J(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr
) {
    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);
    clip_vision_hparams* vision_params = clip_get_vision_hparams(ctx);

    // Get the class CLIPVisionHyperParameters and and its constructor
    // Create a new object of the class with the constructor
    jclass cls = env -> FindClass("android/clip/cpp/CLIPAndroid$CLIPVisionHyperParameters");
    jmethodID constructor = env -> GetMethodID(cls, "<init>", "(IIIIIII)V");
    jvalue args[7];
    args[0].i = vision_params -> image_size;
    args[1].i = vision_params -> patch_size;
    args[2].i = vision_params -> hidden_size;
    args[3].i = vision_params -> projection_dim;
    args[4].i = vision_params -> n_intermediate;
    args[5].i = vision_params -> n_head;
    args[6].i = vision_params -> n_layer;
    jobject object = env -> NewObjectA(cls, constructor, args);

    return object;
}

extern "C" JNIEXPORT jobject JNICALL
Java_android_clip_cpp_CLIPAndroid_clipGetTextHyperParameters__J(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr
) {
    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);
    clip_text_hparams* text_params = clip_get_text_hparams(ctx);

    // Get the class CLIPTextHyperParameters and and its constructor
    // Create a new object of the class with the constructor
    jclass cls = env -> FindClass("android/clip/cpp/CLIPAndroid$CLIPTextHyperParameters");
    jmethodID constructor = env -> GetMethodID(cls, "<init>", "(IIIIIII)V");
    jvalue args[7];
    args[0].i = text_params -> n_vocab;
    args[1].i = text_params -> num_positions;
    args[2].i = text_params -> hidden_size;
    args[3].i = text_params -> projection_dim;
    args[4].i = text_params -> n_intermediate;
    args[5].i = text_params -> n_head;
    args[6].i = text_params -> n_layer;
    jobject object = env -> NewObjectA(cls, constructor, args);

    return object;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_android_clip_cpp_CLIPAndroid_clipTextEncode__JLjava_lang_String_2IIZ(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr,
        jstring text,
        jint n_threads,
        jint vector_dims,
        jboolean normalize
) {
    LOGi("Vector dims: %d", vector_dims);
    LOGi("Normalize: %d", normalize);
    LOGi("Number of threads: %d", n_threads);

    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);

    const char* text_chars = env -> GetStringUTFChars(text, nullptr);
    LOGi("Text: %s", text_chars);

    auto* tokens = new clip_tokens();
    clip_tokenize(ctx, text_chars, tokens);
    env -> ReleaseStringUTFChars(text, text_chars);
    float text_embedding[vector_dims];
    clip_text_encode(ctx, n_threads, tokens, text_embedding, normalize);

    jfloatArray result = env -> NewFloatArray(vector_dims);
    env -> SetFloatArrayRegion(result, 0, vector_dims, text_embedding);
    delete tokens;

    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_android_clip_cpp_CLIPAndroid_clipImageEncode__JLjava_nio_ByteBuffer_2IIIIZ(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr,
        jobject img_buffer,
        jint width,
        jint height,
        jint n_threads,
        jint vector_dims,
        jboolean normalize
) {
    LOGi("Vector dims: %d", vector_dims);
    LOGi("Normalize: %d", normalize);
    LOGi("Image size: %d x %d", width, height);

    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);
    auto* img = clip_image_u8_make();
    img -> nx = width;
    img -> ny = height;
    img -> data = reinterpret_cast<uint8_t*>(env -> GetDirectBufferAddress(img_buffer));
    img -> size = width * height * 3;

    auto* img_f32 = clip_image_f32_make();
    img_f32 -> nx = width;
    img_f32 -> ny = height;
    img_f32 -> data = new float[width * height * 3];
    img_f32 -> size = width * height * 3;
    clip_image_preprocess(ctx, img, img_f32);

    float image_embedding[vector_dims];
    clip_image_encode(ctx, n_threads, img_f32, image_embedding, normalize);
    jfloatArray result = env -> NewFloatArray(vector_dims);
    env -> SetFloatArrayRegion(result, 0, vector_dims, image_embedding);

    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_android_clip_cpp_CLIPAndroid_clipImageEncodeNoResize__JLjava_nio_ByteBuffer_2IIIIZ(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr,
        jobject img_buffer,
        jint width,
        jint height,
        jint n_threads,
        jint vector_dims,
        jboolean normalize
) {
    LOGi("Vector dims: %d", vector_dims);
    LOGi("Normalize: %d", normalize);
    LOGi("Image size: %d x %d", width, height);

    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);
    auto* img = clip_image_u8_make();
    img -> nx = width;
    img -> ny = height;
    img -> data = reinterpret_cast<uint8_t*>(env -> GetDirectBufferAddress(img_buffer));
    img -> size = width * height * 3;

    auto* img_f32 = clip_image_f32_make();
    img_f32 -> nx = width;
    img_f32 -> ny = height;
    img_f32 -> data = new float[width * height * 3];
    img_f32 -> size = width * height * 3;
    clip_image_preprocess(ctx, img, img_f32);

    float image_embedding[vector_dims];
    clip_image_encode(ctx, n_threads, img_f32, image_embedding, normalize);
    jfloatArray result = env -> NewFloatArray(vector_dims);
    env -> SetFloatArrayRegion(result, 0, vector_dims, image_embedding);

    return result;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_android_clip_cpp_CLIPAndroid_clipZeroShotClassify(
        JNIEnv *env,
        jobject thiz,
        jlong context_ptr,
        jint num_threads,
        jobject image_buffer,
        jint width,
        jint height,
        jobjectArray labels) {
    auto* ctx = reinterpret_cast<clip_ctx*>(context_ptr);

    auto* img = clip_image_u8_make();
    img -> nx = width;
    img -> ny = height;
    img -> data = reinterpret_cast<uint8_t*>(env -> GetDirectBufferAddress(image_buffer));
    img -> size = width * height * 3;

    int n_labels = env -> GetArrayLength(labels);
    const char* labels_cstr[n_labels];
    for (int i = 0; i < env -> GetArrayLength(labels); i++) {
        auto label = (jstring) env -> GetObjectArrayElement(labels, i);
        labels_cstr[i] = env -> GetStringUTFChars(label, nullptr);
    }

    float sorted_scores[n_labels];
    int sorted_indices[n_labels];
    clip_zero_shot_label_image(ctx, num_threads, img, labels_cstr, n_labels, sorted_scores, sorted_indices);

    float sorted_indices_fp[n_labels];
    for (int i = 0; i < n_labels; i++) {
        sorted_indices_fp[i] = (float)sorted_indices[i];
    }
    jfloatArray scores_indices = env -> NewFloatArray(2 * n_labels);
    env -> SetFloatArrayRegion(scores_indices, 0, n_labels, sorted_scores);
    env -> SetFloatArrayRegion(scores_indices, n_labels, n_labels, sorted_indices_fp);

    return scores_indices;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_android_clip_cpp_CLIPAndroid_clipBatchImageEncode__J_3Ljava_nio_ByteBuffer_2_3I_3IIIZ(
        JNIEnv *env,
        jobject,
        jlong ctx_ptr,
        jobjectArray arr_img_buffer,
        jintArray widths,
        jintArray heights,
        jint n_threads,
        jint vector_dims,
        jboolean normalize
) {
    auto* ctx = reinterpret_cast<clip_ctx*>(ctx_ptr);

    auto batch_size = env -> GetArrayLength(arr_img_buffer);
    auto* widths_buf = env -> GetIntArrayElements(widths, nullptr);
    auto* heights_buf = env -> GetIntArrayElements(heights, nullptr);

    LOGi("Batch size: %d", batch_size);
    LOGi("Vector dims: %d", vector_dims);
    LOGi("Normalize: %d", normalize);

    auto imgs_u8 = new clip_image_u8_batch();
    imgs_u8 -> data = new clip_image_u8[batch_size];
    imgs_u8 -> size = batch_size;

    auto* imgs_f32 = new clip_image_f32_batch();
    imgs_f32 -> data = new clip_image_f32[batch_size];
    imgs_f32 -> size = batch_size;
    for (int i = 0; i < batch_size; i++) {
        LOGi("Image %d: %d x %d", i, widths_buf[i], heights_buf[i]);
        auto* img = clip_image_u8_make();
        img -> nx = widths_buf[i];
        img -> ny = heights_buf[i];
        img -> data = reinterpret_cast<uint8_t*>(
                env -> GetDirectBufferAddress(
                        env -> GetObjectArrayElement(arr_img_buffer, i)
                       )
                );
        img -> size = widths_buf[i] * heights_buf[i] * 3;
        imgs_u8 -> data[i] = *img;

        auto* img_f32 = clip_image_f32_make();
        img_f32 -> nx = widths_buf[i];
        img_f32 -> ny = heights_buf[i];
        img_f32 -> data = new float[widths_buf[i] * heights_buf[i] * 3];
        img_f32 -> size = widths_buf[i] * heights_buf[i] * 3;
        imgs_f32 -> data[i] = *img_f32;
    }

    clip_image_batch_preprocess(ctx, n_threads, imgs_u8, imgs_f32);

    env -> ReleaseIntArrayElements(widths, widths_buf, 0);
    env -> ReleaseIntArrayElements(heights, heights_buf, 0);

    float image_embedding[vector_dims * batch_size];
    clip_image_batch_encode(ctx, n_threads, imgs_f32, image_embedding, normalize);

    jfloatArray vecs = env -> NewFloatArray(batch_size * vector_dims);
    env -> SetFloatArrayRegion(vecs, 0, batch_size * vector_dims, image_embedding);

    return vecs;
}



extern "C" JNIEXPORT jfloat JNICALL
Java_android_clip_cpp_CLIPAndroid_clipSimilarityScore___3F_3F(
        JNIEnv *env,
        jobject,
        jfloatArray vec1,
        jfloatArray vec2
) {
    auto vec1_len = env -> GetArrayLength(vec1);
    auto vec2_len = env -> GetArrayLength(vec2);

    auto vec1_buf = env -> GetFloatArrayElements(vec1, nullptr);
    auto vec2_buf = env -> GetFloatArrayElements(vec2, nullptr);

    float score = clip_similarity_score(
        reinterpret_cast<float*>(vec1_buf),
        reinterpret_cast<float*>(vec2_buf),
        vec1_len
    );

    env -> ReleaseFloatArrayElements(vec1, vec1_buf, 0);
    env -> ReleaseFloatArrayElements(vec2, vec2_buf, 0);

    return score;
}
