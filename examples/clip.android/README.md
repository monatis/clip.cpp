# clip.cpp - Android App

An Android app demonstrating the usage of `clip.cpp` library. It uses JNI and a Java wrapper class to interface with the functions provided in `clip.h`. 

## Setup

### Build the app

1. Open the current directory (`clip.cpp/examples/clip.android`) in Android Studio. An automatic Gradle build should start, if not click on the `Build` menu and select `Make Project`.

2. Connect the test-device to the computer and make sure that the device is recognized by the computer.

3. Download one of the GGUF models from the [HuggingFace repository](https://huggingface.co/my). For instance, if we download the `CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf` model, we need to push it to the test-device's file-system using `adb push`,

```commandline
adb push CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf /data/local/tmp/clip_model_fp16.gguf
```

4. In `MainActivityViewModel.kt`, ensure that the `modelPath` variable points to the correct model path on the test-device. For instance, if the model is pushed to `/data/local/tmp/clip_model_fp16.gguf`, then the `modelPath` variable should be set to `/data/local/tmp/clip_model_fp16.gguf`. Moreover, you can configure `NUM_THREADS` and `VERBOSITY` variables as well. 

```kotlin
private val MODEL_PATH = "/data/local/tmp/clip_model_fp16.gguf"
private val NUM_THREADS = 4
private val VERBOSITY = 1
```

5. Run the app on the test-device by clicking on the `Run` button (Shift + F10) in Android Studio.

### Run tests

This Android project also includes an instrumented test which would require an Android device (emulator or physical device). 

1. Open the current directory (`clip.cpp/examples/clip.android`) in Android Studio. An automatic Gradle build should start, if not click on the `Build` menu and select `Make Project`.

2. Connect the test-device to the computer and make sure that the device is recognized by the computer.

3. Download one of the GGUF models from the [HuggingFace repository](https://huggingface.co/my). For instance, if we download the `CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf` model, we need to push it to the test-device's file-system using `adb push`,

```commandline
adb push CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf /data/local/tmp/clip_model.gguf
```

4. Get two images from the internet and push them to the test-device's file-system using `adb push`,

```commandline
adb push image1.png /data/local/tmp/sample.png
adb push image2.png /data/local/tmp/sample_2.png
```

5. Navigate to `clip.cpp/examples/clip.android/clip/src/androidTest/java/android/example/clip/CLIPAndroidInstrumentedTest.kt`, right-click on the file, select `Run 'CLIPAndroidInstrumentedTest'` from the context menu.
