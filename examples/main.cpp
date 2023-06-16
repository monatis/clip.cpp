#include <iostream>
#include "clip.h"

void write_floats_to_file(float *array, int size, char *filename)
{
    // Open the file for writing.
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file: %s\n", filename);
        return;
    }

    // Write the float values to the file.
    for (int i = 0; i < size; i++)
    {
        fprintf(file, "%f\n", array[i]);
    }

    // Close the file.
    fclose(file);
}

int main()
{
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    const int64_t t_load_us = ggml_time_us();

    auto ctx = clip_model_load("/home/yusuf/clip-vit-base-patch32/ggml-model-f16.bin");

    const int64_t t_tokenize_us = ggml_time_us();
    float vec[512];

    std::string text = "a dog";
    auto tokens = clip_tokenize(ctx, text);

    const int64_t t_encode_us = ggml_time_us();

    clip_text_encode(ctx, 4, tokens, vec);

    const int64_t t_main_end_us = ggml_time_us();

    printf("\n\nTimings\n");
    printf("%s: Model loaded in %8.2f ms\n", __func__, (t_tokenize_us - t_load_us) / 1000.0);
    printf("%s: Input tokenized in %8.2f ms\n", __func__, (t_encode_us - t_tokenize_us) / 1000.0);
    printf("%s: Input encoded in %8.2f ms\n", __func__, (t_main_end_us - t_encode_us) / 1000.0);
    printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
    // write_floats_to_file(vec, 512, "../tests/pred.txt");

    std::string img_path = "../tests/red_apple.jpg";

    // load the image
    clip_image_u8 img0;
    if (!clip_image_load_from_file(img_path, img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
        return 1;
    }

    float score;
    clip_compare_text_and_image(ctx, 4, text, img0, &score);
    printf("%s Similarity score = %2.3f\n", __func__, score);

    return 0;
}