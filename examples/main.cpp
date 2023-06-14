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

    // load the image
    clip_image_u8 img0;
    if (!clip_image_load_from_file("/mnt/c/users/yusuf/coding/banana-tests/boxing.jpeg", img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, "mysn.jpeg");
        return 1;
    }

    fprintf(stderr, "%s: loaded image (%d x %d)\n", __func__, img0.nx, img0.ny);

    // preprocess to f32
    clip_image_f32 img1;
    if (!clip_image_preprocess(&img0, &img1))
    {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);

    auto ctx = clip_model_load("/home/yusuf/clip-vit-base-patch32/ggml-model-f16.bin");
    float vec[512];
    auto tokens = clip_tokenize(ctx, "a red apple");
    clip_text_encode(ctx, 4, tokens, vec);

    // clip_image_encode(ctx, 4, img1, vec);
    write_floats_to_file(vec, 512, "/home/yusuf/clip-in-ggml/tests/pred.txt");

    printf("done\n");

    return 0;
}