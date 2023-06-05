#include <iostream>
#include "clip.h"

int main()
{

    // load the image
    clip_image_u8 img0;
    if (!clip_image_load_from_file("/home/yusuf/clip-in-ggml/examples/grass.jpeg", img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, "grass.jpeg");
        return 1;
    }

    fprintf(stderr, "%s: loaded image (%d x %d)\n", __func__, img0.nx, img0.ny);

    // preprocess to f32
    clip_image_f32 img1;
    if (!clip_image_preprocess(img0, img1))
    {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);

    auto ctx = clip_model_load("/home/yusuf/clip-vit-base-patch32/ggml-vision-model-f16.bin");
    clip_image_encode(ctx, img1, 4);
    printf("done");
    return 0;
}