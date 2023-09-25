// main example to demonstrate usage of the API

#include "clip.h"
#include "common-clip.h"

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    app_params params;
    if (!app_params_parse(argc, argv, params, 1, 1)) {
        print_help(argc, argv, params, 1, 1);
        return 1;
    }

    const int64_t t_load_us = ggml_time_us();

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    const int64_t t_image_load_us = ggml_time_us();

    // load the image
    const char * img_path = params.image_paths[0].c_str();
    clip_image_u8 img0;
    if (!clip_image_load_from_file(img_path, &img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
        return 1;
    }

    const int64_t t_similarity_score = ggml_time_us();

    const char * text = params.texts[0].c_str();
    float score;

    if (!clip_compare_text_and_image(ctx, params.n_threads, text, &img0, &score)) {
        printf("Unable to compare text and image\n");
        clip_free(ctx);
        return 1;
    }

    const int64_t t_main_end_us = ggml_time_us();

    printf("%s: Similarity score = %2.3f\n", __func__, score);

    if (params.verbose >= 1) {
        printf("\n\nTimings\n");
        printf("%s: Model loaded in %8.2f ms\n", __func__, (t_image_load_us - t_load_us) / 1000.0);
        printf("%s: Image loaded in %8.2f ms\n", __func__, (t_similarity_score - t_image_load_us) / 1000.0);
        printf("%s: Similarity score calculated in %8.2f ms\n", __func__, (t_main_end_us - t_similarity_score) / 1000.0);
        printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
    }

    // Usage of the individual functions that make up clip_compare_text_and_image is demonstrated in the
    // `extract` example.

    clip_free(ctx);

    return 0;
}
