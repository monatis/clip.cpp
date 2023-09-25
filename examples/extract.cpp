// extract vectors of texts or image
// TODO: support multiple text or multiple images
// optionally from a file

#include "clip.h"
#include "common-clip.h"

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    app_params params;
    if (!app_params_parse(argc, argv, params, 0, 0)) {
        print_help(argc, argv, params, 0, 0);
        return 1;
    }

    if (params.image_paths.size() >= 1 && params.texts.size() >= 1) {
        printf("Provide either --text or --image argument\n");
        return 1;
    }

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    const int64_t t_image_load_us = ggml_time_us();
    if (params.image_paths.size() >= 1) {
        // load the image
        const char * img_path = params.image_paths[0].c_str();
        clip_image_u8 img_input;
        if (!clip_image_load_from_file(img_path, &img_input)) {
            fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
            clip_free(ctx);
            return 1;
        }

        clip_image_f32 img_res;
        if (!clip_image_preprocess(ctx, &img_input, &img_res)) {
            printf("Unable to preprocess image\n");
            clip_free(ctx);
            return 1;
        }

        const int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
        int shape[2] = {1, vec_dim};
        float vec[vec_dim];
        clip_image_encode(ctx, params.n_threads, &img_res, vec, false);
        writeNpyFile("./img_vec.npy", vec, shape, 2);

        printf("Wrote to ./img_vec.npy\n");
    } else {
        const char * text = params.texts[0].c_str();
        clip_tokens tokens;
        if (!clip_tokenize(ctx, text, &tokens)) {
            printf("Unable to tokenize text\n");
            clip_free(ctx);
            return 1;
        }

        const int vec_dim = clip_get_text_hparams(ctx)->projection_dim;
        int shape[2] = {1, vec_dim};
        float vec[vec_dim];

        if (!clip_text_encode(ctx, params.n_threads, &tokens, vec, false)) {
            printf("Unable to encode text\n");
            clip_free(ctx);
            return 1;
        }

        writeNpyFile("./text_vec.npy", vec, shape, 2);
        printf("Wrote to ./text_vec.npy\n");
    }

    clip_free(ctx);

    return 0;
}
