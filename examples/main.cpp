// main example to demonstrate usage of the API

#include "clip.h"
#include "common-clip.h"

int main(int argc, char **argv)
{
    // TODO: replace this example with only clip_compare_text_and_image
    // and demonstrate usage of individual functions in zero-shot labelling and image search examples.

    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    app_params params;
    if (!app_params_parse(argc, argv, params))
    {
        print_help(argc, argv, params);
        return 1;
    }

    const int64_t t_load_us = ggml_time_us();

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx)
    {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    const int vec_dim = ctx->vision_model.hparams.projection_dim;

    const int64_t t_tokenize_us = ggml_time_us();

    std::string text = params.texts[0];
    auto tokens = clip_tokenize(ctx, text);

    const int64_t t_text_encode_us = ggml_time_us();

    float txt_vec[vec_dim];

    clip_text_encode(ctx, params.n_threads, tokens, txt_vec);

    const int64_t t_preprocess_us = ggml_time_us();

    // load the image
    std::string img_path = params.image_paths[0];
    clip_image_u8 img0;
    clip_image_f32 img_res;
    if (!clip_image_load_from_file(img_path, img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
        return 1;
    }

    clip_image_preprocess(ctx, &img0, &img_res);

    const int64_t t_image_encode_us = ggml_time_us();

    float img_vec[vec_dim];
    if (!clip_image_encode(ctx, params.n_threads, img_res, img_vec))
    {
        return 1;
    }

    const int64_t t_similarity_score = ggml_time_us();

    float score = clip_similarity_score(txt_vec, img_vec, vec_dim);
    printf("%s Similarity score = %2.3f\n", __func__, score);

    const int64_t t_main_end_us = ggml_time_us();

    if (params.verbose >= 1)
    {
        printf("\n\nTimings\n");
        printf("%s: Model loaded in %8.2f ms\n", __func__, (t_tokenize_us - t_load_us) / 1000.0);
        printf("%s: Text tokenized in %8.2f ms\n", __func__, (t_text_encode_us - t_tokenize_us) / 1000.0);
        printf("%s: Image loaded and preprocessed in %8.2f ms\n", __func__, (t_image_encode_us - t_preprocess_us) / 1000.0);
        printf("%s: Text encoded in %8.2f ms\n", __func__, (t_preprocess_us - t_text_encode_us) / 1000.0);
        printf("%s: Image encoded in %8.2f ms\n", __func__, (t_similarity_score - t_image_encode_us) / 1000.0);
        printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
    }

    // the above code can be replaced with a one-liner
    // clip_compare_text_and_image(ctx, 4, text, img0, &score);

    clip_free(ctx);

    return 0;
}