// zero-shot image labeling example

#include "clip.h"
#include "common-clip.h"

int main(int argc, char ** argv) {
    app_params params;
    if (!app_params_parse(argc, argv, params)) {
        print_help(argc, argv, params);
        return 1;
    }

    int n_labels = params.texts.size();
    if (n_labels < 2) {
        printf("%s: You must specify at least 2 texts for zero-shot labeling\n", __func__);
    }

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    // load the image
    const auto & img_path = params.image_paths[0].c_str();
    clip_image_u8 img0;
    clip_image_f32 img_res;
    if (!clip_image_load_from_file(img_path, &img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
        return 1;
    }

    const int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;

    clip_image_preprocess(ctx, &img0, &img_res);

    float img_vec[vec_dim];
    if (!clip_image_encode(ctx, params.n_threads, &img_res, img_vec, false)) {
        return 1;
    }

    // encode texts and compute similarities
    float txt_vec[vec_dim];
    float similarities[n_labels];

    for (int i = 0; i < n_labels; i++) {
        const auto & text = params.texts[i].c_str();
        auto tokens = clip_tokenize(ctx, text);
        clip_text_encode(ctx, params.n_threads, &tokens, txt_vec, false);
        similarities[i] = clip_similarity_score(img_vec, txt_vec, vec_dim);
    }

    // apply softmax and sort scores
    float sorted_scores[n_labels];
    int indices[n_labels];
    softmax_with_sorting(similarities, n_labels, sorted_scores, indices);

    for (int i = 0; i < n_labels; i++) {
        auto label = params.texts[indices[i]].c_str();
        float score = sorted_scores[i];
        printf("%s = %1.4f\n", label, score);
    }

    clip_free(ctx);

    return 0;
}
