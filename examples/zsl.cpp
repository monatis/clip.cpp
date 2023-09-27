// zero-shot image labeling example

#include "clip.h"
#include "common-clip.h"

int main(int argc, char ** argv) {
    app_params params;
    if (!app_params_parse(argc, argv, params, 2, 1)) {
        print_help(argc, argv, params, 2, 1);
        return 1;
    }

    const size_t n_labels = params.texts.size();
    if (n_labels < 2) {
        printf("%s: You must specify at least 2 texts for zero-shot labeling\n", __func__);
    }

    const char * labels[n_labels];
    for (size_t i = 0; i < n_labels; ++i) {
        labels[i] = params.texts[i].c_str();
    }

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    // load the image
    const auto & img_path = params.image_paths[0].c_str();
    clip_image_u8 input_img;
    if (!clip_image_load_from_file(img_path, &input_img)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
        return 1;
    }

    float sorted_scores[n_labels];
    int sorted_indices[n_labels];
    if (!clip_zero_shot_label_image(ctx, params.n_threads, &input_img, labels, n_labels, sorted_scores, sorted_indices)) {
        fprintf(stderr, "Unable to apply ZSL\n");
        return 1;
    }

    for (int i = 0; i < n_labels; i++) {
        auto label = labels[sorted_indices[i]];
        float score = sorted_scores[i];
        printf("%s = %1.4f\n", label, score);
    }

    clip_free(ctx);

    return 0;
}
