// this is a deadly simple example in C just to demonstrate usage.

#include "clip.h"
#include <stdbool.h>
#include <stdio.h>

int main() {
    char * model_path = "../../models/openai_clip-vit-base-patch32.q4_1.gguf";
    char * img_path = "../../tests/red_apple.jpg";
    char * text = "an apple";
    int n_threads = 4;
    int verbosity = 1;

    // Load CLIP model
    struct clip_ctx * ctx = clip_model_load(model_path, verbosity);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, model_path);
        return 1;
    }

    int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;

    // Load image from disk
    struct clip_image_u8 * img0 = clip_image_u8_make();
    if (!clip_image_load_from_file(img_path, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
        return 1;
    }

    // Preprocess image
    struct clip_image_f32 * img_res = clip_image_f32_make();
    if (!clip_image_preprocess(ctx, img0, img_res)) {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }

    // Encode image
    float img_vec[vec_dim];
    if (!clip_image_encode(ctx, n_threads, img_res, img_vec, true)) {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return 1;
    }

    // Tokenize text
    struct clip_tokens * tokens = NULL;
    clip_tokenize(ctx, text, tokens);

    // Encode text
    float txt_vec[vec_dim];
    if (!clip_text_encode(ctx, n_threads, tokens, txt_vec, true)) {
        fprintf(stderr, "%s: failed to encode text\n", __func__);
        return 1;
    }

    // Calculate image-text similarity
    float score = clip_similarity_score(img_vec, txt_vec, vec_dim);

    // Alternatively, you can replace the above steps with:
    //  float score;
    //  if (!clip_compare_text_and_image_c(ctx, n_threads, text, img0, &score)) {
    //      fprintf(stderr, "%s: failed to encode text\n", __func__);
    //      return 1;
    //  }

    printf("Similarity score = %2.3f\n", score);

    // Cleanup
    clip_free(ctx);

    return 0;
}
