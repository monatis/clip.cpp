#ifndef CLIP_H
#define CLIP_H

#include "ggml/ggml.h"

#ifdef __cplusplus
#include <cstring>
#include <map>
#include <stdbool.h>
#include <thread>
#include <vector>
#endif

// TODO: make the API in C
#ifdef __cplusplus
extern "C" {
#endif

struct clip_text_hparams {
    int32_t n_vocab;
    int32_t num_positions;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
};

struct clip_vision_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
};

// default hparams for ViT-B/32
#ifdef __cplusplus
namespace DefaultHParams {
static constexpr clip_text_hparams text = {49408, 77, 512, 2048, 512, 8, 12};
static constexpr clip_vision_hparams vision = {224, 32, 768, 3072, 512, 12, 12};
} // namespace DefaultHParams
#else
#define DEFAULT_TEXT_HPARAMS                                                                                                   \
    { 49408, 77, 512, 2048, 512, 8, 12 }
#define DEFAULT_VISION_HPARAMS                                                                                                 \
    { 224, 32, 768, 3072, 512, 12, 12 }
#endif

//
// Vocab utils
//

#ifndef __cplusplus
typedef int32_t clip_vocab_id;
#endif

struct clip_vocab {
#ifdef __cplusplus
    using id = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    void add_special_token(const std::string & token);
#endif
};

// C-compatible structure representing a vector of IDs
struct clip_tokens {
#ifdef __cplusplus
    clip_vocab::id * data;
#else
    clip_vocab_id * data;
#endif
    size_t size;
};

//
// clip layers
//

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_text_model {
    struct clip_text_hparams hparams;

    // embeddings
    struct ggml_tensor * token_embeddings;
    struct ggml_tensor * position_embeddings;

#ifdef __cplusplus
    std::vector<clip_layer> layers;
#endif

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

#ifdef __cplusplus
    std::map<std::string, struct ggml_tensor *> tensors;
#endif
};

struct clip_vision_model {
    struct clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

#ifdef __cplusplus
    std::vector<clip_layer> layers;
#endif

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

#ifdef __cplusplus
    std::map<std::string, struct ggml_tensor *> tensors;
#endif
};

struct clip_ctx * clip_model_load(const char * fname, const int verbosity);

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct clip_buffer {
#ifdef __cplusplus
    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~clip_buffer() { delete[] data; }
#endif
};

struct clip_ctx {
    struct clip_text_model text_model;
    struct clip_vision_model vision_model;
    struct clip_vocab vocab;
#ifdef __cplusplus
    int32_t use_gelu = 0;
    int32_t ftype = 1;
#else
    int32_t use_gelu;
    int32_t ftype;
#endif
    struct ggml_context * ctx;
    struct clip_buffer buf_compute;
};

void clip_free(struct clip_ctx * ctx);

struct clip_text_hparams * clip_get_text_hparams(struct clip_ctx * ctx);
struct clip_vision_hparams * clip_get_vision_hparams(struct clip_ctx * ctx);

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

#ifdef __cplusplus
    std::vector<uint8_t> data;
#endif
};

// RGB float32 image
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

#ifdef __cplusplus
    std::vector<float> data;
#endif
};

struct clip_tokens clip_tokenize_c(const struct clip_ctx * ctx, const char * text);

struct clip_image_u8 * make_clip_image_u8();
struct clip_image_f32 * make_clip_image_f32();
bool clip_image_load_from_file_c(const char * fname, struct clip_image_u8 * img);
bool clip_image_preprocess(const struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32 * res);

bool clip_text_encode_c(const struct clip_ctx * ctx, int n_threads, const struct clip_tokens * tokens, float * vec);
bool clip_image_encode_c(const struct clip_ctx * ctx, int n_threads, const struct clip_image_f32 * img, float * vec);

// bool image_normalize(clip_image_u8 *img, clip_image_f32 *res);

bool clip_compare_text_and_image_c(struct clip_ctx * ctx, int n_threads, char * text, struct clip_image_u8 * image,
                                   float * score);
float clip_similarity_score(float * vec1, float * vec2, int vec_dim);
bool softmax_with_sorting(float * arr, int length, float * sorted_scores, int * indices);

#ifdef __cplusplus
}

std::vector<clip_vocab::id> clip_tokenize(const clip_ctx * ctx, const std::string & text);

bool clip_image_load_from_file(const std::string & fname, clip_image_u8 & img);

bool clip_text_encode(const clip_ctx * ctx, int n_threads, const std::vector<clip_vocab::id> & tokens, float * vec);
bool clip_image_encode(const struct clip_ctx * ctx, int n_threads, const struct clip_image_f32 & img, float * vec);

bool clip_compare_text_and_image(clip_ctx * ctx, int n_threads, std::string & text, clip_image_u8 & image, float * score);

// TODO clip_image_batch_encode_c
bool clip_image_batch_encode(const clip_ctx * ctx, int n_threads, const std::vector<clip_image_f32> & imgs, float * vec);

// TODO clip_image_batch_preprocess_c
void clip_image_batch_preprocess(const clip_ctx * ctx, const int n_threads, const std::vector<clip_image_u8> & img_inputs,
                                 std::vector<clip_image_f32> & img_resized);

#endif

#endif // CLIP_H
