#ifndef CLIP_H
#define CLIP_H

#include <vector>
#include <cstring>
#include <map>
#include <thread>
#include "ggml/ggml.h"

// TODO: make the API in C
// #ifdef __cplusplus
// extern "C" {
// #endif

struct app_params
{
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    std::string model = "models/ggml-model-f16.bin";
    std::vector<std::string> image_paths;
    std::vector<std::string> texts;
    int verbose = 1;
};

bool app_params_parse(int argc, char **argv, app_params &params);
void print_help(int argc, char **argv, app_params &params);

// default hparams for text_model (ViT-B/32)
struct clip_text_hparams
{
    int32_t n_vocab = 49408;
    int32_t num_positions = 77;
    int32_t hidden_size = 512;
    int32_t n_intermediate = 2048;
    int32_t projection_dim = 512;
    int32_t n_head = 8;
    int32_t n_layer = 12;
};

// default hparams for vision_model (ViT-B/32)
struct clip_vision_hparams
{
    int32_t image_size = 224;
    int32_t patch_size = 32;
    int32_t hidden_size = 768;
    int32_t n_intermediate = 3072;
    int32_t projection_dim = 512;
    int32_t n_head = 12;
    int32_t n_layer = 12;
};

typedef int32_t clip_vocab_id;

//
// Vocab utils
//

std::string trim(const std::string &s);

std::string replace(
    const std::string &s,
    const std::string &from,
    const std::string &to);

struct clip_vocab
{
    using id = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    void add_special_token(const std::string &token);
};

std::string convert_to_utf8(const std::wstring &input);

std::wstring convert_to_wstring(const std::string &input);

// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//

struct clip_layer
{
    // attention
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // layernorm 1
    struct ggml_tensor *ln_1_w;
    struct ggml_tensor *ln_1_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;

    // layernorm 2
    struct ggml_tensor *ln_2_w;
    struct ggml_tensor *ln_2_b;
};

struct clip_text_model
{
    clip_text_hparams hparams;

    // embeddings
    struct ggml_tensor *token_embeddings;
    struct ggml_tensor *position_embeddings;

    std::vector<clip_layer> layers;

    struct ggml_tensor *post_ln_w;
    struct ggml_tensor *post_ln_b;

    struct ggml_tensor *projection;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct clip_vision_model
{
    clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor *class_embedding;
    struct ggml_tensor *patch_embeddings;
    struct ggml_tensor *position_embeddings;

    struct ggml_tensor *pre_ln_w;
    struct ggml_tensor *pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor *post_ln_w;
    struct ggml_tensor *post_ln_b;

    struct ggml_tensor *projection;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct clip_ctx *clip_model_load(const char *fname, const int verbosity);

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct clip_buffer
{
    uint8_t *data = NULL;
    size_t size = 0;

    void resize(size_t size)
    {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~clip_buffer()
    {
        delete[] data;
    }
};

struct clip_ctx
{
    clip_text_model text_model;
    clip_vision_model vision_model;
    clip_vocab vocab;
    int32_t ftype = 1;
    ggml_context *ctx;
    clip_buffer buf_compute;
};

void clip_free(clip_ctx *ctx);

// RGB uint8 image
struct clip_image_u8
{
    int nx;
    int ny;

    std::vector<uint8_t> data;
};

// RGB float32 image
// Memory layout: RGBRGBRGB...
struct clip_image_f32
{
    int nx;
    int ny;

    std::vector<float> data;
};

std::vector<clip_vocab::id> clip_tokenize(const clip_ctx *ctx, const std::string &text);

bool clip_image_load_from_file(const std::string &fname, clip_image_u8 &img);
bool clip_image_preprocess(const clip_image_u8 *img, clip_image_f32 *res);
bool clip_image_preprocess_bicubic(const clip_image_u8 *img, clip_image_f32 *res);

bool clip_text_encode(
    const clip_ctx *ctx,
    int n_threads,
    const std::vector<clip_vocab::id> &tokens,
    float *vec);

bool clip_image_encode(
    const clip_ctx *ctx,
    int n_threads,
    const clip_image_f32 &img,
    float *vec);

bool image_normalize(clip_image_u8 *img, clip_image_f32 *res);

bool clip_compare_text_and_image(clip_ctx *ctx, int n_threads, std::string &text, clip_image_u8 &image, float *score);
float clip_similarity_score(float *vec1, float *vec2, int vec_dim);
bool softmax_with_sorting(float *arr, int length, float *sorted_scores, int *indices);

// utils for debugging
void write_floats_to_file(float *array, int size, char *filename);

// #ifdef __cplusplus
// }
// #endif

#endif // CLIP_H