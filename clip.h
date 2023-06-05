#ifndef CLIP_H
#define CLIP_H

#include <vector>
#include <string>
#include <map>
#include <thread>

// TODO: make the API in C
// #ifdef __cplusplus
// extern "C" {
// #endif

struct app_params
{
    int32_t seed = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t port = 8080; // server mode port to bind

    std::string model = "models/ggml-clip-vision-model-q5_1.bin"; // model path
    std::string image_path;
};

bool app_params_parse(int argc, char **argv, app_params &params);

// default hparams (ViT-B/32)
struct clip_vision_hparams
{
    int32_t image_size = 224;
    int32_t patch_size = 32;
    int32_t hidden_size = 768;
    int32_t n_intermediate = 3072;
    int32_t projection_dim = 512;
    int32_t n_head = 12;
    int32_t n_layer = 12;
    int32_t ftype = 1;
};

struct clip_vision_layer
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

struct clip_vision_model
{
    clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor *class_embedding;
    struct ggml_tensor *patch_embeddings;
    struct ggml_tensor *position_embeddings;

    struct ggml_tensor *pre_ln_w;
    struct ggml_tensor *pre_ln_b;

    std::vector<clip_vision_layer> layers;

    struct ggml_tensor *post_ln_w;
    struct ggml_tensor *post_ln_b;

    struct ggml_tensor *projection;

    // key + value memory
    // struct ggml_tensor * memory_k;
    // struct ggml_tensor * memory_v;

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct clip_ctx *clip_model_load(const char *fname);

// evaluate the clip vision model
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - embeddings:  the embeddings of the image in the context
//

bool clip_vision_eval(
    const clip_vision_model &model,
    const int n_threads, float *embeddings);

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
    clip_vision_model vision_model;

    size_t mem_per_token;
    int64_t mem_per_input;
    int32_t max_batch_n;
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
bool clip_image_load_from_file(const std::string &fname, clip_image_u8 &img);
bool clip_image_preprocess(const clip_image_u8 &img, clip_image_f32 &res);
bool clip_image_encode(
    const clip_ctx *ctx,
    const clip_image_f32 &img,
    int n_threads);
// #ifdef __cplusplus
// }
// #endif

#endif // CLIP_H