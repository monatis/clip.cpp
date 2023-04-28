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

struct bert_vocab
{
    using id = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<token, id> subword_token_to_id;

    token id_to_token(const id &arg) const
    {
        auto it = _id_to_token.find(arg);
        if (it != _id_to_token.end())
        {
            return it->second;
        }
        it = _id_to_subword_token.find(arg);
        if (it != _id_to_subword_token.end())
        {
            return it->second;
        }
        return "[UNK TOKEN from bert_vocab]";
    }
    std::map<id, token> _id_to_token;
    std::map<id, token> _id_to_subword_token;
};

std::vector<bert_vocab::id> bert_tokenize(const bert_vocab &vocab, const std::string &text);

struct app_params
{
    int32_t seed = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t port = 8080; // server mode port to bind

    std::string model = "models/ggml-clip-vision-model-q5_1.bin"; // model path
    std::string prompt;
};

bool app_params_parse(int argc, char **argv, app_params &params);

// default hparams (ViT-B/32)
struct clip_vision_hparams
{
    int32_t image_size = 224;
    int32_t patch_size = 32;
    int32_t hidden_size = 768;
    int32_t intermediate_size = 3072;
    int32_t projection_dim = 512;
    int32_t num_attention_heads = 12;
    int32_t num_hidden_layers = 12;
    int32_t ftype = 1;
};

struct clip_vision_layer
{
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct clip_vision_model
{
    clip_vision_hparams hparams;

    // embeddings
    // struct ggml_tensor * position_ids;
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;

    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<clip_vision_layer> layers;

    // key + value memory
    // struct ggml_tensor * memory_k;
    // struct ggml_tensor * memory_v;

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

bool clip_vision_model_load(const std::string &fname, clip_vision_model &model);

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - tokens:  the embeddings of the tokens in the context
//

std::vector<float> clip_vision_eval(
    const clip_vision_model &model,
    const int n_threads);

int clip_n_embd(const clip_vision_model &model);

// #ifdef __cplusplus
// }
// #endif

#endif // BERT_H