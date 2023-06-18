#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <regex>
#include <fstream>
#include "ggml/ggml.h"
#include "clip.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::vector<clip_vocab::id> clip_tokenize(const clip_ctx *ctx, const std::string &text)
{
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        // Generate the subpattern from the special_tokens vector if it's not empty
        if (!ctx->vocab.special_tokens.empty())
        {
            std::string special_tokens_subpattern;
            for (const auto &token : ctx->vocab.special_tokens)
            {
                if (!special_tokens_subpattern.empty())
                {
                    special_tokens_subpattern += "|";
                }
                special_tokens_subpattern += token;
            }

            // Modify the regex pattern with the generated special tokens subpattern
            pat = special_tokens_subpattern + "|" + pat;
        }

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re))
        {
            for (auto x : m)
            {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    std::vector<clip_vocab::id> tokens;
    tokens.push_back(49406); // startoftext

    for (const auto &word : words)
    {
        // feel lucky? let's try if it's a full word
        std::string full_word = "";
        if (word.starts_with(" "))
        {
            full_word += word.substr(1);
        }
        else
        {
            full_word += word;
        }
        full_word += "</w>";
        auto wit = ctx->vocab.token_to_id.find(full_word);
        if (wit != ctx->vocab.token_to_id.end())
        {
            tokens.push_back(wit->second);
            continue;
        }

        for (int i = 0; i < word.size();)
        {
            for (int j = word.size() - 1; j >= i; j--)
            {
                auto cand = word.substr(i, j - i + 1);
                auto it = ctx->vocab.token_to_id.find(cand);
                if (it != ctx->vocab.token_to_id.end())
                { // word.substr(i, j-i+1) in vocab
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                }
                else if (j == i)
                { // word.substr(i, 1) has no matching
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                    i++;
                }
            }
        }
    }

    tokens.push_back(49407); // endoftext

    return tokens;
}

bool clip_image_load_from_file(const std::string &fname, clip_image_u8 &img)
{
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data)
    {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

// normalize: x = (x - mean) / std
// TODO: implement bicubic interpolation instead of linear.
bool clip_image_preprocess(const clip_image_u8 *img, clip_image_f32 *res)
{
    const int nx = img->nx;
    const int ny = img->ny;

    const int nx2 = 224;
    const int ny2 = 224;

    res->nx = nx2;
    res->ny = ny2;
    res->data.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / 224.0f;

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const float m3[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float s3[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++)
    {
        for (int x = 0; x < nx3; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = img->data[j00];
                const float v01 = img->data[j01];
                const float v10 = img->data[j10];
                const float v11 = img->data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res->data[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }

    return true;
}

struct clip_ctx *clip_model_load(const char *fname, const int verbosity = 1)
{
    if (verbosity >= 1)
    {
        printf("%s: loading model from '%s' - please wait...", __func__, fname);
    }

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        return nullptr;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname);
            return nullptr;
        }
    }

    clip_ctx *new_clip = new clip_ctx;
    clip_text_model &text_model = new_clip->text_model;
    clip_vision_model &vision_model = new_clip->vision_model;
    clip_vocab &vocab = new_clip->vocab;

    // load hparams for text
    {
        auto &hparams = text_model.hparams;

        fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.num_positions, sizeof(hparams.num_positions));
        fin.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fin.read((char *)&hparams.projection_dim, sizeof(hparams.projection_dim));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));

        if (verbosity >= 2)
        {
            printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
            printf("%s: num_positions   = %d\n", __func__, hparams.num_positions);
            printf("%s: t_hidden_size  = %d\n", __func__, hparams.hidden_size);
            printf("%s: t_n_intermediate  = %d\n", __func__, hparams.n_intermediate);
            printf("%s: t_n_head  = %d\n", __func__, hparams.n_head);
            printf("%s: t_n_layer = %d\n", __func__, hparams.n_layer);
        }
    }

    // load hparams for vision
    {
        auto &hparams = vision_model.hparams;

        fin.read((char *)&hparams.image_size, sizeof(hparams.image_size));
        fin.read((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        fin.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fin.read((char *)&hparams.projection_dim, sizeof(hparams.projection_dim));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&new_clip->ftype, sizeof(new_clip->ftype));

        if (verbosity >= 2)
        {
            printf("%s: image_size = %d\n", __func__, hparams.image_size);
            printf("%s: patch_size   = %d\n", __func__, hparams.patch_size);
            printf("%s: v_hidden_size  = %d\n", __func__, hparams.hidden_size);
            printf("%s: v_n_intermediate  = %d\n", __func__, hparams.n_intermediate);
            printf("%s: v_n_head  = %d\n", __func__, hparams.n_head);
            printf("%s: v_n_layer = %d\n", __func__, hparams.n_layer);
            printf("%s: ftype     = %d\n", __func__, new_clip->ftype);
        }
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *)&n_vocab, sizeof(n_vocab));

        if (n_vocab != new_clip->text_model.hparams.n_vocab)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname, n_vocab, new_clip->text_model.hparams.n_vocab);
            return nullptr;
        }

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            buf.resize(len);
            fin.read((char *)buf.data(), len);
            word.assign(buf.data(), len);

            new_clip->vocab.token_to_id[word] = i;
            new_clip->vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (new_clip->ftype)
    {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname, new_clip->ftype);
        clip_free(new_clip);
        return nullptr;
    }
    }

    auto &ctx = new_clip->ctx;
    size_t model_mem_req = 0;

    {
        // calculate memory requirement for text_model
        const auto &hparams = text_model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int num_positions = hparams.num_positions;
        const int hidden_size = hparams.hidden_size;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int projection_dim = hparams.projection_dim;

        // Calculate size requirements

        model_mem_req += hidden_size * n_vocab * ggml_type_sizef(wtype);       // token_embeddings
        model_mem_req += hidden_size * num_positions * ggml_type_sizef(wtype); // position_embeddings

        model_mem_req += 4 * n_layer * (hidden_size * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_* and ln_2_*

        model_mem_req += 4 * n_layer * (hidden_size * hidden_size * ggml_type_sizef(wtype));    // kqvo weights
        model_mem_req += 4 * n_layer * (hidden_size * ggml_type_sizef(GGML_TYPE_F32));          // kqvo bias
        model_mem_req += 2 * n_layer * (hidden_size * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32));           // ff_i_b
        model_mem_req += n_layer * (hidden_size * ggml_type_sizef(GGML_TYPE_F32));              // ff_o_b

        model_mem_req += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);          // post_ln_*
        model_mem_req += 2 * hidden_size * projection_dim * ggml_type_sizef(wtype); // projection

        model_mem_req += (5 + 16 * n_layer) * 256; // object overhead
    }

    {
        // calculate memory requirement for vision_model
        const auto &hparams = vision_model.hparams;

        const int image_size = hparams.image_size;
        const int patch_size = hparams.patch_size;
        const int num_patches = ((image_size / patch_size) * (image_size / patch_size)) + 1;
        const int hidden_size = hparams.hidden_size;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int projection_dim = hparams.projection_dim;

        // Calculate size requirements

        model_mem_req += hidden_size * ggml_type_sizef(GGML_TYPE_F32);                       // class_embedding
        model_mem_req += hidden_size * 3 * patch_size * patch_size * ggml_type_sizef(wtype); // patch_embeddings
        model_mem_req += hidden_size * num_patches * ggml_type_sizef(wtype);                 // position_embeddings

        model_mem_req += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32); // pre_ln_*

        model_mem_req += 4 * n_layer * (hidden_size * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        model_mem_req += 4 * n_layer * (hidden_size * hidden_size * ggml_type_sizef(wtype)); // kqvo weights
        model_mem_req += 4 * n_layer * (hidden_size * ggml_type_sizef(GGML_TYPE_F32));       // kqvo bias

        model_mem_req += 2 * n_layer * (hidden_size * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32));           // ff_i_b
        model_mem_req += n_layer * (hidden_size * ggml_type_sizef(GGML_TYPE_F32));              // ff_o_b

        model_mem_req += 2 * hidden_size * ggml_type_sizef(GGML_TYPE_F32);          // post_ln_*
        model_mem_req += 2 * hidden_size * projection_dim * ggml_type_sizef(wtype); // projection

        model_mem_req += (5 + 16 * n_layer) * 256; // object overhead
    }

    if (verbosity >= 2)
    {
        printf("%s: ggml ctx size = %6.2f MB\n", __func__, model_mem_req / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        new_clip->ctx = ggml_init(params);
        if (!new_clip->ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            return nullptr;
        }
    }

    // prepare memory for the text_model weights
    {
        const auto &hparams = text_model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int num_positions = hparams.num_positions;
        const int hidden_size = hparams.hidden_size;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int projection_dim = hparams.projection_dim;

        text_model.layers.resize(n_layer);

        text_model.token_embeddings = ggml_new_tensor_2d(ctx, wtype, hidden_size, n_vocab);
        text_model.position_embeddings = ggml_new_tensor_2d(ctx, wtype, hidden_size, num_positions);

        // map by name
        text_model.tensors["text_model.embeddings.token_embedding.weight"] = text_model.token_embeddings;
        text_model.tensors["text_model.embeddings.position_embedding.weight"] = text_model.position_embeddings;

        for (int i = 0; i < n_layer; ++i)
        {
            auto &layer = text_model.layers[i];

            layer.ln_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.ln_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, n_intermediate);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, n_intermediate, hidden_size);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // map by name

            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.k_proj.weight"] = layer.k_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.k_proj.bias"] = layer.k_b;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.v_proj.weight"] = layer.v_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.v_proj.bias"] = layer.v_b;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.q_proj.weight"] = layer.q_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.q_proj.bias"] = layer.q_b;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.out_proj.weight"] = layer.o_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".self_attn.out_proj.bias"] = layer.o_b;

            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".layer_norm1.weight"] = layer.ln_1_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".layer_norm1.bias"] = layer.ln_1_b;

            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".mlp.fc1.weight"] = layer.ff_i_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".mlp.fc1.bias"] = layer.ff_i_b;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".mlp.fc2.weight"] = layer.ff_o_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".mlp.fc2.bias"] = layer.ff_o_b;

            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".layer_norm2.weight"] = layer.ln_2_w;
            text_model.tensors["text_model.encoder.layers." + std::to_string(i) + ".layer_norm2.bias"] = layer.ln_2_b;
        }

        text_model.post_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        text_model.post_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        text_model.projection = ggml_new_tensor_2d(ctx, wtype, hidden_size, projection_dim);

        // map by name
        text_model.tensors["text_model.final_layer_norm.weight"] = text_model.post_ln_w;
        text_model.tensors["text_model.final_layer_norm.bias"] = text_model.post_ln_b;
        text_model.tensors["text_projection.weight"] = text_model.projection;
    }

    // prepare memory for the vision_model weights
    {
        const auto &hparams = vision_model.hparams;

        const int image_size = hparams.image_size;
        const int patch_size = hparams.patch_size;
        const int num_patches = ((image_size / patch_size) * (image_size / patch_size)) + 1;
        const int hidden_size = hparams.hidden_size;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int projection_dim = hparams.projection_dim;

        vision_model.layers.resize(n_layer);

        vision_model.class_embedding = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        vision_model.patch_embeddings = ggml_new_tensor_4d(ctx, wtype, patch_size, patch_size, 3, hidden_size);
        vision_model.position_embeddings = ggml_new_tensor_2d(ctx, wtype, hidden_size, num_patches);

        vision_model.pre_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        vision_model.pre_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // map by name
        vision_model.tensors["vision_model.embeddings.class_embedding"] = vision_model.class_embedding;
        vision_model.tensors["vision_model.embeddings.patch_embedding.weight"] = vision_model.patch_embeddings;
        vision_model.tensors["vision_model.embeddings.position_embedding.weight"] = vision_model.position_embeddings;

        vision_model.tensors["vision_model.pre_layrnorm.weight"] = vision_model.pre_ln_w;
        vision_model.tensors["vision_model.pre_layrnorm.bias"] = vision_model.pre_ln_b;

        for (int i = 0; i < n_layer; ++i)
        {
            auto &layer = vision_model.layers[i];

            layer.ln_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.ln_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, hidden_size, n_intermediate);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, n_intermediate, hidden_size);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            // map by name

            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.k_proj.weight"] = layer.k_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.k_proj.bias"] = layer.k_b;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.v_proj.weight"] = layer.v_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.v_proj.bias"] = layer.v_b;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.q_proj.weight"] = layer.q_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.q_proj.bias"] = layer.q_b;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.out_proj.weight"] = layer.o_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".self_attn.out_proj.bias"] = layer.o_b;

            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".layer_norm1.weight"] = layer.ln_1_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".layer_norm1.bias"] = layer.ln_1_b;

            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc1.weight"] = layer.ff_i_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc1.bias"] = layer.ff_i_b;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc2.weight"] = layer.ff_o_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc2.bias"] = layer.ff_o_b;

            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".layer_norm2.weight"] = layer.ln_2_w;
            vision_model.tensors["vision_model.encoder.layers." + std::to_string(i) + ".layer_norm2.bias"] = layer.ln_2_b;
        }

        vision_model.post_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        vision_model.post_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        vision_model.projection = ggml_new_tensor_2d(ctx, wtype, hidden_size, projection_dim);

        // map by name
        vision_model.tensors["vision_model.post_layernorm.weight"] = vision_model.post_ln_w;
        vision_model.tensors["vision_model.post_layernorm.bias"] = vision_model.post_ln_b;
        vision_model.tensors["visual_projection.weight"] = vision_model.projection;
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof())
            {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            struct ggml_tensor *tensor;
            if (text_model.tensors.find(name.data()) != text_model.tensors.end())
            {
                tensor = text_model.tensors[name.data()];
            }
            else if (vision_model.tensors.find(name.data()) != vision_model.tensors.end())
            {
                tensor = vision_model.tensors[name.data()];
            }
            else
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                clip_free(new_clip);
                return nullptr;
            }

            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                clip_free(new_clip);
                return nullptr;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                clip_free(new_clip);
                return nullptr;
            }

            if (0)
            {
                static const char *ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype)
            {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                // assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                clip_free(new_clip);
                return nullptr;
            }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                clip_free(new_clip);
                return nullptr;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

#ifdef CLIP_DEBUG
            printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor) / 1024.0 / 1024.0);
#endif

            total_size += ggml_nbytes(tensor);
            if (verbosity >= 1)
            {
                if (++n_tensors % 8 == 0)
                {
                    printf(".");
                    fflush(stdout);
                }
            }
        }

        if (verbosity >= 1)
        {
            printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
        }
    }

    fin.close();

    // Calculate space requirements for setting up context buffers later
    {
        // TODO: We set the initial buffer size to 16MB and hope it's enough. Maybe there is a better way to do this?
        new_clip->buf_compute.resize(96 * 1024 * 1024);
    }
    if (verbosity >= 1)
    {
        printf("%s: model loadded\n", __func__);
    }

    return new_clip;
}

void clip_free(clip_ctx *ctx)
{
    ggml_free(ctx->ctx);
    delete ctx;
}

bool clip_text_encode(
    const clip_ctx *ctx,
    int n_threads,
    const std::vector<clip_vocab::id> &tokens,
    float *vec)
{
    const auto &model = ctx->text_model;
    const auto &hparams = model.hparams;
    const int N = tokens.size();

    const int n_vocab = hparams.n_vocab;
    const int num_positions = hparams.num_positions;
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;
    const int n_layer = hparams.n_layer;
    const int n_intermediate = hparams.n_intermediate;
    const int projection_dim = hparams.projection_dim;

    auto &buf_compute = ctx->buf_compute;

    struct ggml_init_params params = {
        .mem_size = buf_compute.size,
        .mem_buffer = buf_compute.data,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor *input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(input_ids->data, tokens.data(), N * ggml_element_size(input_ids));

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; i++)
    {
        ggml_set_i32_1d(positions, i, i);
    }

    struct ggml_tensor *embeddings = ggml_get_rows(ctx0, model.token_embeddings, input_ids);

    embeddings = ggml_add(ctx0,
                          ggml_get_rows(ctx0, model.position_embeddings, positions),
                          embeddings);

    // loop over layers
    for (int il = 0; il < n_layer; il++)
    {
        struct ggml_tensor *cur = embeddings; // embeddings = residual, cur = hidden_states

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_1_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // self-attention
        {
            struct ggml_tensor *Q = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur),
                                             ggml_mul_mat(ctx0, model.layers[il].q_w, cur));

            Q = ggml_scale_inplace(ctx0, Q, ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head)));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, N, 1);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, N, n_head);

            struct ggml_tensor *K =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur),
                         ggml_mul_mat(ctx0, model.layers[il].k_w, cur));

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, N, 1);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, N, n_head);

            struct ggml_tensor *V =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur),
                         ggml_mul_mat(ctx0, model.layers[il].v_w, cur));
            V = ggml_reshape_4d(ctx0, V, d_head, n_head, N, 1);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, N, d_head, n_head);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, 0); // causal masking
            KQ = ggml_soft_max_inplace(ctx0, KQ);

            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, N, n_head, 1);
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

            cur = ggml_cpy(ctx0,
                           KQV,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, N));
        }

        // attention output
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].o_b, cur),
                       ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_2_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                       cur);

        cur = ggml_gelu_quick_inplace(ctx0, cur);

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                       cur);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);
        // ggml_set_name(cur, "check");

        embeddings = cur;
    }

    // final -layer_norm
    {
        embeddings = ggml_norm(ctx0, embeddings);

        embeddings = ggml_add(ctx0,
                              ggml_mul(ctx0,
                                       ggml_repeat(ctx0, model.post_ln_w, embeddings),
                                       embeddings),
                              ggml_repeat(ctx0, model.post_ln_b, embeddings));
    }

    // get the output of eot token, e.g., last index
    struct ggml_tensor *eot = ggml_new_i32(ctx0, N - 1);
    embeddings = ggml_get_rows(ctx0, embeddings, eot);

    // text projection
    embeddings = ggml_mul_mat(ctx0, model.projection, embeddings);

    // normalize output embeddings
    ggml_tensor *length = ggml_sqrt(ctx0,
                                    ggml_sum(ctx0, ggml_sqr(ctx0, embeddings)));
    embeddings = ggml_scale_inplace(ctx0, embeddings, ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length));

    ggml_set_name(embeddings, "check");

    // run the computation
    ggml_build_forward_expand(&gf, embeddings);
    ggml_graph_compute(ctx0, &gf);

// print
#ifdef CLIP_DEBUG
    {
        auto print_t_f32 = [&](struct ggml_tensor *t)
        {
            float *data = (float *)t->data;
            printf("dtype: f32, dims: %jd %jd %jd %jd, nb: %jd %jd %jd %jd\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int)t->ne[0], 20); i++)
            {
                printf("%f ", data[i]);
            }

            // printf("\n\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++)
            {
                sum += data[i];
            }
            printf("sum:  %f\n", sum);
        };

        auto print_t_f16 = [&](struct ggml_tensor *t)
        {
            ggml_fp16_t *data = (ggml_fp16_t *)t->data;
            printf("dtype: f16, dims: %jd %jd %jd %jd\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int)t->ne[0], 10); i++)
            {
                printf("%f ", ggml_fp16_to_fp32(data[i]));
            }
            printf("\n\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++)
            {
                sum += ggml_fp16_to_fp32(data[i]);
            }
            printf("sum:  %f\n", sum);
        };

        auto *t = ggml_get_tensor(ctx0, "check");
        if (t->type == GGML_TYPE_F32)
        {
            print_t_f32(t);
        }
        else
        {
            print_t_f16(t);
        }
    }

    printf("used_mem = %zu\n", ggml_used_mem(ctx0));
#endif
    memcpy(vec, ggml_get_data_f32(embeddings), sizeof(float) * projection_dim);

    ggml_free(ctx0);

    return true;
}

bool clip_image_encode(
    const clip_ctx *ctx,
    int n_threads,
    const clip_image_f32 &img,
    float *vec)
{
    const auto &model = ctx->vision_model;
    const auto &hparams = model.hparams;

    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size / patch_size) * (image_size / patch_size));
    const int num_positions = num_patches + 1;
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;
    const int n_layer = hparams.n_layer;
    const int n_intermediate = hparams.n_intermediate;
    const int projection_dim = hparams.projection_dim;

    auto &buf_compute = ctx->buf_compute;

    struct ggml_init_params params = {
        .mem_size = buf_compute.size,
        .mem_buffer = buf_compute.data,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor *inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, 1);
    // auto fin = std::ifstream("/home/yusuf/clip-in-ggml/tests/inputs.bin", std::ios::binary);
    // fin.read(reinterpret_cast<char *>(inp->data), ggml_nbytes(inp));

    // if (0)
    {
        float *data = (float *)ggml_get_data(inp);

        const int nx = img.nx;
        const int ny = img.ny;
        const int n = nx * ny;

        GGML_ASSERT(nx == image_size && ny == image_size);

        for (int k = 0; k < 3; k++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[k * n + y * nx + x] = img.data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    inp = ggml_conv_2d_sk_p0(ctx0, model.patch_embeddings, inp);

    inp = ggml_reshape_2d(ctx0, inp, num_patches, hidden_size);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // concat class_embeddings and patch_embeddings
    struct ggml_tensor *embeddings = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, num_positions);
    ggml_set_zero(embeddings);
    embeddings = ggml_acc(ctx0, embeddings, model.class_embedding, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);
    embeddings = ggml_acc(ctx0, embeddings, inp, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], ggml_element_size(model.class_embedding) * hidden_size);

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    for (int i = 0; i < num_positions; i++)
    {
        ggml_set_i32_1d(positions, i, i);
    }

    embeddings = ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.position_embeddings, positions));

    // pre-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings);

        embeddings = ggml_add(ctx0,
                              ggml_mul(ctx0,
                                       ggml_repeat(ctx0, model.pre_ln_w, embeddings),
                                       embeddings),
                              ggml_repeat(ctx0, model.pre_ln_b, embeddings));
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++)
    {
        struct ggml_tensor *cur = embeddings; // embeddings = residual, cur = hidden_states

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_1_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // self-attention
        {
            struct ggml_tensor *Q = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur),
                                             ggml_mul_mat(ctx0, model.layers[il].q_w, cur));

            Q = ggml_scale_inplace(ctx0, Q, ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head)));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, 1);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head);

            struct ggml_tensor *K =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur),
                         ggml_mul_mat(ctx0, model.layers[il].k_w, cur));

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, 1);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head);

            struct ggml_tensor *V =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur),
                         ggml_mul_mat(ctx0, model.layers[il].v_w, cur));
            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, 1);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            KQ = ggml_soft_max_inplace(ctx0, KQ);

            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, 1);
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

            cur = ggml_cpy(ctx0,
                           KQV,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, num_positions));
        }

        // attention output
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].o_b, cur),
                       ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_2_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                       cur);

        cur = ggml_gelu_quick_inplace(ctx0, cur);
        // ggml_set_name(cur, "check");

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                       cur);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);
        // ggml_set_name(cur, "check");

        embeddings = cur;
        // break;
    }

    // get the output of cls token, e.g., 0th index
    struct ggml_tensor *cls = ggml_new_i32(ctx0, 0);
    embeddings = ggml_get_rows(ctx0, embeddings, cls);

    // post-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings);

        embeddings = ggml_add(ctx0,
                              ggml_mul(ctx0,
                                       ggml_repeat(ctx0, model.post_ln_w, embeddings),
                                       embeddings),
                              ggml_repeat(ctx0, model.post_ln_b, embeddings));
    }

    // final visual projection
    embeddings = ggml_mul_mat(ctx0, model.projection, embeddings);

    // normalize output embeddings
    ggml_tensor *length = ggml_sqrt(ctx0,
                                    ggml_sum(ctx0, ggml_sqr(ctx0, embeddings)));
    embeddings = ggml_scale_inplace(ctx0, embeddings, ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length));

    ggml_set_name(embeddings, "check");

    // run the computation
    ggml_build_forward_expand(&gf, embeddings);
    ggml_graph_compute(ctx0, &gf);

// print
#ifdef CLIP_DEBUG
    {
        auto print_t_f32 = [&](struct ggml_tensor *t)
        {
            float *data = (float *)t->data;
            printf("dtype: f32, dims: %jd %jd %jd %jd, nb: %jd %jd %jd %jd\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int)t->ne[0], 20); i++)
            {
                printf("%f ", data[i]);
            }

            // printf("\n\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++)
            {
                sum += data[i];
            }
            printf("sum:  %f\n", sum);
        };

        auto print_t_f16 = [&](struct ggml_tensor *t)
        {
            ggml_fp16_t *data = (ggml_fp16_t *)t->data;
            printf("dtype: f16, dims: %jd %jd %jd %jd\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int)t->ne[0], 10); i++)
            {
                printf("%f ", ggml_fp16_to_fp32(data[i]));
            }
            printf("\n\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++)
            {
                sum += ggml_fp16_to_fp32(data[i]);
            }
            printf("sum:  %f\n", sum);
        };

        auto *t = ggml_get_tensor(ctx0, "check");
        if (t->type == GGML_TYPE_F32)
        {
            print_t_f32(t);
        }
        else
        {
            print_t_f16(t);
        }
    }

    printf("used_mem = %zu\n", ggml_used_mem(ctx0));
#endif

    memcpy(vec, ggml_get_data_f32(embeddings), sizeof(float) * projection_dim);

    ggml_free(ctx0);

    return true;
}

float clip_similarity_score(float *vec1, float *vec2, int vec_dim)
{
    float dot_product = 0.0;
    for (int i = 0; i < vec_dim; i++)
    {
        dot_product += vec1[i] * vec2[i];
    }

    // Clamp the dot product to the range [0, 1].
    float clamped_dot_product = fmin(fmax(dot_product, 0.0), 1.0);

    return clamped_dot_product;
}

bool clip_compare_text_and_image(clip_ctx *ctx, int n_threads, std::string &text, clip_image_u8 &image, float *score)
{
    // prepare image and text vectors
    const int projection_dim = ctx->vision_model.hparams.projection_dim;
    float img_vec[projection_dim];
    float txt_vec[projection_dim];

    // preprocess and encode image
    clip_image_f32 img_res;

    if (!clip_image_preprocess(&image, &img_res))
    {
        return false;
    }

    if (!clip_image_encode(ctx, n_threads, img_res, img_vec))
    {
        return false;
    }

    // tokenize and encode text
    auto tokens = clip_tokenize(ctx, text);

    if (!clip_text_encode(ctx, n_threads, tokens, txt_vec))
    {
        return false;
    }

    // compute similarity
    *score = clip_similarity_score(img_vec, txt_vec, projection_dim);

    return true;
}

typedef struct
{
    float score;
    int index;
} ScoreIndexPair;

int compare_scores(const void *a, const void *b)
{
    const ScoreIndexPair *pair1 = (const ScoreIndexPair *)a;
    const ScoreIndexPair *pair2 = (const ScoreIndexPair *)b;

    if (pair1->score < pair2->score)
    {
        return 1;
    }
    else if (pair1->score > pair2->score)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

bool softmax_with_sorting(float *arr, int length, float *sorted_scores, int *indices)
{
    ScoreIndexPair *score_index_pairs = (ScoreIndexPair *)malloc(length * sizeof(ScoreIndexPair));
    if (!score_index_pairs)
    {
        return false;
    }

    // Calculate softmax probabilities
    float max_val = arr[0];
    for (int i = 1; i < length; i++)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < length; i++)
    {
        arr[i] = exp(arr[i] - max_val);
        sum += arr[i];
    }

    for (int i = 0; i < length; i++)
    {
        arr[i] /= sum;
        score_index_pairs[i].score = arr[i];
        score_index_pairs[i].index = i;
    }

    // Sort scores in descending order
    qsort(score_index_pairs, length, sizeof(ScoreIndexPair), compare_scores);

    // Copy sorted scores and indices to the respective arrays
    for (int i = 0; i < length; i++)
    {
        sorted_scores[i] = score_index_pairs[i].score;
        indices[i] = score_index_pairs[i].index;
    }

    free(score_index_pairs);
    return true;
}

bool image_normalize(clip_image_u8 *img, clip_image_f32 *res)
{
    if (img->nx != 224 || img->ny != 224)
    {
        printf("%s: long input shape: %d x %s\n", __func__, img->nx, img->ny);
        return false;
    }

    const float m3[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float s3[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < img->ny; y++)
    {
        for (int x = 0; x < img->nx; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                const int i = 3 * (y * img->nx + x) + c;
                float v = (float)img->data[i];
                res->data[i] = ((v / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    return true;
}

// utility functions mainly intended for examples and debugging

bool app_params_parse(int argc, char **argv, app_params &params)
{
    for (int i = 0; i < argc; i++)
    {
        std::string arg = std::string(argv[i]);
        if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--text")
        {
            params.texts.push_back(argv[++i]);
        }
        else if (arg == "--image")
        {
            params.image_paths.push_back(argv[++i]);
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            params.verbose = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_help(argc, argv, params);
            exit(0);
        }
        else
        {
            if (i != 0)
            {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        }
    }
    return params.image_paths.size() >= 1 && params.texts.size() >= 1;
}

void print_help(int argc, char **argv, app_params &params)
{
    printf("Usage: %s [options]\n", argv[0]);
    printf("\nOptions:");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model. Default: %s\n", params.model.c_str());
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  --text <text>: Text to encode. At least one text should be specified\n");
    printf("  --image <path>: Path to an image file. At least one image path should be specified\n");
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n", params.verbose);
}

void write_floats_to_file(float *array, int size, char *filename)
{
    // Open the file for writing.
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file: %s\n", filename);
        return;
    }

    // Write the float values to the file.
    for (int i = 0; i < size; i++)
    {
        fprintf(file, "%f\n", array[i]);
    }

    // Close the file.
    fclose(file);
}
