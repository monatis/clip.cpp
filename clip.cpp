#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include "ggml/ggml.h"
#include "clip.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

// ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/clip.py#L164
// resize largest dimension to 1024
// normalize: x = (x - mean) / std
//     mean = [123.675, 116.28, 103.53]
//     std  = [58.395, 57.12, 57.375]
//     TODO: why are these hardcoded !?
// pad to 1024x1024
// TODO: for some reason, this is not numerically identical to pytorch's interpolation
bool clip_image_preprocess(const clip_image_u8 &img, clip_image_f32 &res)
{
    const int nx = img.nx;
    const int ny = img.ny;

    const int nx2 = 224;
    const int ny2 = 224;

    res.nx = nx2;
    res.ny = ny2;
    res.data.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / 224.0f;

    fprintf(stderr, "%s: scale = %f\n", __func__, scale);

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const float m3[3] = {0.45145466f, 0.4578275f, 0.40821073f};
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

                const float v00 = img.data[j00];
                const float v01 = img.data[j01];
                const float v10 = img.data[j10];
                const float v11 = img.data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res.data[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }

    return true;
}

struct clip_ctx *clip_model_load(const char *fname)
{
    printf("%s: loading model from '%s' - please wait...", __func__, fname);

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
    clip_vision_model &vision_model = new_clip->vision_model;

    // load hparams
    {
        auto &hparams = vision_model.hparams;

        fin.read((char *)&hparams.image_size, sizeof(hparams.image_size));
        fin.read((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        fin.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fin.read((char *)&hparams.projection_dim, sizeof(hparams.projection_dim));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        printf("%s: image_size = %d\n", __func__, hparams.image_size);
        printf("%s: patch_size   = %d\n", __func__, hparams.patch_size);
        printf("%s: hidden_size  = %d\n", __func__, hparams.hidden_size);
        printf("%s: n_intermediate  = %d\n", __func__, hparams.n_intermediate);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype     = %d\n", __func__, hparams.ftype);
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (vision_model.hparams.ftype)
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
                __func__, fname, vision_model.hparams.ftype);
        clip_free(new_clip);
        return nullptr;
    }
    }

    auto &ctx = vision_model.ctx;
    size_t model_mem_req = 0;

    {
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
        model_mem_req += 2 * hidden_size * projection_dim * ggml_type_sizef(wtype);             // projection

        model_mem_req += (5 + 16 * n_layer) * 256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, model_mem_req / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        vision_model.ctx = ggml_init(params);
        if (!vision_model.ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            return nullptr;
        }
    }

    // prepare memory for the weights
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

        printf("%s: ", __func__);

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

            if (vision_model.tensors.find(name.data()) == vision_model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                clip_free(new_clip);
                return nullptr;
            }

            auto tensor = vision_model.tensors[name.data()];
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
                assert(ne[0] % 64 == 0);
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

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    // Calculate space requirements for setting up context buffers later
    {
        // TODO: We set the initial buffer size to 16MB and hope it's enough. Maybe there is a better way to do this?
        new_clip->buf_compute.resize(20 * 1024 * 1024);
    }

    return new_clip;
}

void clip_free(clip_ctx *ctx)
{
    ggml_free(ctx->vision_model.ctx);
    delete ctx;
}

bool clip_image_encode(
    const clip_ctx *ctx,
    const clip_image_f32 &img,
    int n_threads)
{
    const auto &model = ctx->vision_model;
    const auto &hparams = model.hparams;

    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size / patch_size) * (image_size / patch_size));
    const int num_positions = num_patches + 1;
    const int hidden_size = hparams.hidden_size;
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

    struct ggml_tensor *cur = ggml_conv_2d_sk_p0(ctx0, model.patch_embeddings, inp);
    cur = ggml_reshape_2d(ctx0, cur, num_patches, hidden_size);
    struct ggml_tensor *embeddings = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, num_positions, hidden_size);
    embeddings = ggml_acc(ctx0, embeddings, model.class_embedding, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);
    embeddings = ggml_acc(ctx0, embeddings, cur, cur->nb[1], cur->nb[2], cur->nb[3], model.class_embedding->nb[0] * hidden_size);
    embeddings = ggml_add(ctx0, embeddings,
                          ggml_transpose(ctx0, model.position_embeddings));
    embeddings = ggml_transpose(ctx0, embeddings);
    embeddings = ggml_cont(ctx0, embeddings);

    // pre-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings);

        embeddings = ggml_add(ctx0,
                              ggml_mul(ctx0,
                                       ggml_repeat(ctx0, model.pre_ln_w, embeddings),
                                       embeddings),
                              ggml_repeat(ctx0, model.pre_ln_b, embeddings));
    }
    ggml_set_name(embeddings, "check");

    // run the computation
    ggml_build_forward_expand(&gf, embeddings);
    ggml_graph_compute(ctx0, &gf);

    // print
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

    // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);
    return true;
}