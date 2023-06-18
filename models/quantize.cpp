#include "ggml/ggml.h"
#include "clip.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

// quantize a model
bool clip_model_quantize(const std::string &fname_inp, const std::string &fname_out, int itype)
{
    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype)
    {
    case 2:
        type = GGML_TYPE_Q4_0;
        break;
    case 3:
        type = GGML_TYPE_Q4_1;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
        return 1;
    };

    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1)
    {
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, type);
        return false;
    }

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto fin = std::ifstream(fname_inp, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout)
    {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *)&magic, sizeof(magic));
    }

    clip_text_hparams t_hparams;
    clip_vision_hparams v_hparams;
    int32_t ftype;

    // load text hparams
    {
        fin.read((char *)&t_hparams.n_vocab, sizeof(t_hparams.n_vocab));
        fin.read((char *)&t_hparams.num_positions, sizeof(t_hparams.num_positions));
        fin.read((char *)&t_hparams.hidden_size, sizeof(t_hparams.hidden_size));
        fin.read((char *)&t_hparams.n_intermediate, sizeof(t_hparams.n_intermediate));
        fin.read((char *)&t_hparams.projection_dim, sizeof(t_hparams.projection_dim));
        fin.read((char *)&t_hparams.n_head, sizeof(t_hparams.n_head));
        fin.read((char *)&t_hparams.n_layer, sizeof(t_hparams.n_layer));

        printf("%s: n_vocab = %d\n", __func__, t_hparams.n_vocab);
        printf("%s: num_positions   = %d\n", __func__, t_hparams.num_positions);
        printf("%s: t_hidden_size  = %d\n", __func__, t_hparams.hidden_size);
        printf("%s: t_n_intermediate  = %d\n", __func__, t_hparams.n_intermediate);
        printf("%s: t_n_head  = %d\n", __func__, t_hparams.n_head);
        printf("%s: t_n_layer = %d\n", __func__, t_hparams.n_layer);

        fout.write((char *)&t_hparams.n_vocab, sizeof(t_hparams.n_vocab));
        fout.write((char *)&t_hparams.num_positions, sizeof(t_hparams.num_positions));
        fout.write((char *)&t_hparams.hidden_size, sizeof(t_hparams.hidden_size));
        fout.write((char *)&t_hparams.n_intermediate, sizeof(t_hparams.n_intermediate));
        fout.write((char *)&t_hparams.projection_dim, sizeof(t_hparams.projection_dim));
        fout.write((char *)&t_hparams.n_head, sizeof(t_hparams.n_head));
        fout.write((char *)&t_hparams.n_layer, sizeof(t_hparams.n_layer));
    }

    // load vision hparams
    {
        fin.read((char *)&v_hparams.image_size, sizeof(v_hparams.image_size));
        fin.read((char *)&v_hparams.patch_size, sizeof(v_hparams.patch_size));
        fin.read((char *)&v_hparams.hidden_size, sizeof(v_hparams.hidden_size));
        fin.read((char *)&v_hparams.n_intermediate, sizeof(v_hparams.n_intermediate));
        fin.read((char *)&v_hparams.projection_dim, sizeof(v_hparams.projection_dim));
        fin.read((char *)&v_hparams.n_head, sizeof(v_hparams.n_head));
        fin.read((char *)&v_hparams.n_layer, sizeof(v_hparams.n_layer));
        fin.read((char *)&ftype, sizeof(ftype));

        printf("%s: image_size = %d\n", __func__, v_hparams.image_size);
        printf("%s: patch_size   = %d\n", __func__, v_hparams.patch_size);
        printf("%s: t_hidden_size  = %d\n", __func__, v_hparams.hidden_size);
        printf("%s: t_n_intermediate  = %d\n", __func__, v_hparams.n_intermediate);
        printf("%s: t_n_head  = %d\n", __func__, v_hparams.n_head);
        printf("%s: t_n_layer = %d\n", __func__, v_hparams.n_layer);

        fout.write((char *)&v_hparams.image_size, sizeof(v_hparams.image_size));
        fout.write((char *)&v_hparams.patch_size, sizeof(v_hparams.patch_size));
        fout.write((char *)&v_hparams.hidden_size, sizeof(v_hparams.hidden_size));
        fout.write((char *)&v_hparams.n_intermediate, sizeof(v_hparams.n_intermediate));
        fout.write((char *)&v_hparams.projection_dim, sizeof(v_hparams.projection_dim));
        fout.write((char *)&v_hparams.n_head, sizeof(v_hparams.n_head));
        fout.write((char *)&v_hparams.n_layer, sizeof(v_hparams.n_layer));
        fout.write((char *)&itype, sizeof(itype));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *)&n_vocab, sizeof(n_vocab));
        fout.write((char *)&n_vocab, sizeof(n_vocab));

        std::string word;
        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));
            fout.write((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);
            fout.write((char *)word.data(), len);
        }
    }

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t> data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float> data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

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

            int32_t nelements = 1;
            int32_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            {
                static const char *ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                ".*weight",
            };

            bool quantize = false;
            for (const auto &s : k_names)
            {
                if (std::regex_match(name, std::regex(s)))
                {
                    quantize = true;
                    break;
                }
            }

            // quantize only 2D tensors
            quantize &= (n_dims == 2);

            if (quantize)
            {
                if (ftype != 0 && ftype != 1)
                {
                    fprintf(stderr, "%s: unsupported ftype %d for integer quantization\n", __func__, ftype);
                    return false;
                }

                if (ftype == 1)
                {
                    data_f16.resize(nelements);
                    fin.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i)
                    {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                }
                else
                {
                    data_f32.resize(nelements);
                    fin.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = itype;
            }
            else
            {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements * bpe);
                fin.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype), sizeof(ftype));
            for (int i = 0; i < n_dims; ++i)
            {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            if (quantize)
            {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (type)
                {
                case GGML_TYPE_Q4_0:
                {
                    cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                case GGML_TYPE_Q4_1:
                {
                    cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                }
                break;
                default:
                {
                    fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                    return false;
                }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float) / 1024.0 / 1024.0, cur_size / 1024.0 / 1024.0);
                for (size_t i = 0; i < hist_cur.size(); ++i)
                {
                    hist_all[i] += hist_cur[i];
                }

                for (size_t i = 0; i < hist_cur.size(); ++i)
                {
                    printf("%5.3f ", hist_cur[i] / (float)nelements);
                }
                printf("\n");
            }
            else
            {
                printf("size = %8.3f MB\n", data_u8.size() / 1024.0 / 1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

        {
            int64_t sum_all = 0;
            for (size_t i = 0; i < hist_all.size(); ++i)
            {
                sum_all += hist_all[i];
            }

            printf("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); ++i)
            {
                printf("%5.3f ", hist_all[i] / (float)sum_all);
            }
            printf("\n");
        }
    }

    fin.close();
    fout.close();

    return true;
}

// usage:
//  ./bin/quantize models/ggml-model-f16.bin models/ggml-model-14_1.bin type
//
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s /path/to/ggml-modelf16.bin /path/to/ggml-model-quantized.bin type\n", argv[0]);
        fprintf(stderr, "  type = 2 - q4_0\n");
        fprintf(stderr, "  type = 3 - q4_1\n");
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = {0, NULL, false};
        struct ggml_context *ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!clip_model_quantize(fname_inp, fname_out, itype))
        {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}