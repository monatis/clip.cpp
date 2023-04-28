#include "ggml/ggml.h"

#include "common-ggml.h"
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
bool clip_vision_model_quantize(const std::string &fname_inp, const std::string &fname_out, ggml_ftype ftype)
{

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp)
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
        finp.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *)&magic, sizeof(magic));
    }

    clip_vision_hparams hparams;

    // load hparams
    {
        finp.read((char *)&hparams.image_size, sizeof(hparams.image_size));
        finp.read((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        finp.read((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        finp.read((char *)&hparams.intermediate_size, sizeof(hparams.intermediate_size));
        finp.read((char *)&hparams.projection_dim, sizeof(hparams.projection_dim));
        finp.read((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        finp.read((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        finp.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        printf("%s: image_size = %d\n", __func__, hparams.image_size);
        printf("%s: patch_size   = %d\n", __func__, hparams.patch_size);
        printf("%s: hidden_size  = %d\n", __func__, hparams.hidden_size);
        printf("%s: intermediate_size  = %d\n", __func__, hparams.intermediate_size);
        printf("%s: projection_dim = %d\n", __func__, hparams.projection_dim);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);

        fout.write((char *)&hparams.image_size, sizeof(hparams.image_size));
        fout.write((char *)&hparams.patch_size, sizeof(hparams.patch_size));
        fout.write((char *)&hparams.hidden_size, sizeof(hparams.hidden_size));
        fout.write((char *)&hparams.intermediate_size, sizeof(hparams.intermediate_size));
        fout.write((char *)&hparams.projection_dim, sizeof(hparams.projection_dim));
        fout.write((char *)&hparams.num_attention_heads, sizeof(hparams.num_attention_heads));
        fout.write((char *)&hparams.num_hidden_layers, sizeof(hparams.num_hidden_layers));
        fout.write((char *)&ftype, sizeof(hparams.ftype));
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> to_quant = {
        ".*weight",
    };

    if (!ggml_common_quantize_0(finp, fout, ftype, to_quant, {}))
    {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

// usage:
//  ./bin/quantize models/ggml-clip-vision-model-f16.bin models/ggml-clip-vision-model-quant.bin type
//
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        ggml_print_ftypes(stderr);
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

    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!clip_vision_model_quantize(fname_inp, fname_out, ggml_ftype(ftype)))
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
