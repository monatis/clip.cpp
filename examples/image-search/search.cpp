#include "clip.h"
#include "usearch/index.hpp"

#include <fstream>

struct my_app_params {
    int32_t n_threads {4};
    std::string model = "../models/ggml-model-f16.bin";
    // TODO: index dir

    // TODO: search by image
    std::string search_text;
};

void my_print_help(int argc, char **argv, my_app_params &params) {
    printf("Usage: %s [options] <search string>\n", argv[0]);
    printf("\nOptions:");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model. Default: %s\n", params.model.c_str());
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
}

// returns success
bool my_app_params_parse(int argc, char **argv, my_app_params &params) {
    bool invalid_param = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            my_print_help(argc, argv, params);
            exit(0);
        } else if (arg.starts_with('-')) {
            if (i != 0) {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        } else {
            // assume search string from here on out
            params.search_text = arg;
            for (++i; i < argc; i++) {
                params.search_text += " ";
                params.search_text += argv[i];
            }
        }
    }

    return !(invalid_param || params.search_text.empty());
}

int main(int argc, char** argv) {
    my_app_params params;
    if (!my_app_params_parse(argc, argv, params)) {
        my_print_help(argc, argv, params);
        return 1;
    }

    auto clip_ctx = clip_model_load(params.model.c_str(), 1); // TODO: verbosity via cli arg
    if (!clip_ctx) {
        printf("%s: Unable to load model from %s\n", __func__, params.model.c_str());
        return 1;
    }

    std::vector<std::string> image_file_index;
    unum::usearch::index_gt<unum::usearch::cos_gt<float>> embd_index;

    embd_index.view("images.usearch");

    // load paths
    std::ifstream image_file_index_file("images.paths", std::ios::binary);
    std::string line;
    do {
        std::getline(image_file_index_file, line);
        if (line.empty()) {
            break;
        }
        image_file_index.push_back(line);
    } while(image_file_index_file.good());

    if (image_file_index.size() != embd_index.size()) {
        printf("%s: index files size missmatch\n", __func__);
    }

    const size_t vec_dim = clip_ctx->vision_model.hparams.projection_dim;

    auto tokens = clip_tokenize(clip_ctx, params.search_text);

    std::vector<float> txt_vec(vec_dim);

    clip_text_encode(clip_ctx, params.n_threads, tokens, txt_vec.data());

    auto results = embd_index.search({txt_vec.data(), txt_vec.size()}, 5);

    printf("search results:\n");
    printf("similarity path\n");
    for (std::size_t i = 0; i != results.size(); ++i) {
        printf("  %f %s\n", results[i].distance, image_file_index.at(results[i].element.label).c_str());
    }

    clip_free(clip_ctx);

    return 0;
}

