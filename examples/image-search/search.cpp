#include "clip.h"
#include "usearch/index.hpp"

#include <fstream>

struct my_app_params {
    int32_t n_threads {4};
    std::string model;
    int32_t verbose {1};
    // TODO: index dir

    // TODO: search by image
    std::string search_text;

    int32_t n_results {5};
};

void my_print_help(int argc, char **argv, my_app_params &params) {
    printf("Usage: %s [options] <search string>\n", argv[0]);
    printf("\nOptions:");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: overwrite path to model. Read from images.paths by default.\n");
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n", params.verbose);

    printf("  -n N, --results N: Number of results to display. Default: %d\n", params.n_results);
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
        } else if (arg == "-n" || arg == "--results") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_results = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.verbose = std::stoi(argv[i]);
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

    // load model path
    std::ifstream image_file_index_file("images.paths", std::ios::binary);
    std::string line;
    std::getline(image_file_index_file, line);
    if (params.model.empty()) {
        params.model = line;
    } else {
        printf("%s: using alternative model from %s. Make sure you use the same model you used for indexing, or the embeddings wont work.\n", __func__, params.model.c_str());
    }

    // load model
    auto clip_ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!clip_ctx) {
        printf("%s: Unable to load model from %s\n", __func__, params.model.c_str());
        return 1;
    }

    // load paths and embeddings database
    std::vector<std::string> image_file_index;
    unum::usearch::index_gt<unum::usearch::cos_gt<float>> embd_index;

    embd_index.view("images.usearch");

    // load image paths
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

    auto results = embd_index.search({txt_vec.data(), txt_vec.size()}, params.n_results);

    printf("search results:\n");
    printf("distance path\n");
    for (std::size_t i = 0; i != results.size(); ++i) {
        printf("  %f %s\n", results[i].distance, image_file_index.at(results[i].member.label).c_str());
    }

    clip_free(clip_ctx);

    return 0;
}

