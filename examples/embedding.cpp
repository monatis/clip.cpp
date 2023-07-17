#include "clip.h"
#include "common-clip.h"
#include <iostream>
#include <iomanip>

struct embeddings_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    std::string model = "models/ggml-model-f16.bin";
    std::string text;
    std::string image_path;
    int verbose = 1;
};

bool embeddings_params_parse(int argc, char ** argv, embeddings_params & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            return false;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--text" && i + 1 < argc) {
            if (!params.image_path.empty()) {
                std::cerr << "Error: Both --text and --image were specified." << std::endl;
                return false;
            }
            params.text = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            if (!params.text.empty()) {
                std::cerr << "Error: Both --text and --image were specified." << std::endl;
                return false;
            }
            params.image_path = argv[++i];
        } else if ((arg == "-v" || arg == "--verbose") && i + 1 < argc) {
            params.verbose = std::stoi(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (params.text.empty() && params.image_path.empty()) {
        std::cerr << "Error: Either --text or --image must be specified." << std::endl;
        return false;
    }

    return true;
}

void print_help(int argc, char ** argv, embeddings_params & params) {
    std::cout << "Options:\n"
              << "-h, --help: Show this message and exit\n"
              << "-m <path>, --model <path>: Path to model. Default: " << params.model << "\n"
              << "-t N, --threads N: Number of threads to use for inference. Default: " << params.n_threads << "\n"
              << "--text <text>: Text to encode. One text or one image should be specified\n"
              << "--image <path>: Path to an image file. One text or one image should be specified\n"
              << "-v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: " << params.verbose << std::endl;
}

int main(int argc, char ** argv) {
    embeddings_params params;
    if (!embeddings_params_parse(argc, argv, params)) {
        print_help(argc, argv, params);
        return 1;
    }

    // Load the model
    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        std::cerr << "Unable to load model from " << params.model << std::endl;
        return 1;
    }

    const int vec_dim = ctx->vision_model.hparams.projection_dim;

    if (!params.text.empty()) {
        // Encode text
        auto tokens = clip_tokenize(ctx, params.text);
        float* vec = new float[vec_dim];
        clip_text_encode(ctx, params.n_threads, tokens, vec);

        // Print the text embeddings
        for (int i = 0; i < vec_dim; ++i) {
            std::cout << std::fixed << std::setprecision(6) << vec[i] << " ";
        }
        std::cout << std::endl;

    } else {
        // Load and preprocess image
        clip_image_u8 img_u8;
        clip_image_f32 img_f32;
        if (!clip_image_load_from_file(params.image_path, img_u8)) {
            std::cerr << "Failed to load image from " << params.image_path << std::endl;
            return 1;
        }

        clip_image_preprocess(ctx, &img_u8, &img_f32);

        // Encode image
        float* vec = new float[vec_dim];
        if (!clip_image_encode(ctx, params.n_threads, img_f32, vec)) {
            return 1;
        }

        // Print the image embeddings
        for (int i = 0; i < vec_dim; ++i) {
            std::cout << std::fixed << std::setprecision(6) << vec[i] << " ";
        }
        std::cout << std::endl;
    }

    clip_free(ctx);

    return 0;
}
