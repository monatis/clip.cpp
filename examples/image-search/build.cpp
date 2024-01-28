#include "clip.h"
#include "common-clip.h"
#include "usearch/index.hpp"

#include <fstream>

struct my_app_params {
    int32_t n_threads{4};
    std::string model{"../models/ggml-model-f16.bin"};
    int32_t verbose{1};
    std::vector<std::string> image_directories;
};

void my_print_help(int argc, char ** argv, my_app_params & params) {
    printf("Usage: %s [options] dir/with/pictures [more/dirs]\n", argv[0]);
    printf("\nOptions:\n");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model. Default: %s\n", params.model.c_str());
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n",
           params.verbose);
}

// returns success
bool my_app_params_parse(int argc, char ** argv, my_app_params & params) {
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
        } else if (arg == "-v" || arg == "--verbose") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.verbose = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            my_print_help(argc, argv, params);
            exit(0);
        } else if (arg.find('-') == 0) {
            if (i != 0) {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        } else {
            // assume image directory
            params.image_directories.push_back(argv[i]);
        }
    }

    return !(invalid_param || params.image_directories.empty());
}

int main(int argc, char ** argv) {
    my_app_params params;
    if (!my_app_params_parse(argc, argv, params)) {
        my_print_help(argc, argv, params);
        return 1;
    }

    auto clip_ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!clip_ctx) {
        printf("%s: Unable  to load model from %s\n", __func__, params.model.c_str());
        return 1;
    }

    std::vector<std::string> image_file_index;
    unum::usearch::index_gt<unum::usearch::cos_gt<float>> embd_index;

    const size_t vec_dim = clip_get_vision_hparams(clip_ctx)->projection_dim;
    const size_t batch_size = 4;

    size_t label = 0;

    std::vector<float> vec(vec_dim * batch_size);
    std::vector<clip_image_u8> img_inputs(batch_size);
    std::vector<clip_image_f32> imgs_resized(batch_size);

    // search for images in path and write embedding to database
    for (const auto & base_dir : params.image_directories) {
        printf("%s: starting base dir scan of '%s'\n", __func__, base_dir.c_str());
        auto results = get_dir_keyed_files(base_dir, 0);

        for (auto & entry : results) {
            printf("\n%s: processing %zu files in '%s'\n", __func__, entry.second.size(), entry.first.c_str());

            size_t n_batched = (entry.second.size() / batch_size) * batch_size;
            img_inputs.resize(batch_size);
            imgs_resized.resize(batch_size);

            if (embd_index.capacity() == embd_index.size() || embd_index.capacity() < entry.second.size()) {
                embd_index.reserve(embd_index.size() + entry.second.size());
            }

            for (size_t i = 0; i < n_batched; i += batch_size) {
                for (size_t ib = i; ib < i + batch_size; ib++) {
                    const std::string & img_path = entry.second[ib];
                    if (params.verbose >= 2) {
                        printf("%s: found image file '%s'\n", __func__, img_path.c_str());
                    }

                    if (!clip_image_load_from_file(img_path.c_str(), &img_inputs[ib % batch_size])) {
                        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
                        continue;
                    }

                    image_file_index.push_back(img_path);
                }

                if (params.verbose == 1) {
                    printf(".");
                    fflush(stdout);
                }

                auto img_inputs_batch = clip_image_u8_batch{};
                img_inputs_batch.data = img_inputs.data();
                img_inputs_batch.size = img_inputs.size();

                auto imgs_resized_batch = clip_image_f32_batch{};
                imgs_resized_batch.data = imgs_resized.data();
                imgs_resized_batch.size = imgs_resized.size();

                clip_image_batch_preprocess(clip_ctx, params.n_threads, &img_inputs_batch, &imgs_resized_batch);

                clip_image_batch_encode(clip_ctx, params.n_threads, &imgs_resized_batch, vec.data(), true);

                // add image vectors to the database
                for (size_t b = 0; b < batch_size; b++) {
                    embd_index.add(label++, {vec.data() + b * vec_dim, vec_dim});
                }
            }

            // process leftover if needed

            const size_t leftover = entry.second.size() - n_batched;
            if (leftover > 0) {

                img_inputs.resize(leftover);
                imgs_resized.resize(leftover);

                for (size_t il = n_batched; il < entry.second.size(); il++) {
                    const std::string & img_path = entry.second[il];
                    if (params.verbose >= 2) {
                        printf("%s: found image file '%s'\n", __func__, img_path.c_str());
                    }

                    if (!clip_image_load_from_file(img_path.c_str(), &img_inputs[il - n_batched])) {
                        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
                        continue;
                    }

                    image_file_index.push_back(img_path);
                }

                if (params.verbose == 1) {
                    printf(".");
                    fflush(stdout);
                }

                auto img_inputs_batch = clip_image_u8_batch_make(img_inputs);
                auto imgs_resized_batch = clip_image_f32_batch_make(imgs_resized);

                clip_image_batch_preprocess(clip_ctx, params.n_threads, &img_inputs_batch, &imgs_resized_batch);
                clip_image_batch_encode(clip_ctx, params.n_threads, &imgs_resized_batch, vec.data(), true);

                // add image vectors to the database
                for (size_t l = 0; l < leftover; l++) {
                    embd_index.add(label++, {vec.data() + l * vec_dim, vec_dim});
                }
            }
        }
    }

    clip_free(clip_ctx);

    // save to disk

    embd_index.save("images.usearch");

    std::ofstream image_file_index_file("images.paths", std::ios::binary | std::ios::trunc);
    // first line is model
    image_file_index_file << params.model << "\n";
    for (const auto & i_path : image_file_index) {
        image_file_index_file << i_path << "\n";
    }

    printf("%s: %zu images processed and indexed\n", __func__, image_file_index.size());

    return 0;
}
