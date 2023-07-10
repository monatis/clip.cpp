#include "clip.h"
#include "usearch/index.hpp"

#include <fstream>
#include <filesystem>

struct my_app_params
{
    int32_t n_threads{4};
    std::string model{"../models/ggml-model-f16.bin"};
    int32_t verbose{1};
    std::vector<std::string> image_directories;
};

void my_print_help(int argc, char **argv, my_app_params &params)
{
    printf("Usage: %s [options] dir/with/pictures [more/dirs]\n", argv[0]);
    printf("\nOptions:");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model. Default: %s\n", params.model.c_str());
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n", params.verbose);
}

// returns success
bool my_app_params_parse(int argc, char **argv, my_app_params &params)
{
    bool invalid_param = false;
    for (int i = 1; i < argc; i++)
    {

        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        }
        else if (arg == "-t" || arg == "--threads")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.verbose = std::stoi(argv[i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            my_print_help(argc, argv, params);
            exit(0);
        }
        else if (arg.find('-') == 0)
        {
            if (i != 0)
            {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        }
        else
        {
            // assume image directory
            params.image_directories.push_back(argv[i]);
        }
    }

    return !(invalid_param || params.image_directories.empty());
}

bool is_image_file_extension(std::string_view ext)
{
    if (ext == ".jpg")
        return true;
    if (ext == ".JPG")
        return true;

    if (ext == ".jpeg")
        return true;
    if (ext == ".JPEG")
        return true;

    if (ext == ".gif")
        return true;
    if (ext == ".GIF")
        return true;

    if (ext == ".png")
        return true;
    if (ext == ".PNG")
        return true;

    // TODO(green-sky): determine if we should add more formats from stbi. tga/hdr/pnm seem kinda niche.

    return false;
}

int main(int argc, char **argv)
{
    my_app_params params;
    if (!my_app_params_parse(argc, argv, params))
    {
        my_print_help(argc, argv, params);
        return 1;
    }

    auto clip_ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!clip_ctx)
    {
        printf("%s: Unable  to load model from %s\n", __func__, params.model.c_str());
        return 1;
    }

    std::vector<std::string> image_file_index;
    unum::usearch::index_gt<unum::usearch::cos_gt<float>> embd_index;

    const size_t vec_dim = clip_ctx->vision_model.hparams.projection_dim;

    size_t label = 0;

    std::vector<float> vec(vec_dim);

    // search for images in path and write embedding to database
    for (const auto &base_dir : params.image_directories)
    {
        fprintf(stdout, "%s: starting base dir scan of '%s'\n", __func__, base_dir.c_str());

        for (auto const &dir_entry : std::filesystem::recursive_directory_iterator(base_dir))
        {
            if (!dir_entry.is_regular_file())
            {
                continue;
            }

            // check for image file
            const auto &ext = dir_entry.path().extension();
            if (ext.empty())
            {
                continue;
            }
            if (!is_image_file_extension(ext.c_str()))
            {
                continue;
            }

            std::string img_path{dir_entry.path()};
            if (params.verbose >= 1)
            {
                fprintf(stdout, "%s: found image file '%s'\n", __func__, img_path.c_str());
            }

            clip_image_u8 img0;
            if (!clip_image_load_from_file(img_path, img0))
            {
                fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
                continue;
            }

            clip_image_f32 img_res;
            clip_image_preprocess(clip_ctx, &img0, &img_res);

            if (!clip_image_encode(clip_ctx, params.n_threads, img_res, vec.data()))
            {
                fprintf(stderr, "%s: failed to encode image from '%s'\n", __func__, img_path.c_str());
                continue;
            }

            if (embd_index.capacity() == embd_index.size())
            {
                embd_index.reserve(embd_index.size() + 32);
            }

            // add the image to the database
            embd_index.add(label++, {vec.data(), vec.size()});
            image_file_index.push_back(std::filesystem::canonical(dir_entry.path()));
        }
    }

    clip_free(clip_ctx);

    // save to disk

    embd_index.save("images.usearch");

    std::ofstream image_file_index_file("images.paths", std::ios::binary | std::ios::trunc);
    // first line is model
    image_file_index_file << params.model << "\n";
    for (const auto &i_path : image_file_index)
    {
        image_file_index_file << i_path << "\n";
    }

    return 0;
}
