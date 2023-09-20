#ifndef COMMON_CLIP_H
#define COMMON_CLIP_H

#include "clip.h"
#include <cstring>
#include <map>
#include <thread>
#include <vector>

// #ifdef __cplusplus
// extern "C" {
// #endif

std::map<std::string, std::vector<std::string>> get_dir_keyed_files(const std::string & path, uint32_t max_files_per_dir);

bool is_image_file_extension(const std::string & path);

#include <algorithm>
#include <string>
#include <vector>

struct app_params {
    int32_t n_threads;
    std::string model;
    std::vector<std::string> image_paths;
    std::vector<std::string> texts;
    int verbose;

    app_params()
        : n_threads(std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()))), model("models/ggml-model-f16.bin"),
          verbose(1) {
        // Initialize other fields if needed
    }
};

/*
struct app_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    std::string model = "models/ggml-model-f16.bin";
    std::vector<std::string> image_paths;
    std::vector<std::string> texts;
    int verbose = 1;
};
*/

bool app_params_parse(int argc, char ** argv, app_params & params);
void print_help(int argc, char ** argv, app_params & params);

// utils for debugging
void write_floats_to_file(float * array, int size, char * filename);

// constructor-like functions
struct clip_image_u8_batch make_clip_image_u8_batch(std::vector<clip_image_u8> & images);
struct clip_image_f32_batch make_clip_image_f32_batch(std::vector<clip_image_f32> & images);

// #ifdef __cplusplus
// }
// #endif

#endif // COMMON_CLIP_H