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
        : n_threads(std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()))),
          model("models/ggml-model-f16.gguf"), verbose(1) {
        // Initialize other fields if needed
    }
};

bool app_params_parse(int argc, char ** argv, app_params & params, const int min_text_arg, const int min_image_arg);
void print_help(int argc, char ** argv, app_params & params, const int min_text_arg, const int min_image_arg);

int writeNpyFile(const char * filename, const float * data, const int * shape, int ndims);

// utils for debugging
void write_floats_to_file(float * array, int size, char * filename);

// constructor-like functions
struct clip_image_u8_batch clip_image_u8_batch_make(std::vector<clip_image_u8> & images);
struct clip_image_f32_batch clip_image_f32_batch_make(std::vector<clip_image_f32> & images);

// #ifdef __cplusplus
// }
// #endif

#endif // COMMON_CLIP_H