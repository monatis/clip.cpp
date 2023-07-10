#ifndef COMMON_CLIP_H
#define COMMON_CLIP_H

#include <vector>
#include <map>
#include <cstring>
#include <thread>

// #ifdef __cplusplus
// extern "C" {
// #endif

std::map<std::string, std::vector<std::string>> get_dir_keyed_files(const std::string &path, uint32_t max_files_per_dir);

bool is_image_file_extension(std::string &ext);

struct app_params
{
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    std::string model = "models/ggml-model-f16.bin";
    std::vector<std::string> image_paths;
    std::vector<std::string> texts;
    int verbose = 1;
};

bool app_params_parse(int argc, char **argv, app_params &params);
void print_help(int argc, char **argv, app_params &params);

// utils for debugging
void write_floats_to_file(float *array, int size, char *filename);

// #ifdef __cplusplus
// }
// #endif

#endif // COMMON_CLIP_H