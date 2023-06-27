#include "ggml/ggml.h"
#include "clip.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

// TODO: consider moving this to clip.cpp so that downstream examples and apps can also use it
std::map<std::string, std::vector<std::string>> get_dir_keyed_files(const std::string &path, uint32_t max_files_per_dir = 0)
{
    std::map<std::string, std::vector<std::string>> result;

#ifdef _WIN32
    std::string wildcard = path + "\\*";
    WIN32_FIND_DATAA fileData;
    HANDLE hFind = FindFirstFileA(wildcard.c_str(), &fileData);

    if (hFind == INVALID_HANDLE_VALUE)
    {
        std::cerr << "Failed to open directory: " << path << std::endl;
        return result;
    }

    uint32_t fileCount = 0;

    do
    {
        std::string name = fileData.cFileName;
        std::string fullPath = path + "\\" + name;

        if (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            // Skip . and ..
            if (name == "." || name == "..")
                continue;

            std::map<std::string, std::vector<std::string>> subResult = get_dir_keyed_files(fullPath, max_files_per_dir);
            result.insert(subResult.begin(), subResult.end());
        }
        else
        {
            size_t pos = path.find_last_of("\\/");
            std::string parentDir = (pos != std::string::npos) ? path.substr(pos + 1) : path;
            result[parentDir].push_back(fullPath);

            ++fileCount;
            if (max_files_per_dir > 0 && fileCount >= max_files_per_dir)
                break;
        }
    } while (FindNextFileA(hFind, &fileData));

    FindClose(hFind);
#else
    DIR *dir;
    struct dirent *entry;
    struct stat fileStat;

    if ((dir = opendir(path.c_str())) == NULL)
    {
        std::cerr << "Failed to open directory: " << path << std::endl;
        return result;
    }

    uint32_t fileCount = 0;

    while ((entry = readdir(dir)) != NULL)
    {
        std::string name = entry->d_name;
        std::string fullPath = path + "/" + name;

        if (stat(fullPath.c_str(), &fileStat) < 0)
        {
            std::cerr << "Failed to get file stat: " << fullPath << std::endl;
            continue;
        }

        if (S_ISDIR(fileStat.st_mode))
        {
            // Skip . and ..
            if (name == "." || name == "..")
                continue;

            std::map<std::string, std::vector<std::string>> subResult = get_dir_keyed_files(fullPath, max_files_per_dir);
            result.insert(subResult.begin(), subResult.end());
        }
        else
        {
            size_t pos = path.find_last_of("/");
            std::string parentDir = (pos != std::string::npos) ? path.substr(pos + 1) : path;
            result[parentDir].push_back(fullPath);

            ++fileCount;
            if (max_files_per_dir > 0 && fileCount >= max_files_per_dir)
                break;
        }
    }

    closedir(dir);
#endif

    return result;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("usage: %s <model_path> <images_dir> <num_images_per_dir>\n\n", argv[0]);
        printf("image_path: path to CLIP model in GGML format\n");
        printf("images_dir: path to a directory of images where images are organized into subdirectories named classes\n");
        printf("num_images_per_dir: maximum number of images to read from each one of subdirectories. if 0, read all files\n");
        return 1;
    }

    std::string model_path = argv[1];
    std::string dir_path = argv[2];
    uint32_t max_files_per_dir = std::stoi(argv[3]); // Example: Limit to 100 files per directory

    auto result = get_dir_keyed_files(dir_path, max_files_per_dir);

    size_t n_labels = result.size();
    if (n_labels < 2)
    {
        printf("%s There must be at least 2 directories of images, but %d found\n", __func__, n_labels);
        return 1;
    }

    printf("%s: %d directories found\n", __func__, n_labels);

    auto ctx = clip_model_load(model_path.c_str(), 2);
    if (!ctx)
    {
        printf("%s: unable to load model from %s\n", __func__, model_path.c_str());
        return 1;
    }

    const int vec_dim = ctx->text_model.hparams.projection_dim;

    // allocate memory for text vectors
    float *txt_vecs = (float *)malloc(n_labels * vec_dim * sizeof(float));
    if (!txt_vecs)
    {
        printf("%s: Could not allocate memory for %d vectors of %d dimensions\n", __func__, n_labels, vec_dim);
    }

    ggml_time_init();

    // walk through directory names and encode them as texts

    int label_idx = 0;

    const int64_t t_start_encode_texts = ggml_time_us();

    for (const auto &entry : result)
    {
        auto tokens = clip_tokenize(ctx, entry.first);
        if (!clip_text_encode(ctx, 4, tokens, txt_vecs + label_idx * vec_dim))
        {
            printf("%s: Could not encode the label at index %d: %s\n", __func__, label_idx, entry.first.c_str());
            return 1;
        }

        label_idx += 1;
    }

    const int64_t t_end_encode_texts = ggml_time_us();

    label_idx = 0;                 // reset label index
    int n_total_items = 0;         // total number of images processed
    float total_acc1_score = 0.0f; // total accuracy at 1 for the intire dataset
    float total_acc5_score = 0.0f; // total accuracy at 5 in intitre dataset
    float img_vec[vec_dim];
    float similarities[n_labels];
    float sorted_scores[n_labels];
    int indices[n_labels];
    clip_image_u8 img;
    clip_image_f32 img_res;

    int64_t t_start_encode_images = ggml_time_us();

    for (auto &entry : result)
    {
        int n_items = 0;
        int n_acc1 = 0;
        int n_acc5 = 0;

        int64_t t_start_encode_images = ggml_time_us();

        for (auto &file_path : entry.second)
        {
            if (!clip_image_load_from_file(file_path, img))
            {
                printf("%s: cannot load file from %s\n", __func__, file_path.c_str());
                return 1;
            }

            if (!clip_image_preprocess(ctx, &img, &img_res))
            {
                printf("%s: cannot preprocess image loaded from %s\n", __func__, file_path.c_str());
                return 1;
            }

            clip_image_encode(ctx, 4, img_res, img_vec);
            for (size_t i = 0; i < n_labels; i++)
            {
                similarities[i] = clip_similarity_score(img_vec, txt_vecs + i * vec_dim, vec_dim);
            }

            softmax_with_sorting(similarities, n_labels, sorted_scores, indices);
            for (int j = 0; j < 5; j++)
            {
                if (j == 0 && indices[j] == label_idx)
                {
                    n_acc1 += 1;
                    n_acc5 += 1;
                    break;
                }
                else if (indices[j] == label_idx)
                {
                    n_acc5 += 1;
                    break;
                }
            }

            n_items += 1;
            n_total_items += 1;
        }

        float acc1_score = (float)n_acc1 / n_items;
        float acc5_score = (float)n_acc5 / n_items;
        total_acc1_score += acc1_score;
        total_acc5_score += acc5_score;
        printf("%s: acc@1 = %2.4f - acc@5 = %2.4f\n", entry.first.c_str(), acc1_score, acc5_score);

        label_idx += 1;
    }

    int64_t t_end_encode_images = ggml_time_us();

    printf("total: acc@1 = %2.4f - acc@5 = %2.4f\n\n", total_acc1_score / (float)n_labels, total_acc5_score / (float)n_labels);

    printf("Timings:\n");
    printf("%d texts encoded in %8.2f ms\n", n_labels, (t_end_encode_texts - t_start_encode_texts) / 1000.0f);
    printf("%d images encoded in %8.2f ms\n", n_total_items, (t_end_encode_images - t_start_encode_images) / 1000.0f);

    clip_free(ctx);
    return 0;
}
