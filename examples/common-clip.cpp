#include "common-clip.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

// common utility functions mainly intended for examples and debugging

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

bool app_params_parse(int argc, char **argv, app_params &params)
{
    for (int i = 0; i < argc; i++)
    {
        std::string arg = std::string(argv[i]);
        if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--text")
        {
            params.texts.push_back(argv[++i]);
        }
        else if (arg == "--image")
        {
            params.image_paths.push_back(argv[++i]);
        }
        else if (arg == "-v" || arg == "--verbose")
        {
            params.verbose = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            print_help(argc, argv, params);
            exit(0);
        }
        else
        {
            if (i != 0)
            {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        }
    }
    return params.image_paths.size() >= 1 && params.texts.size() >= 1;
}

void print_help(int argc, char **argv, app_params &params)
{
    printf("Usage: %s [options]\n", argv[0]);
    printf("\nOptions:");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model. Default: %s\n", params.model.c_str());
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  --text <text>: Text to encode. At least one text should be specified\n");
    printf("  --image <path>: Path to an image file. At least one image path should be specified\n");
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n", params.verbose);
}

void write_floats_to_file(float *array, int size, char *filename)
{
    // Open the file for writing.
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file: %s\n", filename);
        return;
    }

    // Write the float values to the file.
    for (int i = 0; i < size; i++)
    {
        fprintf(file, "%f\n", array[i]);
    }

    // Close the file.
    fclose(file);
}
