#include "clip.h"
#include "common-clip.h"
#include "ggml/ggml.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        printf("usage: %s <model_path> <images_dir> <num_images_per_dir> [output_file]\n\n", argv[0]);
        printf("model_path: path to CLIP model in GGML format\n");
        printf("images_dir: path to a directory of images where images are organized into subdirectories named classes\n");
        printf("num_images_per_dir: maximum number of images to read from each one of subdirectories. if 0, read all files\n");
        printf("output_file: optional. if specified, dump the output to this file instead of stdout\n");
        return 1;
    }

    std::string model_path = argv[1];
    std::string dir_path = argv[2];
    uint32_t max_files_per_dir = std::stoi(argv[3]); // Example: Limit to 100 files per directory
    FILE * fout = stdout;
    if (argc == 5) {
        fout = fopen(argv[4], "w");
        if (fout == NULL) {
            printf("%s: cannot open file %s\n", __func__, argv[4]);
            return 1;
        }

        printf("%s: dumping benchmarking results to %s...\n", __func__, argv[4]);
    }

    auto result = get_dir_keyed_files(dir_path, max_files_per_dir);

    size_t n_labels = result.size();
    if (n_labels < 2) {
        printf("%s There must be at least 2 directories of images, but %zu found\n", __func__, n_labels);
        return 1;
    }

    fprintf(fout, "%s: %zu directories found in %s\n\n", __func__, n_labels, dir_path.c_str());

    auto ctx = clip_model_load(model_path.c_str(), 2);
    if (!ctx) {
        printf("%s: unable to load model from %s\n", __func__, model_path.c_str());
        return 1;
    }

    const size_t batch_size = 4;
    const size_t n_threads = 4;

    const int vec_dim = clip_get_text_hparams(ctx)->projection_dim;

    float txt_vecs[n_labels * vec_dim];

    ggml_time_init();

    // walk through directory names and encode them as texts

    int label_idx = 0;

    const int64_t t_start_encode_texts = ggml_time_us();

    for (const auto & entry : result) {
        clip_tokens tokens;
        clip_tokenize(ctx, entry.first.c_str(), &tokens);
        if (!clip_text_encode(ctx, n_threads, &tokens, txt_vecs + label_idx * vec_dim, true)) {
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
    float img_vecs[vec_dim * batch_size];

    float similarities[n_labels];
    float sorted_scores[n_labels];
    int indices[n_labels];
    std::vector<clip_image_u8> img_inputs(batch_size);
    std::vector<clip_image_f32> imgs_resized(batch_size);

    // print table headers
    fprintf(fout, "| class name           | acc@1  | acc@5  |\n");
    fprintf(fout, "| -------------------- | ------ | ------ |\n");

    int64_t t_start_encode_images = ggml_time_us();

    for (auto & entry : result) {
        int n_items = 0;
        int n_acc1 = 0;
        int n_acc5 = 0;

        size_t n_batched = (entry.second.size() / batch_size) * batch_size;

        for (size_t i = 0; i < n_batched; i += batch_size) {
            for (size_t ib = i; ib < i + batch_size; ib++) {
                std::string file_path = entry.second[ib];

                if (!clip_image_load_from_file(file_path.c_str(), &img_inputs[ib % batch_size])) {
                    printf("%s: cannot load file from %s\n", __func__, file_path.c_str());
                    return 1;
                }
            }

            auto img_inputs_batch = clip_image_u8_batch_make(img_inputs);
            auto imgs_resized_batch = clip_image_f32_batch_make(imgs_resized);

            clip_image_batch_preprocess(ctx, n_threads, &img_inputs_batch, &imgs_resized_batch);

            clip_image_batch_encode(ctx, n_threads, &imgs_resized_batch, img_vecs, true);

            for (size_t b = 0; b < batch_size; b++) {
                for (size_t j = 0; j < n_labels; j++) {
                    similarities[j] = clip_similarity_score(img_vecs + b * vec_dim, txt_vecs + j * vec_dim, vec_dim);
                }
                softmax_with_sorting(similarities, n_labels, sorted_scores, indices);

                for (int k = 0; k < 5; k++) {
                    if (k == 0 && indices[k] == label_idx) {
                        n_acc1 += 1;
                        n_acc5 += 1;
                        break;
                    } else if (indices[k] == label_idx) {
                        n_acc5 += 1;
                        break;
                    }
                }

                n_items += 1;
                n_total_items += 1;
            }
        }

        float acc1_score = (float)n_acc1 / n_items;
        float acc5_score = (float)n_acc5 / n_items;
        total_acc1_score += acc1_score;
        total_acc5_score += acc5_score;
        fprintf(fout, "| %-*s ", 20, entry.first.c_str());
        fprintf(fout, "| %2.4f | %2.4f |\n", acc1_score, acc5_score);
        label_idx += 1;
    }

    int64_t t_end_encode_images = ggml_time_us();

    fprintf(fout, "| total                | %2.4f | %2.4f |\n\n", total_acc1_score / (float)n_labels,
            total_acc5_score / (float)n_labels);

    // print timings
    float total_text_duration = (t_end_encode_texts - t_start_encode_texts) / 1000.0f;
    float total_image_duration = (t_end_encode_images - t_start_encode_images) / 1000.0f;
    fprintf(fout, "# Timings\n");
    fprintf(fout, "- %zu texts encoded in %8.2f ms (%8.2f ms per text)\n", n_labels, total_text_duration,
            total_text_duration / (float)n_labels);
    fprintf(fout, "- %d images encoded in %8.2f ms (%8.2f ms per image)\n", n_total_items, total_image_duration,
            total_image_duration / (float)n_total_items);

    if (fout != stdout) {
        fclose(fout);
    }

    clip_free(ctx);

    return 0;
}
