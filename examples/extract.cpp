// extract vectors of texts or images
// TODO: encode image in batches
// TODO: encode texts from a text file and images from a directory

#include "clip.h"
#include "common-clip.h"

int main(int argc, char ** argv) {
    app_params params;
    if (!app_params_parse(argc, argv, params, 0, 0)) {
        print_help(argc, argv, params, 0, 0);
        return 1;
    }

    if (params.image_paths.empty() && params.texts.empty()) {
        printf("You should provide at least 1 --text or --image argument\n");

        print_help(argc, argv, params, 0, 0);
        return 1;
    }

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    int totalInputs = params.image_paths.size() + params.texts.size();
    int processedInputs = 0;
    int textCounter = 0; // Counter for generating unique filenames for text vectors
    for (const std::string & img_path : params.image_paths) {
        // load the image
        const char * img_path_cstr = img_path.c_str();
        clip_image_u8 img_input;
        if (!clip_image_load_from_file(img_path_cstr, &img_input)) {
            fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path_cstr);
            continue;
        }

        clip_image_f32 img_res;
        if (!clip_image_preprocess(ctx, &img_input, &img_res)) {
            printf("Unable to preprocess image\n");
            continue;
        }

        const int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;
        int shape[2] = {1, vec_dim};
        float vec[vec_dim];
        clip_image_encode(ctx, params.n_threads, &img_res, vec, false);

        // Generate a unique output filename for each image
        std::string output_filename = "./img_vec_" + img_path.substr(img_path.find_last_of('/') + 1) + ".npy";
        writeNpyFile(output_filename.c_str(), vec, shape, 2);

        // Update progress
        processedInputs++;
        float progressPercentage = (float)processedInputs / totalInputs * 100.0f;
        printf("\rProcessing: %.2f%%", progressPercentage);
        fflush(stdout);
    }

    for (const std::string & text : params.texts) {
        const char * text_cstr = text.c_str();
        clip_tokens tokens;
        if (!clip_tokenize(ctx, text_cstr, &tokens)) {
            printf("Unable to tokenize text\n");
            continue;
        }

        const int vec_dim = clip_get_text_hparams(ctx)->projection_dim;
        int shape[2] = {1, vec_dim};
        float vec[vec_dim];

        if (!clip_text_encode(ctx, params.n_threads, &tokens, vec, false)) {
            printf("Unable to encode text\n");
            continue;
        }

        // Update progress
        processedInputs++;
        float progressPercentage = (float)processedInputs / totalInputs * 100.0f;
        printf("\rProcessing: %.2f%%", progressPercentage);
        fflush(stdout);

        // Generate a unique output filename for each text
        std::string output_filename = "./text_vec_" + std::to_string(textCounter++) + ".npy";
        writeNpyFile(output_filename.c_str(), vec, shape, 2);
    }

    printf("\n"); // Print a newline to clear the progress bar line
    clip_free(ctx);

    return 0;
}
