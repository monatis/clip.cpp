import argparse

from clip_cpp import Clip

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="clip")
    ap.add_argument("-m", "--model", help="path to GGML file or repo_id", required=True)
    ap.add_argument(
        "-fn",
        "--filename",
        help="path to GGML file in the Hugging face repo",
        required=False,
    )
    ap.add_argument(
        "-v",
        "--verbosity",
        type=int,
        help="Level of verbosity. 0 = minimum, 2 = maximum",
        default=0,
    )
    ap.add_argument(
        "-t",
        "--text",
        help="text to encode. Multiple values allowed. In this case, apply zero-shot labeling",
        nargs="+",
        type=str,
        required=True,
    )
    ap.add_argument("-i", "--image", help="path to an image file", required=True)
    args = ap.parse_args()

    clip = Clip(args.model, args.verbosity)
    if len(args.text) == 1:
        score = clip.compare_text_and_image(args.text[0], args.image)

        print(f"Similarity score: {score}")
    else:
        sorted_scores, sorted_indices = clip.zero_shot_label_image(
            args.image, args.text
        )
        for ind, score in zip(sorted_indices, sorted_scores):
            label = args.text[ind]
            print(f"{label}: {score:.4f}")
