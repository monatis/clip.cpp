import argparse

from clip_cpp import Clip

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="clip")
    ap.add_argument("-m", "--model", help="path to GGML file or repo_id", required=True)
    ap.add_argument("-fn", "--filename", help="path to GGML file in the Hugging face repo", required=False)
    ap.add_argument(
        "-v",
        "--verbosity",
        type=int,
        help="Level of verbosity. 0 = minimum, 2 = maximum",
        default=0,
    )
    ap.add_argument("-t", "--text", help="text to encode", required=True)
    ap.add_argument("-i", "--image", help="path to an image file", required=True)
    args = ap.parse_args()

    clip = Clip(args.model, args.verbosity)

    tokens = clip.tokenize(args.text)
    text_embed = clip.encode_text(tokens)

    image_embed = clip.load_preprocess_encode_image(args.image)

    score = clip.calculate_similarity(text_embed, image_embed)

    # Alternatively, you can just do:
    # score = clip.compare_text_and_image(text, image_path)

    print(f"Similarity score: {score}")
