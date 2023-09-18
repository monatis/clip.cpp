"""
Small script to get and parse imagenet1k dataset into benchmark format

Dataset comments
    Change classes names containing "/" to "or"
    Some classes have '/' in their name
    For compatibility with folder benchmarks we replace them with 'or'
    Skip classes (744, missiles) and (837, sunglasses) as they are duplicates

"""

import argparse
import json
import os
from pathlib import Path
import shutil
from subprocess import call
from torchvision.datasets import ImageNet


# Files
_CLASSNAMES_FILENAME = "classnames.json"
_CLASSTEMPLATES_FILENAME = "class_templates.json"
_DEVKIT_FILENAME = "ILSVRC2012_devkit_t12.tar.gz"
_IMG_VAL_FILENAME = "ILSVRC2012_img_val.tar"

# Name for folder with final dataset
_PROCESSED_DIR_NAME = "dataset"


def download_dataset(path: Path, verbose: bool = False):
    if verbose:
        print("Downloading dataset")
    path.mkdir(exist_ok=True, parents=True)

    dk_output_path = path / _DEVKIT_FILENAME
    iv_output_path = path / _IMG_VAL_FILENAME

    template_path = path / _CLASSTEMPLATES_FILENAME
    classnames_path = path / _CLASSNAMES_FILENAME

    if not dk_output_path.exists():
        if verbose:
            print("\tDidnt find devkit file, downloading..")
        call(
            (
                f"wget https://image-net.org/data/ILSVRC/2012/{_DEVKIT_FILENAME} "
                + f"--output-document={dk_output_path}"
            ),
            shell=True,
        )
    else:
        if verbose:
            print("\tFound devkit file, skipping download..")

    if not iv_output_path.exists():
        if verbose:
            print("\tDidnt find image validation file, downloading..")
        call(
            (
                f"wget https://image-net.org/data/ILSVRC/2012/{_IMG_VAL_FILENAME} "
                + f"--output-document={iv_output_path}"
            ),
            shell=True,
        )
    else:
        if verbose:
            print("\tFound image validation file, skipping download..")

    if not template_path.exists():
        if verbose:
            print("\tDidnt find class templates file, downloading..")
        call(
            (
                "wget "
                + "https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/clip_benchmark/datasets/en_zeroshot_classification_templates.json "
                + f"--output-document={template_path}"
            ),
            shell=True,
        )

        class_templates = json.load(template_path.open("r"))
        class_templates = class_templates["imagenet1k"]
        json.dump(class_templates, template_path.open("w"), indent=2)
    else:
        if verbose:
            print("\tFound class templates file, skipping download..")

    if not classnames_path.exists():
        if verbose:
            print("\tDidnt find class names file, downloading..")
        call(
            (
                "wget "
                + "https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/clip_benchmark/datasets/en_classnames.json "
                + f"--output-document={classnames_path}"
            ),
            shell=True,
        )
        classnames = json.load(classnames_path.open("r"))
        classnames = classnames["imagenet1k"]

        if verbose:
            print(
                "\tFixing classnames, replacing '/' with 'or' and removing duplicates.."
            )
        # Described in top comment section
        classnames = [
            c.replace("/", "or")
            for i, c in enumerate(classnames)
            if i not in [744, 837]
        ]

        json.dump(classnames, classnames_path.open("w"), indent=2)


def parse_dataset(path: Path, verbose=False):
    if verbose:
        print("Parsing dataset")
    # Load cases
    classes_path = path.joinpath(_CLASSNAMES_FILENAME)
    classes = json.load(classes_path.open("r"))

    # Check if dataset has already been processed
    processed_dataset_path = path / _PROCESSED_DIR_NAME
    dataset_exists = all(processed_dataset_path.joinpath(c).exists() for c in classes)

    if dataset_exists:
        return processed_dataset_path

    processed_dataset_path.mkdir(exist_ok=True)

    # ImageNet dataset handles the parsing
    if verbose:
        print("\tUnpacking dataset, this can take a bit..")
    ds = ImageNet(root=path, split="val")

    # Track with counter as some classes are removed from classes
    cls_index = 0
    for i, dir_name in enumerate(ds.wnids):
        if dir_name in ["n04356056", "n04008634"]:
            if verbose:
                print("\tSkipped class", ds.classes[i])
            continue

        class_name = classes[cls_index]
        src_dir = Path(ds.split_folder).joinpath(dir_name)
        dst_dir = processed_dataset_path.joinpath(class_name)

        os.rename(src=src_dir, dst=dst_dir)
        if verbose:
            print(f"\tMoved class: {ds.classes[i]} to {class_name}")

        cls_index += 1

    # Remove other files
    shutil.rmtree(ds.split_folder)
    if verbose:
        print("\tCleaned up unpacked dataset folder")

    return processed_dataset_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    path = Path(args.save_path).absolute()

    download_dataset(path=path, verbose=args.verbose)
    dataset_path = parse_dataset(path=path, verbose=args.verbose)
    print(f"Dataset is ready at {dataset_path}")
