#!/bin/bash

# Change to the project directory
cd "$(dirname "$0")"/..

# Check if the main directory argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <main_directory>"
  exit 1
fi

# Get the main directory from the command line argument
main_directory="$1"
output_prefix="ggml_"

# Check if the main directory exists
if [ ! -d "$main_directory" ]; then
  echo "Main directory does not exist: $main_directory"
  exit 1
fi

# Iterate over immediate subdirectories of the main directory
for subdirectory in "$main_directory"/*/; do
  if [ -d "$subdirectory" ]; then
    echo "Processing subdirectory: $subdirectory-gguf"
        
    # Extract the subdirectory name without path
    subdirectory_name=$(basename "$subdirectory")
    
    # Create the output directory name with the prefix
    output_directory="$main_directory/$output_prefix$subdirectory_name"

    # Check if the subdirectory name contains "CLIP-ViT-L-14-laion2B-s32B-b82K"
    if [[ $subdirectory_name == *"CLIP-ViT-L-14-laion2B-s32B-b82K"* ]]; then
      extra_args="--image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5"
    else
      extra_args=""
    fi

    # Call the conversion script with the subdirectory, output_directory, and extra arguments
    python3 ./models/convert_hf_to_gguf.py -m "$subdirectory" -o "$output_directory" $extra_args
    python3 ./models/convert_hf_to_gguf.py -m "$subdirectory" -o "$output_directory" --text-only $extra_args
    python3 ./models/convert_hf_to_gguf.py -m "$subdirectory" -o "$output_directory" --vision-only $extra_args
    python3 ./models/convert_hf_to_gguf.py -m "$subdirectory" -o "$output_directory" --use-f32 $extra_args
    python3 ./models/convert_hf_to_gguf.py -m "$subdirectory" -o "$output_directory" --use-f32 --text-only $extra_args
    python3 ./models/convert_hf_to_gguf.py -m "$subdirectory" -o "$output_directory" --use-f32 --vision-only $extra_args

    cp ./scripts/hf-readme.md "$output_directory/README.md"
    
  fi
done
