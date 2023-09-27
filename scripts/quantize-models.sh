#!/bin/bash

# Change to the build directory
cd "$(dirname "$0")"/../build/

# Function to call the quantize script with arguments
quantize_model() {
  input_file="$1"
  output_file="$2"
  quantization_level="$3"
  echo "Quantizing model file: $input_file"
  ./bin/quantize "$input_file" "$output_file" "$quantization_level"
}

# Check if the main directory argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <main_directory>"
  exit 1
fi

# Get the main directory from the command line argument
main_directory="$1"

# Check if the main directory exists
if [ ! -d "$main_directory" ]; then
  echo "Main directory does not exist: $main_directory"
  exit 1
fi

# Iterate over immediate subdirectories starting with 'ggml_'
for subdirectory in "$main_directory"/ggmlq_*; do
  if [ -d "$subdirectory" ]; then
    echo "Processing subdirectory: $subdirectory"
    
    # Iterate over model files ending in 'f32.gguf' in the subdirectory
    for model_file in "$subdirectory"/*f32.gguf; do
      if [ -f "$model_file" ]; then
        # Define the output file with a new ending (e.g., replace 'f32.gguf' with the type of quantization)
        output_file_base="${model_file%f32.gguf}"
        
        # Quantize the model for different quantization levels
        quantize_model "$model_file" "${output_file_base}q8_0.gguf" 8
        quantize_model "$model_file" "${output_file_base}q5_1.gguf" 7
        quantize_model "$model_file" "${output_file_base}q5_0.gguf" 6
        quantize_model "$model_file" "${output_file_base}q4_1.gguf" 3
        quantize_model "$model_file" "${output_file_base}q4_0.gguf" 2
      fi
    done
  fi
done
