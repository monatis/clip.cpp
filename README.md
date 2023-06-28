# clip.cpp
CLIP inference in plain C/C++ with no extra dependencies

## Description
This is a dependency free implementation of well known [CLIP](https://github.com/openai/clip) by OpenAI,
thanks to the great work in [GGML](https://github.com/ggerganov/ggml).
You can use it to work with CLIP models from both OpenAI and LAION
in Transformers format.

## Motivation
CLIP is deployed for several task from semantic image search to zero-shot image labeling.
It's also a part of Stable Diffusion and and the recently emerging field of large multimodal models (LMM).
This repo is aimed at powering useful applications based on such models on computation- or memory-constraint devices.
4-bit quantized CLIP is only 85.6 MB!

clip.cpp also has a short startup time compared to large ML frameworks, which makes it suitable for serverless deployments where the cold start is an issue.

## Note about image preprocessing
PIL uses a two-pass convolutions-based bicubic interpolation in resizing with antialiasing applied. In Pytorch, antialiasing is optional. It needs some extra attention to implement this preprocessing logic that matches their results numerically. However, I found that linear interpolation is also good enough for both comparison of different embeddings from this implementation and also comparison of an embedding from this implementation and another one from Transformers. So let's use it until we craft a proper bicubic interpolation.


## Preconverted Models
Preconverted Models can be found in [HuggingFace Repositories tagged with `clip.cpp`](https://huggingface.co/models?other=clip.cpp).
If you want to do conversion yourself for some reason, see below for how.
Otherwise, download a model of your choice from the link above and then feel free to jump to the building section.

## Model conversion
You can convert CLIP models from OpenAI and LAION in Transformers format. Apparently, LAION's models outperform OpenAI models in several benchmarks, so they are recommended.

1. Clone the model repo from HF Hub:

```shell
git lfs init

git clone https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

2. Clone this repository:

```shell
git clone --recurse-submodules https://github.com/monatis/clip.cpp.git

cd clip.cpp/models
```

3. You need to install pytorch and transformers packages if you haven't already. Otherwise, you can skip this step:

```shell
pip install -r requirements.txt
```

4. Use `models/convert_hf_to_ggml.py` to convert it to GGML format:

```shell
python convert_hf_to_ggml.py ../../CLIP-ViT-B-32-laion2B-s34B-b79K 1
```

The output `ggml-model-f16.bin` file is in the model directory specified in the command above.

## Building
```shell
git clone --recurse-submodules https://github.com/monatis/clip.cpp.git

cd clip.cpp

mkdir build

cd build

cmake -DCLIP_NATIVE=ON ..

make
```

And the binaries are in the `./bin` directory.

TODO: Detail other build optimizations.

## Quantization
`clip.cpp` currently supports q4_0 and q4_1 quantization types.
You can quantize a model in F16 to one of these types by using the `./bin/quantize` binary.

```
usage: ./bin/quantize /path/to/ggml-model-f16.bin /path/to/ggml-model-quantized.bin type                                 
  type = 2 - q4_0                                                                                                       
  type = 3 - q4_1                                                                                                       
```

For example, you can run the following to convert the model to q4_0:

```shell
./bin/quantize ./CLIP-ViT-B-32-laion2B-s34B-b79K/ggml-model-f16.bin ./CLIP-ViT-B-32-laion2B-s34B-b79K/ggml-model-q4_0.bin 2
```

Now you can use `ggml-model-q4_0.bin` just like the model in F16.

## Usage
Currently we have two examples: `main` and `zsl`.

1. `main` is just for demonstrating the usage of API and optionally print out some verbose timings. It simply calculates the similarity between one image and one text passed as CLI args.

```
Usage: ./bin/main [options]                                                                                             
                                                                                                                        
Options:  -h, --help: Show this message and exit                                                                        
  -m <path>, --model <path>: path to model. Default: models/ggml-model-f16.bin                                          
  -t N, --threads N: Number of threads to use for inference. Default: 4                                                 
  --text <text>: Text to encode. At least one text should be specified                                                  
  --image <path>: Path to an image file. At least one image path should be specified                                    
  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: 1                    
```

2. `zsl` is a zero-shot image labeling example. It labels an image with one of the labels.
The CLI args are the same as in `main`,
but you must specify multiple `--text` arguments to specify the labels.

## Benchmarking
You can use the benchmarking utility to compare the performances of different checkpoints and quantization types.

```
usage: ./bin/benchmark <model_path> <images_dir> <num_images_per_dir> [output_file]                                     
                                                                                                                        
image_path: path to CLIP model in GGML format                                                                           
images_dir: path to a directory of images where images are organized into subdirectories named classes                  
num_images_per_dir: maximum number of images to read from each one of subdirectories. if 0, read all files              
output_file: optional. if specified, dump the output to this file instead of stdout                                     
```


TODO: share benchmarking results for a common dataset later on.

## Future Work
- [ ] Support `text-only`, `image-only` and `both` (current) options when exporting, and modify model loading logic accordingly. It might be relevant to use a single modality in certain cases, as in large multimodal models, or building and/or searching for semantic image search.
- [ ] Seperate memory buffers for text and image models, as their memory requirements are different.
- [ ] Implement proper bicubic interpolation (PIL uses a convolutions-based algorithm, and it's more stable than affine transformations).
