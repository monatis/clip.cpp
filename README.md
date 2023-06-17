# clip.cpp
CLIP inference in plain C/C++ with no extra dependencies

## WARNING
This is not recommended for production yet, and and it needs some extra work to prepare examples and do benchmarks. Also see the node below.

## Note about image preprocessing
PIL uses a two-pass convolutions-based bicubic interpolation in resizing with antialiasing applied. In Pytorch, antialiasing is optional. It needs some extra attention to implement this preprocessing logic that matches their results numerically. However, I found that linear interpolation is also good enough for both comparison of different embeddings from this implementation and also comparison of an embedding from this implementation and another one from Transformers. So let's use it until we craft a proper bicubic interpolation.

## Model conversion
You can convert CLIP models from OpenAI and LAION in Transformers format. Apparently, LAION's models outperform OpenAI models in several benchmarks, so they are recommended.

1. Clone the model repo from HF Hub:

```shell
git lfs init.git
git clone https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

2. Use `models/convert_hf_to_ggml.py` to convert it to GGML format.

```shell
python convert_hf_to_ggml.py CLIP-ViT-B-32-laion2B-s34B-b79K 1
```

The output `ggml-model-f16.bin` file is in the same directory.

## Building
```shell
git clone --recurse-submodules https://github.com/monatis/clip.cpp.git

cd clip.cpp

mkdir build

cd build

cmake -DCLIP_NATIVE=ON ..

make
```

And the main example is in `./bin/main`.

TODO: Detail other build optimizations.

## Usage
```
Usage: ./bin/main [options]                                                                                             
                                                                                                                        
Options:  -h, --help: Show this message and exit                                                                        
  -m <path>, --model <path>: path to model. Default: models/ggml-model-f16.bin                                          
  -t N, --threads N: Number of threads to use for inference. Default: 4                                                 
  --text <text>: Text to encode. At least one text should be specified                                                  
  --image <path>: Path to an image file. At least one image path should be specified                                    
```
## Roadmap
- [ ] Support `text-only`, `image-only` and `both` (current) options when exporting, and modify model loading logic accordingly. It might be relevant to use a single modality in certain cases, as in large multimodal models.
- [ ] Seperate memory buffers for text and image models, as their memory requirements are different.
- [ ] Implement proper bicubic interpolation (PIL uses a convolutions-based algorithm, and it's more stable than affine transformations).
- [ ] Do benchmarks and announce the results.
