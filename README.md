# clip.cpp
CLIP inference in plain C/C++ with no extra dependencies

## WARNING
This is not ready for use yet, and it needs some extra work to prepare examples and do benchmarks.

## Note about image preprocessing
PIL uses a two-pass convolutions-based bicubic interpolation in resizing with antialiasing applied. In Pytorch, antialiasing is optional. It needs some extra attention to implement these preprocessing logic that matches their results numerically. However, I found that linear interpolation is also good enough for some initial tests. So let's use it until we craft a proper bicubic interpolation.

## Roadmap
- [x] Implement Quick GELU (see [PR](https://github.com/ggerganov/ggml/pulls/254)).
- [x] Implement causal attention in text model and varify its correctness.
- [x] Implement tokenization.
- [x] Return normalize text and image embeddings.
- [x] Introduce functions to compute similarity.
- [ ] Seperate memory buffers for text and image models, as their memory requirements are different.
- [ ] Implement proper bicubic interpolation (PIL uses a convolutions-based algorithm, and it's more stable than affine transformations).
- [ ] Do benchmarks and announce the results.
