# clip.cpp
CLIP inference in plain C/C++ with no extra dependencies

## WARNING
This is not ready for use yet, and it needs some more work to implement tokenization and bicubic interpolation that matches PIL's implementation numerically.

## Roadmap
- [x] Implement Quick GELU (see [PR](https://github.com/ggerganov/ggml/pulls/254)).
- [x] Implement causal attention in text model and varify its correctness.
- [ ] Implement tokenization.
- [ ] Implement proper bicubic interpolation (PIL uses a convolutions-based algorithm, and it's more stable than affine transformations).
- [ ] Seperate memory buffers for text and image models, as their memory requirements are different.
- [ ] Do benchmarks and announce the results.
