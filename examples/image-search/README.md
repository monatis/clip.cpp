# Image search

This example implements basic semantic image search using [usearch](https://github.com/unum-cloud/usearch) as a vector database for accelerated similarity search.

Use `image-search-build` to build the database of images and their embeddings beforehand. Currently it does not support updating.

Use `image-search` to search for indexed images by semantic similarity.

### examples

#### build db

help:
```sh
./image-search-build -h
Usage: ./image-search-build [options] dir/with/pictures [more/dirs]

Options:  -h, --help: Show this message and exit
  -m <path>, --model <path>: path to model. Default: ../models/ggml-model-f16.bin
  -t N, --threads N: Number of threads to use for inference. Default: 4
  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: 1
```

creating db for `tests/`:
```sh
./image-search-build -m models/openai_clip-vit-base-patch32.ggmlv0.f16.bin ./tests/
```

#### search by text

help:
```sh
./image-search -h
Usage: ./image-search [options] <search string or /path/to/query/image>

Options:  -h, --help: Show this message and exit
  -m <path>, --model <path>: overwrite path to model. Read from images.paths by default.
  -t N, --threads N: Number of threads to use for inference. Default: 4
  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: 1
  -n N, --results N: Number of results to display. Default: 5
```

searching for `apple` in the db in the current directory:
```sh
./image-search apple
clip_model_load: loading model from 'models/openai_clip-vit-base-patch32.ggmlv0.f16.bin' - please wait....................................................clip_model_load: model size =   288.93 MB / num tensors = 397
clip_model_load: model loaded

search results:
distance path
  0.674587 /home/xxxx/tests/red_apple.jpg
  0.785591 /home/xxxx/tests/white.jpg
```

note: lower score for search results is better as it indicates the distance, not the similarity.

