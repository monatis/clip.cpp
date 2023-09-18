## Tests

You can use `prepare_imagenet1k.py` to download and prepare the imagenet1k dataset
in a format expected by the `benchmark` utility.
If you haven't already, you need to install torch and torchvision to
use this Python script:

```sh
pip install -r requirements.txt
```

## Note about benchmark results
Please note that the results in this benchmark do not match those reported in the open-clip repository because:

1. Most importantly, they use a different test protocol that includes averaging vectors of text templates etc.
2. There are still gatchas in the tokenization implementation in this repo.
3. This repo uses a linear interpolation instead of bicubic in image preprocessing.

The 2nd and 3rd items will be fixed soon.
I don't agree with their test protocol, so I am not so motivated to fix the first item.
