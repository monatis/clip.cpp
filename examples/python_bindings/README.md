# Python bindings for clip.cpp

This package provides basic Python bindings for [clip.cpp](https://github.com/monatis/clip.cpp).

It requires no third-party libraries and no big dependencies such as PyTorch, TensorFlow, Numpy, ONNX etc.

## Install
If you are on a X64 Linux distribution, you can simply Pip-install it:

```sh
pip install clip_cpp
```


If you are on another operating system or architecture,
or if you want to make use of support for instruction sets other than AVX2 (e.g., AVX512),
you can build it from source.
Se [clip.cpp](https://github.com/monatis/clip.cpp) for more info.

All you need to do is to compile with the `-DBUILD_SHARED_LIBS=ON` option and copy `libclip.so` to `examples/python_bindings/clip_cpp`.

## Usage
## Clip Class

The `Clip` class provides a Python interface to clip.cpp, allowing you to perform various tasks such as text and image encoding, similarity scoring, and text-image comparison. Below are the constructor and public methods of the `Clip` class:

### Constructor

```python
def __init__(self, model_file: str, verbosity: int = 0):
```

- **Description**: Initializes a `Clip` instance with the specified CLIP model file and optional verbosity level.
- `model_file` (str): The path to the CLIP model file.
- `verbosity` (int, optional): An integer specifying the verbosity level (default is 0).

### Public Methods

#### 1. `vision_config`

```python
@property
def vision_config(self) -> Dict[str, Any]:
```

- **Description**: Retrieves the configuration parameters related to the vision component of the CLIP model.

#### 2. `text_config`

```python
@property
def text_config(self) -> Dict[str, Any]:
```

- **Description**: Retrieves the configuration parameters related to the text component of the CLIP model.

#### 3. `tokenize`

```python
def tokenize(self, text: str) -> List[int]:
```

- **Description**: Tokenizes a text input into a list of token IDs.
- `text` (str): The input text to be tokenized.

#### 4. `encode_text`

```python
def encode_text(
    self, tokens: List[int], n_threads: int = os.cpu_count(), normalize: bool = True
) -> List[float]:
```

- **Description**: Encodes a list of token IDs into a text embedding.
- `tokens` (List[int]): A list of token IDs obtained through tokenization.
- `n_threads` (int, optional): The number of CPU threads to use for encoding (default is the number of CPU cores).
- `normalize` (bool, optional): Whether or not to normalize the output vector (default is `True`).

#### 5. `load_preprocess_encode_image`

```python
def load_preprocess_encode_image(
    self, image_path: str, n_threads: int = os.cpu_count(), normalize: bool = True
) -> List[float]:
```

- **Description**: Loads an image, preprocesses it, and encodes it into an image embedding.
- `image_path` (str): The path to the image file to be encoded.
- `n_threads` (int, optional): The number of CPU threads to use for encoding (default is the number of CPU cores).
- `normalize` (bool, optional): Whether or not to normalize the output vector (default is `True`).

#### 6. `calculate_similarity`

```python
def calculate_similarity(
    self, text_embedding: List[float], image_embedding: List[float]
) -> float:
```

- **Description**: Calculates the similarity score between a text embedding and an image embedding.
- `text_embedding` (List[float]): The text embedding obtained from `encode_text`.
- `image_embedding` (List[float]): The image embedding obtained from `load_preprocess_encode_image`.

#### 7. `compare_text_and_image`

```python
def compare_text_and_image(
    self, text: str, image_path: str, n_threads: int = os.cpu_count()
) -> float:
```

- **Description**: Compares a text input and an image file, returning a similarity score.
- `text` (str): The input text.
- `image_path` (str): The path to the image file for comparison.
- `n_threads` (int, optional): The number of CPU threads to use for encoding (default is the number of CPU cores).

#### 8. `__del__`

```python
def __del__(self):
```

- **Description**: Destructor that frees resources associated with the `Clip` instance.

With the `Clip` class, you can easily work with the CLIP model for various natural language understanding and computer vision tasks.

## Example
A basic example can be found in the [clip.cpp examples](https://github.com/monatis/clip.cpp/blob/main/examples/python_bindings/example_main.py).

```
python example_main.py --help                                
usage: clip [-h] -m MODEL [-v VERBOSITY] -t TEXT -i IMAGE                                                               
                                                                                                                        
optional arguments:                                                                                                     
  -h, --help            show this help message and exit                                                                 
  -m MODEL, --model MODEL                                                                                               
                        path to GGML file                                                                               
  -v VERBOSITY, --verbosity VERBOSITY                                                                                   
                        Level of verbosity. 0 = minimum, 2 = maximum                                                    
  -t TEXT, --text TEXT  text to encode                                                                                  
  -i IMAGE, --image IMAGE                                                                                               
                        path to an image file                                                                           
```

Bindings to the DLL are implemented in `clip_cpp/clip.py` and 
