# Python bindings

This example provide basic Python bindings for clip.cpp.
It requires no third-party libraries.
All you need to do is to compile with the `-DBUILD_SHARED_LIBS=ON` option and copy `libclip.so` to `examples/python_bindings/clip_cpp`.
Bindings to the DLL are implemented in `clip_cpp/clip.py` and a basic usage is in `example_main.py`.

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

## TODO
- [ ] Better documentation and publish `clip_cpp` as a Pip-installable package.
- [ ] Provide a more interesting example in Python.
