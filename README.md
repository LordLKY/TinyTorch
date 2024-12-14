# TinyTorch

## About

TinyTorch is a tiny, torch-like deep learning framework, which is based on the labs of [CMU 10-414/714 Deep Learning Systems](https://dlsyscourse.org/lectures/). It consists of a Python frontend and 3 different backends including CPU(CPP), CPU(Python numpy), and GPU(CUDA). Simple models such as MLP, CNN can be implemented using this framework.

## Build

TinyTorch is built by CMake. Windows and Visual Studio is recommended for building (or you can modify the project in several places and build it on Linux with Make). It may be helpful to add "Python_EXECUTABLE" in order to specify your Python interpreter when using CMake and make sure pybind11 is equipped.

## Usage

Use `import tinytorch as ttorch` to get started. Make sure the libraries of backends named 'ndarray_backend_cpu' and 'ndarray_backend_cuda' are imported properly in [ndarray.py](https://github.com/LordLKY/TinyTorch/blob/main/python/tinytorch/backend_ndarray/ndarray.py).

There 2 examples in ./demo to show how to use TinyTorch. One is to train a MLP on MNIST and the other is to train a simple CNN on CIFAR-10. Both of them can achieve acceptable accuracy after training for several minutes.

### TODO

- [ ] Implement more models like RNN, Transformer, etc.
- [ ] Persue more efficient implementation of backends

Feel free to contribute to this project if you have any ideas or suggestions.