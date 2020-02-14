# ConvAcc
Accelerating convolution using numba, cupy and xnor in python.

**Numba** is a just-in-time, type-specializing, **function compiler** for accelerating numerically-focused Python. It can be  typically enabled by applying a **decorator** to a python function and can compile your code for CPU or GPU. It uses **LLVM** to compile python functions **just-in-time**, under the hood. **Cupy** is a numpy-like library accelerated with CUDA. It's syntax is very similar to numpy and in most cases you can directly replace the numpy import with cupy. It allows us to write **custom kernels** in CUDA and can be easily used with numba CUDA functions.The deep learning library **chainer** uses cupy in it's backend.

In **XNOR convolution**, both the filters and the input to convolutional layers are **binary**. Now, by approximating the  convolution operations with **XNOR and bitcounting** operations, we can gain massive **speed-up and memory savings**. Even though this seems straight-forward(theoretically),in practice  an efficient implementation of **bitpacking**, approximation techniques and  **training mechanisms** are required to acheive sufficient **accuracy** and speed on conventional hardware platforms.

In the IPython Notebook, we try to implement a **basic convolution** using python and subsequently improve it's speed using **numba** and other **optimization** techniques. Finally,we **compare** and benchmark the various techniques in python for CPU and GPU in terms of **execution speed**. The notebook can be directly run on **google colaboratory** ,using a GPU runtime without any additional installaton of libraries or packages.

**Note**: The benchmarks heavily depends on the **hardware and library versions** used for experimentation.

## Dependencies

* Numba
* Cupy
* CUDA

## References

* [Boost Python With GPU](https://thedatafrog.com/en/boost-python-gpu)
* [Numba: Talks and Tutorials](https://numba.pydata.org/numba-doc/dev/user/talks.html) 
* [ContinuumIO:GTC 2020 Numba ](https://github.com/ContinuumIO/gtc2020-numba)
* [Create CUDA kernels from Python using Numba and CuPy](https://www.youtube.com/watch?v=CQDsT81GyS8)
* [Numba Cuda: Fast Matrix Multiplication](https://numba.pydata.org/numba-doc/dev/cuda/examples.html)
* [Python: Popcount Benchmark](https://gist.github.com/nixeneko/036df003dd985ce7fa4e2c894f055d17)
* [Understanding Binary Neural Networks](https://sushscience.wordpress.com/2017/10/01/understanding-binary-neural-networks/)
* [Binary Neural Networks](https://minjekim.com/demo_bnn.html)
* [Python: Packing Bitlist to UINT64](https://stackoverflow.com/questions/60118227/python-fastest-way-of-packing-a-2d-array-of-binary-values-into-uint64-array)
* [Python: Im2col Implementation](https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python)
* [Numba: Automatic Parallelization](https://numba.pydata.org/numba-doc/dev/user/parallel.html)
* [Numba: Using Stencils](https://numba.pydata.org/numba-doc/dev/user/stencil.html)
* [Binarization of Low Level Operations in DNN](https://edu.authorcafe.com/academies/7718/binarization-of-low-level-operations-in-deep-neural-networks)
* [Stanford:CS231n Im2Col Assignment](https://github.com/ShibiHe/Stanford-CS231n-assignments/blob/master/assignment3/cs231n/im2col.py)
* [Introduction to CUDA in Python](https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3)
