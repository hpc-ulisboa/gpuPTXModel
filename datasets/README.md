# Datasets

Here are provided the datasets used to train (and later test) the models presented in the paper.
The PTX information is obtained at compile-time, using the ``nvcc`` compiler with flags the appropriate flags.
Example:
```
nvcc -O0 -Xcompiler -O0 -Xptxas -O0 -gencode=arch=compute_70,code=compute_70 -ptx -o dp.ptx dp.cu
```

## Directory structure

* ``gtxtitanx`` - Results obtained on the NVIDIA GTX Titan X GPU, from the Maxwell microarchitecture.
