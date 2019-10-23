# Benchmarks

## Micro-Benchmarks (Training + Validation)

The models presented in the paper are trained (and validated) using synthetic benchmarks from:

* [joaofilipedg-gpupowermodel](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/datasets/microbenchmarks/joaofilipedg-gpupowermodel) - Taken from [joaofilipedg-gpupowermodel](https://github.com/hpc-ulisboa/gpupowermodel/tree/master/v2.0_TPDS2019/microbenchmarks)
* [akhilarunkumar-GPURelease](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/datasets/microbenchmarks/akhilarunkumar-GPURelease) - Taken from [akhilarunkumar-GPURelease](https://github.com/akhilarunkumar/GPUJoule_release)

To compile (for GPU with 7.5 compute capability):
    ```
    make all ptx ARCH=75
    ```

## Standard Benchmarks (Testing)
Taken from:
* [NVIDIA-SDK](https://developer.nvidia.com/cuda-code-samples)
* [Parboil](http://impact.crhc.illinois.edu/parboil/parboil_download_page.aspx)
* [Polybench](http://web.cs.ucla.edu/~pouchet/software/polybench/)
* [Rodinia](http://lava.cs.virginia.edu/Rodinia/download_links.htm)
* [SHOC](https://github.com/vetter/shoc)
