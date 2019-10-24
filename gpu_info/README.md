## GPU Information

One directory for each of the GPU devices considered in the paper:

* [GTX Titan X](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/gpu_info/gtxtitanx): from NVIDIA Maxwell microarchitecture.
* [Titan Xp](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/gpu_info/titanxp): from NVIDIA Pascal microarchitecture.
* [Titan V](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/gpu_info/titanv): from NVIDIA Volta microarchitecture.
* [Tesla T4](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/gpu_info/teslat4): from NVIDIA Turing microarchitecture.

## Required Files

To characterize a GPU device the following files are required (example when there are two memory frequency levels):
* ``clks_mem_<GPUNAME>.txt``: with the allowed memory frequency levels (one per line) and the default memory level (last line);
* ``clks_core_<GPUNAME>_<MEMLEVEL0>.txt``: with the allowed core frequency levels with fmem=\<MEMLEVEL0\> (one per line) and the default memory level (last line);
* ``clks_core_<GPUNAME>_<MEMLEVEL1>.txt``: with the allowed core frequency levels with fmem=\<MEMLEVEL1\> (one per line) and the default memory level (last line);
* ``idle_pows_<GPUNAME>.txt``: idle power consumption of each allowed frequency level (one per line)

## Contact
If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt.

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
