# gpuPTXParser

## 1. gpuPTXParser Tool

``gpuPTXParser`` is a command line tool that can be used for reading [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) files and extracting the number of occurrences of each different instructions per GPU kernel. The tool can also extract the sequence of instructions of the kernels in the source file.

* Usage:
```bash
gpuPTXParser.py <PATH_TO_ISA_FILES> <APPLICATION.ptx> [--histogram] [--v]
```

* Arguments:

    ``<PATH_TO_ISA_FILES>``:  PATH to the directory with the isa files (ptx_isa.txt, ptx_state_spaces.txt, ptx_instruction_types.txt).

    ``<APPLICATION.ptx>`` :  .ptx file to be parsed.

* Options:

    ``--histogram`` : output a pdf file with the histogram of instructions used per kernel (default: False).

    ``--v`` : turn on verbose mode (default: False).

* Example:
```bash
gpuPTXParser.py aux_files/ Microbenchmarks/pure_DRAM/DRAM.ptx --histogram
```

* Output Files:

    1 ``outputOccurrences_per_kernel.csv`` file : with the count of occurrences of each instruction in the PTX ISA in each kernel from the parsed .ptx file. 1 row for each kernel. 1 column for each instruction.

    1 ``outputSequenceReadable_kernel_i.csv`` file for each kernel ``i`` in the parsed .ptx file.

    1 ``outputSequence_kernel_i.csv`` file for each kernel ``i`` in the parsed .ptx file. Values encoded.

## Contact
If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt.

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
