# gpuPTXModel - GPU Static Modeling using PTX and Deep Structured Learning

<p align="center"><img width="100%" src="png/model.png" /></p>

## Directory structure

* ``assembly_info`` - files with Assembly ISA (list of possible instructions) and available modifiers;
* ``benchmarks`` - information on considered benchmarks, including source code of the microbenchmarks;
* ``datasets`` - datasets used to train (and validate) the models;
* ``gpu_info`` - files that characterize the architecture to be considered in the trained models (number of freq. domains, number of levels in each domain, etc.);
* ``model_configs`` - configurations of the models;
* ``src`` - gpuPTXModel source auxiliary files.

### Tools
- ``gpuPTXModel`` - main command line tool that trains static models (Performance+Power+Energy) based on a given dataset for a specific GPU;
- ``toolReadBenchs`` - command line tool that reads dataset and aggregates in suitable format;
- ``gpuPTXParser`` - command line tool that can read PTX files and output the required information to be used in the proposed models;


## 1. gpuPTXModel Tool

``gpuPTXModel`` is a command line tool that allows creating DVFS-aware GPU static models based solely on the sequence of [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) instructions in the kernel code.
The proposed models, published in [IEEE Access](https://ieeexplore.ieee.org/document/8890640), implemented using recurrent neural networks (LSTM-based), take into account the sequence of GPU assembly instructions and can be used to accurately predict changes in the execution time, power and energy consumption of applications when the frequencies of different GPU domains (core and memory) are scaled.

To train the models, the tool receives as argument the path to the microbenchmark (and optionally the testing dataset), which need to have been properly aggregated using the [``toolReadBenchs``](https://github.com/hpc-ulisboa/gpuPTXModel#2-toolreadbenchs-tool) tool.

If you use the ``gpuPTXModel`` tool in a publication, please cite [[1]](#4-references).

* Usage:
```bash
gpuPTXModel.py <PATH_TO_MICROBENCHMARK_DATASET> <GPU_NAME> [--test_data_path <PATH_TO_TESTING_DATASET>] [--device <cpu|gpu>] [--device_id <ID>] [--encoder_file <FILE>]  [--time_dvfs_file <FILE>] [--pow_dvfs_file <FILE>] [--energy_dvfs_file <FILE>] [--no_time_dvfs] [--no_pow_dvfs] [--no_energy_dvfs] [--num_epochs <NUM_EPOCHS>] [--v]
```

* Arguments:

    ``<PATH_TO_MICROBENCHMARK_DATASET>`` : PATH to the directory with the microbenchmark profilling data (to be used in training).

    ``<GPU_NAME>`` : name of the GPU device the dataset was executed on.
* Options:

    ``--test_data_path <PATH_TO_TESTING_DATASET>`` : PATH to the directory with the testing dataset (to be later used for testing the models) (default: '').    

    ``--device <cpu|gpu>`` : select the device where the training will execute (default: gpu).

    ``--device_id <id>`` : target a specific device (default: 0).

    ``--encoder_file <FILE>`` : name of the file with the encoder configuration (default: '').

    ``--time_dvfs_file <FILE>`` : name of the file with the Time FNN configuration (default: '').

    ``--pow_dvfs_file <FILE>`` : name of the file with the Power FNN configuration (default: '').

    ``--energy_dvfs_file <FILE>`` : name of the file with the Energy FNN configuration (default: '').

    ``--no_time_dvfs`` : to turn off Time FNN, i.e., not use it.

    ``--no_pow_dvfs`` : to turn off Power FNN, i.e., not use it.

    ``--no_energy_dvfs`` : to turn off Energy FNN, i.e., not use it.

    ``--num_epochs <NUM_EPOCHS>`` : to select the maximum number of epochs to use during training (default: 20).

    ``--v`` : turn on verbose mode (default: False).

* Example:
```bash
  gpuPTXModel.py Microbenchmarks/Outputs/ --device gpu --model_name LSTM --num_layers 2 --learning_rate 0.001 --num_epochs 50
```

## 2. toolReadBenchs Tool

``toolReadBenchs`` is a command line tool that can be used for reading the measured values (execution times and power consumption) and organizing them in the format that can be used by the main ``gpuPTXModel`` tool.
The tool also creates .pdf files with the plots of the measured values across the different frequency levels.

If you use the ``toolReadBenchs`` tool in a publication, please cite [[1]](#4-references).

* Usage:
```bash
toolReadBenchs.py <PATH_TO_MICROBENCHMARK_DATASET> <GPU_NAME> [--benchs_file <MICROBENCHMARK_NAMES>] [--test_data_path <PATH_TO_TESTING_DATASET>] [--benchs_test_file <TESTING_NAMES>] [--tdp <TDP_VALUE>] [--o] [--v]
```

* Arguments:

    ``<PATH_TO_MICROBENCHMARK_DATASET>`` : PATH to the directory with the microbenchmark dataset (to be later used for training/validation the models).

    ``<GPU_NAME>`` : name of the GPU device the dataset was executed on.

* Options:

    ``--benchs_file <MICROBENCHMARK_NAMES>`` : provide file with names of the microbenchmarks (default: all).

    ``--test_data_path <PATH_TO_TESTING_DATASET>`` : PATH to the directory with the testing dataset (to be later used for testing the models) (default: '').

    ``--benchs_test_file <TESTING_NAMES>`` : provide file with names of the testing benchmarks (default: all).

    ``--tdp <TDP_VALUE>`` : give TDP value of the GPU device (default: 250).

    ``--o`` : create output file with the aggregated datasets (default: False).

    ``--v`` : turn on verbose mode (default: False).

* Example:
```bash
toolReadBenchs.py Outputs/Microbenchmarks/GTXTitanX/ gtxtitanx --benchs_file benchs_all.txt --test_data_path Outputs/RealBenchmarks/GTXTitanX/ --benchs_test_file benchs_real_best.txt --o
```

## 3. gpuPTXParser Tool

``gpuPTXParser`` is a command line tool that can be used for reading [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) files and extracting the number of occurrences of each different instructions per GPU kernel. The tool can also extract the sequence of instructions of the kernels in the source file.

If you use the ``gpuPTXParser`` tool in a publication, please cite [[1]](#4-references).

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

    ``outputOccurrences_per_kernel.csv`` file : with the count of occurrences of each instruction in the PTX ISA in each kernel from the parsed .ptx file. 1 row for each kernel. 1 column for each instruction.

    ``outputSequenceReadable_kernel_i.csv`` file for each kernel ``i`` in the parsed .ptx file.

    ``outputSequence_kernel_i.csv`` file for each kernel ``i`` in the parsed .ptx file. Values encoded.

## 4. REFERENCES

[1] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. [GPU Static Modeling Using PTX and Deep Structured Learning](https://ieeexplore.ieee.org/document/8890640). IEEE Access, Volume 7, November 2019.


## Dependencies

* [Python 3](https://www.continuum.io/downloads)
* [PyTorch 1.2.0+](http://pytorch.org/)

## Contact
If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt.

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
