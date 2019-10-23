# gpuPTXModel - GPU Power Model through Assembly Analysis using Deep Structured Learning

<p align="center"><img width="100%" src="png/model.png" /></p>

## Directory structure

* ``assembly_info`` - files with Assembly ISA (list of possible instructions) and available modifiers;
* ``benchmarks`` - information on considered benchmarks, including source code of the microbenchmarks;
* ``datasets`` - datasets used to train (and validate) the models;
* ``gpu_info`` - files that characterize the architecture to be considered in the trained models (number of freq. domains, number of levels in each domain, etc.);
* ``model_configs`` - configurations of the models;
* ``ptx_parser`` - command line tool that can read PTX files and output the required information to be used in the proposed models;
* ``src`` - gpuPTXModel source auxiliary files.


## 1. gpuPTXModel Tool

``gpuPTXModel`` is a command line tool that allows creating a GPU power consumption model, which can be used to predict the power consumption of applications based solely on the sequence of [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) instructions in the kernel code.
The tool receives as argument the path to the microbenchmark (and optionally the testing dataset), which need to have been properly aggregated using the ``toolReadBenchs`` tool.

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
The tool also creates pdf files with the plots of the measured values across the different frequency levels.

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

``gpuPTXParser`` is a command line [tool](https://github.com/hpc-ulisboa/gpuPTXModel/tree/master/ptx_parser) that can be used for reading [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) files and extracting the number of occurrences of each different instructions per GPU kernel. The tool can also extract the sequence of instructions of the kernels in the source file.

## Contact
If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt.

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
