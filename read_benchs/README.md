# toolReadBenchs

``toolReadBenchs`` is a command line tool that can be used for reading the measured values (execution times and power consumption) and organizing them in the format that can be used by the main ``gpuPTXModel`` tool.
The tool also creates .pdf files with the plots of the measured values across the different frequency levels.

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

## Contact
If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt.

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
