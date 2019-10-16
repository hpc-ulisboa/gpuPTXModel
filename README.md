# gpuPTXModel - GPU Power Model through Assembly Analysis using Deep Structured Learning

``gpuPTXModel`` is a command line tool that allows creating a GPU power consumption model, which can be used to predict the power consumption of applications based solely on the sequence of [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) (or [CUDA Assembly](https://docs.nvidia.com/cuda/cuda-binary-utilities/#overview)) instructions in the kernel code.

* Usage:
```
gpuPTXModel.py <path_to_data> [--device <cpu|gpu>] [--device_id <id>] [--model_name <multiNN|LSTM>] [--optimizer <SGD|Adagrad|Adam>] [--num_layers <num>] [--learning_rate <rate>] [--dropout_prob <prob>] [--num_epochs <epochs>] [--hidden_size <size>] [--embed_size <size>] [--tdp <tdp>] [--benchs_file <file>] [--use_folds] [--time]

Options:

  <path_to_data>: PATH to the directory with the microbenchmark profilling data (to be used in training).

  --device <cpu|gpu>: select the device where the training will execute (default: gpu)
  --device_id <id>: target a specific device (default: 0)
  --model_name <multiNN|LSTM>: to select which model to use in training (Multi-layer neural network or LSTM) (default: LSTM)
  --optimizer  <SGD|Adagrad|Adam>: to select which optimizer to use in training (default: SGD)   
  --num_layers <num>: for choosing the number of layers of the Model (default: 1)
  --learning_rate <rate>: for choosing the learning rate (default: 0.001)
  --dropout_prob <prob>: for choosing the dropout probability (default: 0)
  --num_epochs <epochs>: for choosing the number of epochs (default: 20)
  --hidden_size <size>: for choosing the size of the first hidden layer (or LSTM) (default: 100)
  --embed_size <size>: for choosing the size of embeddings (default: 5)
  --tdp <tdp>: for changing the value of the device TDP (default: 250)
  --benchs_file <file>: for selecting which microbenchmarks are used in the training (default: all benchmarks in given <path_to_data> directory)
  --use_folds: to toggle the usage of 5-fold cross-validation (default: not used)
  --time: to train an execution time model (default: power model)

Example:

  gpuPTXModel.py Microbenchmarks/Outputs/ --device gpu --model_name LSTM --num_layers 2 --learning_rate 0.001 --num_epochs 50
```

## gpuPTXParser

``gpuPTXParser`` is a command line tool that can be used for reading [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) files and extracting the number of occurrences of each different instructions per GPU kernel. The tool can also extract the sequence of instructions of the kernels in the source file.


* Usage:
```
gpuPTXParser.py <isa_file.txt> <application_to_profile.ptx>
```

* Example:
```
gpuPTXParser.py ptx_isa.txt Microbenchmarks/pure_DRAM/DRAM.ptx
```
## Contact
If you have any questions regarding gpuPTXModel please email joao.guerreiro@inesc-id.pt

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
