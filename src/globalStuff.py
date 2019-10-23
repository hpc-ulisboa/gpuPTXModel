import os
import sys
import torch
import time
import math

from os import listdir

# POW_THRESHOLD_GPU_ACTIVE = 30 #above this value (in Watts) the GPU is considered active
POW_THRESHOLD_GPU_ACTIVE = 7 #threshold above the idle power to consider the GPU is active
SAVE_STATE = True #decides if saves intermediate networks
SAVE_EVERY_TH_EPOCH = 50


output_dir_train = 'output_training_networks'
OutputFile= "output_log.txt"

gpu_info_folder = 'gpu_info'
assembly_info_folder = 'assembly_info'
benchmarks_info_folder = 'benchmarks'

isa_file = '%s/ptx_isa.txt' %assembly_info_folder
inst_types_file = '%s/ptx_instruction_types.txt' %assembly_info_folder
state_spaces_file = '%s/ptx_state_spaces.txt' %assembly_info_folder

num_iters_pow = 2
max_operands = 5
buffer_max_size = 9
dependencies_types = 3

dataset_dict_headers = ['names', 'inst_seq', 'time_dvfs', 'pow_dvfs', 'energy_dvfs', 'time_default', 'pow_default', 'energy_default']

percentage_training = .9
threshold_consecutive_epochs = 0.5
max_consecutive_epochs_no_improvement = 4
IND_PAD = 2

list_event_names = ['l2_read_throughput', 'l2_write_throughput', 'shared_load_throughput', 'shared_store_throughput', 'dram_read_throughput', 'dram_write_throughput', 'achieved_occupancy', 'cf_fu_utilization']
metrics_to_use = ['l2_read_throughput', 'l2_write_throughput', 'shared_load_throughput', 'shared_store_throughput', 'dram_read_throughput', 'dram_write_throughput']
metrics_to_use = ['l2_read_throughput', 'l2_write_throughput']
max_values_metrics = [5e08, 5e08, 1e09, 1e09, 3e08, 2e08, 1.0, 10.0]

possible_outputs = ['time_dvfs', 'pow_dvfs', 'energy_dvfs']

def init():
    global ISA_size
    ISA_size = 0
    global inst_types_size
    inst_types_size = 0
    global state_spaces_size
    state_spaces_size = 0
    init_plots()

def init_plots():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
    plt.rcParams.update({'font.size': 4})

    global colors
    colors = {'Training': 'teal', 'Validation': 'coral', 'Testing': 'indigo'}
    global markers
    markers = {'Training': '^', 'Validation': 'x', 'Testing': 'D'}
    global marker_sizes
    marker_sizes = {'Training': 2, 'Validation': 2, 'Testing': 2}


def printing(text, no_output=False):
    if no_output == False:
        print(text)
    OF.write('%s\n' %text)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))

def createTestOutputFolder(output_dir, script_dir, test_number, device_id):
    try:
        os.makedirs(os.path.join(script_dir, output_dir))
    except OSError:
        pass # already exists

    num_tests_completed = len(listdir(output_dir))

    if test_number == -1:
        test_number = num_tests_completed

    test_output_dir = os.path.join(script_dir, output_dir, 'dev%d_test_%d' %(device_id, test_number))
    try:
        os.makedirs(test_output_dir)
    except OSError:
        print('Test has already been completed')
        sys.exit()
        pass # already exists

    createOutputLogFile(test_output_dir, OutputFile)

    return test_output_dir

def createOutputLogFile(dir, filename):
    global OF
    OF = open('%s/%s' %(dir, filename), 'w')

def closeOutputLogFile():
    OF.close()

def reorderData(data, pc):
    indexes = list(range(len(data['names'])))
    seq_len_list = [len(i) for i in data['inst_seq']] #gets the number of instructions in each program
    indexes.sort(key=seq_len_list.__getitem__) #orders them from shorter to larger

    if pc == True:
        dataset_headers = dataset_dict_headers + metrics_to_use
    else:
        dataset_headers = dataset_dict_headers

    reordered_data = {}
    for str_aux in dataset_headers:
        reordered_data[str_aux] = list(map(data[str_aux].__getitem__, indexes)) #reorders all the fields

    return reordered_data

def arrangeDataset(dataset, indexes, pc):
    new_dataset = {}
    if pc == True:
        dataset_headers = dataset_dict_headers + metrics_to_use
    else:
        dataset_headers = dataset_dict_headers

    for str_aux in dataset_headers:
        new_dataset[str_aux] = [dataset[str_aux][i] for i in indexes]

    return new_dataset

def initPytorch(device_arg, device_id):
    if (device_arg == 'gpu'):
        if torch.cuda.is_available():
            device = torch.device("cuda:%d" %(device_id))
            printing(torch.cuda.get_device_name(device_id))
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device
