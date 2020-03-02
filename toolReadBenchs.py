import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from src import readFiles as rf
from os import listdir
import re


def getDirectoriesAtPath(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

# Function to read the execution time an application at a given path
def readExecTimeKernels(data_path, bench, gpu):
    files = listdir('%s/%s/' % (data_path, bench))
    files.sort()
    total_execution_all_kernels_aux = []

    #cycles accross the files at the given path
    for file in files:
        #only interested in nvprof output files
        if file.startswith("output_nvprof_"):
            #not interested in other metrics than the execution time
            if not file.startswith("output_nvprof_metrics_"):
                total_time = []
                time_per_kernel = {}
                num_calls = []
                avg_time_call = []
                max_time_call = []
                with open('%s/%s/%s' % (data_path, bench, file)) as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    for row_id, row in enumerate(spamreader):
                        if row_id > 4:  # header of the file
                            # only accounts kernels (in CUDA 10.0 only the kernels don't have the CUDA word in the output line)
                            if not 'CUDA' in row[7]:
                                total_time.append(float(row[2]))
                                time_per_kernel[row[7]] = float(row[2])
                                num_calls.append(float(row[3]))
                                avg_time_call.append(float(row[4]))
                                max_time_call.append(float(row[6]))
                num_calls = np.asarray(num_calls, dtype=np.int32)
                total_time = np.asarray(total_time, dtype=np.float32)
                avg_time_call = np.asarray(avg_time_call, dtype=np.float32)
                max_time_call = np.asarray(max_time_call, dtype=np.float32)

                total_execution_all_kernels_aux.append(np.sum(total_time))
    if len(total_execution_all_kernels_aux) == 0:
        print('Missing execution times for %s benchmark' % (bench))
        sys.exit()
    total_execution_all_kernels = np.asarray(total_execution_all_kernels_aux)

    return np.mean(total_execution_all_kernels), time_per_kernel


def checkEqual1(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

# Function to read the performance metrics files of an application at a given path
def readNvprofFile(data_path, bench, gpu, list_event_names, time_per_kernel):
    files = listdir('%s/%s/' % (data_path, bench))
    files.sort()

    list_events = {}
    total_execution_all_kernels_aux = []
    for file in files:
        if file.startswith("output_nvprof_metrics_"):
            with open('%s/%s/%s' % (data_path, bench, file)) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row_id, row in enumerate(spamreader):
                    if row_id > 5:
                        if 'overflow' in row[0]:
                            pass
                        else:
                            event = row[3]
                            if event in list_event_names:
                                kernel = row[1]
                                if event not in list_events:
                                    num_kernels = 1
                                    list_events[event] = {}
                                else:
                                    num_kernels += 1

                                if 'utilization' in event:
                                    list_events[event][kernel] = int(
                                        row[7].split()[1][1:-1])
                                else:
                                    # print(row[7])
                                    aux_value = float(
                                        (re.findall('\d+\.\d+', row[7]))[0])
                                    if 'GB/s' in row[7]:
                                        aux_value = aux_value * 1000000
                                    elif 'MB/s' in row[7]:
                                        aux_value = aux_value * 1000
                                    elif 'KB/s' in row[7]:
                                        aux_value = aux_value
                                    elif 'B/s' in row[7]:
                                        aux_value = aux_value / 1000.0

                                    list_events[event][kernel] = aux_value

    #confirms if all values were Measured
    if checkEqual1([list_events[key].keys() for key in list_events]) == False:
        print('Missing values (possible overflow) for benchmarks: %s' % bench)

    #currently the program ends if there are missing values
    for event in list_event_names:
        if event not in list_events.keys():
            print(list_events)
            print('Missing values for event \'%s\' (possible overflow) for benchmarks: %s' % (
                event, bench))
            sys.exit()

    aggregated_list = {}
    for event_id, event in enumerate(list_event_names):
        aggregated_list[event] = 0.0
        total_time = 0
        for kernel_name in time_per_kernel.keys():
            aggregated_list[event] += time_per_kernel[kernel_name] * \
                list_events[event][kernel_name]
            total_time += time_per_kernel[kernel_name]

        aggregated_list[event] = aggregated_list[event] / total_time

    return aggregated_list

# Function that cycles across all benchmarks at a given path
# and reads their profiling data
#
# OUTPUTS: lists_data dictionary with keys:
#              lists_data["time"]: list of the execution times of the diferent kernels
#              lists_data["pow"]: list of the power consumptions of the diferent kernels
#              lists_data["energy"]: list of the energies of the diferent kernels
def readListsData(benchs, clocks, benchs_data_path, gpu_name, idle_powers):
    mem_clocks = clocks['mem_clocks']
    core_clocks = clocks['core_clocks']
    num_mem_clocks = clocks['num_mem_clocks']
    num_core_clocks = clocks['num_core_clocks']

    max_num_core_clocks = np.max(num_core_clocks)

    num_benchs = len(benchs)
    list_pow = [None]*num_benchs
    list_time = [None]*num_benchs
    list_energy = [None]*num_benchs
    for bench_id, bench in enumerate(benchs):
        list_pow[bench_id] = np.zeros(
            (num_mem_clocks, max_num_core_clocks), dtype=np.float32)
        list_time[bench_id] = np.zeros(
            (num_mem_clocks, max_num_core_clocks), dtype=np.float32)
        list_energy[bench_id] = np.zeros(
            (num_mem_clocks, max_num_core_clocks), dtype=np.float32)

        for clock_mem_id, clock_mem in enumerate(mem_clocks):
            for clock_core_id, clock_core in enumerate(core_clocks[clock_mem_id]):
                avg_pow_bench = rf.readPowerBench(benchs_data_path, '%s/%d/%d' % (
                    bench, clock_mem, clock_core), idle_powers[clock_mem_id][clock_core_id])
                time_bench = rf.readExecTime(
                    benchs_data_path, '%s/%d/%d' % (bench, clock_mem, clock_core), gpu_name)
                list_pow[bench_id][clock_mem_id, clock_core_id] = avg_pow_bench
                list_time[bench_id][clock_mem_id,
                                    clock_core_id] = (time_bench/1000.0)
                list_energy[bench_id][clock_mem_id,
                                      clock_core_id] = avg_pow_bench*(time_bench/1000.0)

    lists_data = {'time': list_time, 'pow': list_pow, 'energy': list_energy}

    return lists_data

# Creates and completes the output file aggregating the dataset of all considered kernels
# Format:
#   line 1: benchmark_name_0
#   line 2: clock_mem_0, clock_core_0, time_sample, power_sample, energy_sample
#   line 3: clock_mem_0, clock_core_1, time_sample, power_sample, energy_sample
#   ...
def writeOutputFile(benchs_data_path, lists, benchs, clocks, gpu_name):
    mem_clocks = clocks['mem_clocks']
    core_clocks = clocks['core_clocks']

    list_time = lists['time']
    list_pow = lists['pow']
    list_energy = lists['energy']

    out = open("%s/aggregated_dataset_%s.csv" %
               (benchs_data_path, gpu_name), "w")  # output file
    for bench_id, bench in enumerate(benchs):
        out.write("%s\n" % (bench))
        for clock_mem_id, clock_mem in enumerate(mem_clocks):
            for clock_core_id, clock_core in enumerate(core_clocks[clock_mem_id]):
                out.write("%d,%d,%f,%f,%f\n" % (clock_mem, clock_core, list_time[bench_id][clock_mem_id, clock_core_id],
                                                list_pow[bench_id][clock_mem_id, clock_core_id], list_energy[bench_id][clock_mem_id, clock_core_id]))
    out.close()

# Function to print to output display the lists information
def printListsData(benchs, clocks, lists, energy_mode):
    mem_clocks = clocks['mem_clocks']
    core_clocks = clocks['core_clocks']

    list_time = lists['time']
    list_pow = lists['pow']
    list_energy = lists['energy']

    maxwidth = len(max(benchs, key=len))

    for bench_id, bench in enumerate(benchs):
        if bench_id == 0:
            header_line = '{message: >{width}}'.format(
                message='Clock Mem|', width=maxwidth+21)
            for clock_core_id, clock_core in enumerate(core_clocks[0]):
                if clock_core_id > 0:
                    header_line += '|'
                if energy_mode == True:
                    header_line += '{clock: >{width}}'.format(
                        clock='%d MHz' % clock_core, width=11)
                else:
                    header_line += '{clock: >{width}}'.format(
                        clock='%d MHz' % clock_core, width=13)
            print(header_line)
        bench_line = '{message: >{width}}: '.format(
            message=bench, width=maxwidth+2)
        for clock_mem_id, clock_mem in enumerate(mem_clocks):
            if clock_mem_id > 0:
                bench_line += '\n{message: >{width}}'.format(
                    message="%s MHz| " % clock_mem, width=maxwidth+22)
            else:
                bench_line += '%4d MHz| ' % clock_mem

            for clock_core_id, clock_core in enumerate(core_clocks[clock_mem_id]):
                if clock_core_id > 0:
                    bench_line += '| '

                if energy_mode == True:
                    bench_line += '{energy:8.1f} J'.format(
                        energy=list_energy[bench_id][clock_mem_id, clock_core_id])
                else:
                    bench_line += '{time:6.1f},{power:5.1f}'.format(
                        time=list_time[bench_id][clock_mem_id, clock_core_id], power=list_pow[bench_id][clock_mem_id, clock_core_id])
        print(bench_line)

# Function to create 3 output plots with the time, power and energy over different frequencies
#   (core and memory) across all the considered benchmarks
#   vertical axis represents the considered metric (time, power or energy depending on the plot)
#   horizontal axis displays the core frequency values, and different subplots correspond to
#   different memory frequencies. Each line in a subplot corresponds to a different benchmark.
def plotValues(name, lists, clocks, benchmarks, normalized_t, normalized_p, normalized_e, type):
    mem_clocks = clocks['mem_clocks']
    core_clocks = clocks['core_clocks']

    list_data_time = lists['time']
    list_data_pow = lists['pow']
    list_data_energy = lists['energy']

    fig_t = plt.figure(1)
    axes_t = fig_t.subplots(clocks['num_mem_clocks'], 1, sharex=True)
    fig_p = plt.figure(2)
    axes_p = fig_p.subplots(clocks['num_mem_clocks'], 1, sharex=True)
    fig_e = plt.figure(3)
    axes_e = fig_e.subplots(clocks['num_mem_clocks'], 1, sharex=True)
    count_bad = 0

    #plot the time, power and energy lines
    for bench_id, bench in enumerate(benchmarks):
        good_bench = True
        for clock_mem_id, clock_mem in enumerate(mem_clocks):
            if clocks['num_mem_clocks'] > 1:
                axis_t = axes_t[clock_mem_id]
                axis_p = axes_p[clock_mem_id]
                axis_e = axes_e[clock_mem_id]
            else:
                axis_t = axes_t
                axis_p = axes_p
                axis_e = axes_e

            #this cycle if the gathered samples display a consistent behaviour, i.e. if the time and power curves of an application are monotonic when the core frequency decreases
            for clock_core_id, clock_core in enumerate(core_clocks[clock_mem_id]):
                if clock_core_id + 1 < clocks['num_core_clocks'][clock_mem_id] and list_data_time[bench_id][clock_mem_id, clock_core_id] < list_data_time[bench_id][clock_mem_id, clock_core_id+1]:
                    good_bench = False
                    count_bad += 1
                    break
                if clock_core_id + 1 < clocks['num_core_clocks'][clock_mem_id] and list_data_pow[bench_id][clock_mem_id, clock_core_id] > list_data_pow[bench_id][clock_mem_id, clock_core_id+1]:
                    good_bench = False
                    count_bad += 1
                    break

            # type determines the benchmarks to be plotted (type=0 plots all benchmarks; type=1 plots only good benchmarks; and type=2 plots only bad benchmarks)
            if (type == 0) or (type == 1 and good_bench == True) or (type == 2 and good_bench == False):
                if normalized_t == True:
                    axis_t.plot(core_clocks[clock_mem_id], list_data_time[bench_id][clock_mem_id,
                                                                                    :]/list_data_time[bench_id][-1, -1], linestyle='--', label=bench)
                else:
                    axis_t.plot(core_clocks[clock_mem_id], list_data_time[bench_id]
                                [clock_mem_id, :], linestyle='--', label=bench)
                if normalized_p == True:
                    axis_p.plot(core_clocks[clock_mem_id], list_data_pow[bench_id][clock_mem_id,
                                                                                   :]/list_data_pow[bench_id][-1, -1], linestyle='--', label=bench)
                else:
                    axis_p.plot(core_clocks[clock_mem_id], list_data_pow[bench_id]
                                [clock_mem_id, :], linestyle='--', label=bench)
                if normalized_e == True:
                    axis_e.plot(core_clocks[clock_mem_id], list_data_energy[bench_id][clock_mem_id,
                                                                                      :]/list_data_energy[bench_id][-1, -1], linestyle='--', label=bench)
                else:
                    axis_e.plot(core_clocks[clock_mem_id], list_data_energy[bench_id]
                                [clock_mem_id, :], linestyle='--', label=bench)

    for clock_mem_id, clock_mem in enumerate(mem_clocks):
        if clocks['num_mem_clocks'] > 1:
            ax2_aux = axes_e[clock_mem_id].twinx()
        else:
            ax2_aux = axes_e.twinx()

    if type == 2:
        name = 'bad_' + name
    elif type == 1:
        name = 'good_' + name
    else:
        name = 'all_' + name

    if clocks['num_mem_clocks'] > 1:
        axes_t[0].set_title('time_%s' % name)
        axes_p[0].set_title('power_%s' % name)
        axes_e[0].set_title('energy_%s' % name)
    else:
        axes_t.set_title('time_%s' % name)
        axes_p.set_title('power_%s' % name)
        axes_e.set_title('energy_%s' % name)

    print('bad benchmarks %s: %d' % (name, count_bad))

    fig_t.savefig('time_%s.pdf' % (name))
    fig_p.savefig('pow_%s.pdf' % (name))
    fig_e.savefig('energy_%s.pdf' % (name))

    plt.close("all")


def main():
    """Main function."""
    import argparse
    import sys
    import sys
    from src import globalStuff as gls
    from src.globalStuff import printing, output_dir_train, list_event_names
    from src.readFiles import readIdlePowers, getBenchmarksAvailable
    use_test = False

    gls.init()

    parser = argparse.ArgumentParser()

    # path to the microbenchmarks dataset
    parser.add_argument('benchs_data_path', type=str)
    parser.add_argument('gpu', type=str)  # gpu name
    # path to the standard benchmarks dataset
    parser.add_argument('--test_data_path', type=str, default='')
    # file with the microbenchmark names
    parser.add_argument('--benchs_file', type=str, default='all')
    # file with the standard benchmarks names
    parser.add_argument('--benchs_test_file', type=str, default='all')
    parser.add_argument('--tdp', type=int, default=250)  # TDP
    parser.add_argument('--v', action='store_const',
                        const=True, default=False)  # verbose mode
    # calculates energy values from time and power samples
    parser.add_argument('--e', action='store_const', const=False, default=True)
    # plot/print only bad benchmarks (default is ALL benchmarks)
    parser.add_argument('--bad', action='store_const',
                        const=True, default=False)
    # plot/print only good benchmarks (default is ALL benchmarks)
    parser.add_argument('--good', action='store_const',
                        const=True, default=False)
    # create output file (file aggregated_dataset_<gpu_name>.csv)
    parser.add_argument('--o', action='store_const', const=True, default=False)
    # also reads performance counters samples
    parser.add_argument('--pc', action='store_const',
                        const=True, default=False)

    args = vars(parser.parse_args())
    print(args)

    benchs_data_path = args['benchs_data_path']
    gpu_name = args['gpu']
    test_data_path = args['test_data_path']
    benchs_file = args['benchs_file']
    benchs_test_file = args['benchs_test_file']
    tdp = args['tdp']
    verbose = args['v']
    energy_mode = args['e']
    bad_values_mode = args['bad']
    good_values_mode = args['good']
    create_output_file = args['o']

    ubenchmarks = getBenchmarksAvailable(
        gls.benchmarks_info_folder, benchs_file, benchs_data_path)
    ubenchmarks.sort()
    num_ubenchmarks = len(ubenchmarks)
    print("\n=============================Reading Data=============================\n")
    print('Number of microbenchmarks: %d' % (num_ubenchmarks))

    print('Benchs file: %s' % benchs_file)

    clocks = rf.getClocksGPU(gpu_name)

    idle_powers = readIdlePowers(clocks, gpu_name)
    lists_data_ubench = readListsData(
        ubenchmarks, clocks, benchs_data_path, gpu_name, idle_powers)
    if test_data_path != '':
        use_test = True
        test_benchmarks = getBenchmarksAvailable(
            gls.benchmarks_info_folder, benchs_test_file, test_data_path)
        test_benchmarks.sort()
        num_test_benchmarks = len(test_benchmarks)
        print('\nNumber of testing benchmarks: %d' % (num_test_benchmarks))
        print('Test Benchs file: %s' % benchs_test_file)

        lists_data_testbench = readListsData(
            test_benchmarks, clocks, test_data_path, gpu_name, idle_powers)

    #print read values
    if verbose == True:
        #if core clocks for all memory levels are the same
        if clocks['core_clocks'].count(clocks['core_clocks'][0]) == len(clocks['core_clocks']):
            print(
                "\n=============================Microbenchmarks=============================\n")
            printListsData(ubenchmarks, clocks, lists_data_ubench, energy_mode)
            if test_data_path != '':
                print(
                    "\n\n=============================Test Benchmarks=============================\n")
                printListsData(test_benchmarks, clocks,
                               lists_data_testbench, energy_mode)
        else:
            print("Cannot print list of values")

    print("\n=============================The End=============================\n")

    if bad_values_mode == True:
        type = 2
    elif good_values_mode:
        type = 1
    else:
        type = 0

    #choose if the output plots have the values normalized or not
    normalized_t = True
    normalized_p = False
    normalized_e = True

    plotValues('micro_%s' % gpu_name, lists_data_ubench, clocks,
               ubenchmarks, normalized_t, normalized_p, normalized_e, type)
    plotValues('test_%s' % gpu_name, lists_data_testbench, clocks,
               test_benchmarks, normalized_t, normalized_p, normalized_e, type)

    if create_output_file == True:
        writeOutputFile(benchs_data_path, lists_data_ubench,
                        ubenchmarks, clocks, gpu_name)
        writeOutputFile(test_data_path, lists_data_testbench,
                        test_benchmarks, clocks, gpu_name)


if __name__ == "__main__":
    main()
