def main():
    """Main function."""
    import argparse
    import sys
    import torch

    import numpy as np
    import os
    from os import listdir
    from src import readFiles as rf
    from src import functionsPyTorch as pytor
    from src import globalStuff as gls
    from src.globalStuff import printing, output_dir_train, isa_file, inst_types_file, state_spaces_file, initPytorch, arrangeDataset, possible_outputs, closeOutputLogFile

    use_test = False

    gls.init()

    parser = argparse.ArgumentParser()

    parser.add_argument('benchs_data_path', type=str)
    parser.add_argument('gpu_name', type=str)
    parser.add_argument('--test_data_path', type=str, default='')
    parser.add_argument('--benchs_file', type=str, default = 'all')
    parser.add_argument('--benchs_test_file', type=str, default = 'all')
    parser.add_argument('--tdp', type=int, default=250)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--test_number', type=int, default=-1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--fast', action='store_const', const=True, default=False)
    parser.add_argument('--v', action='store_const', const=True, default=False)
    parser.add_argument('--no_pow_dvfs', action='store_const', const=True, default=False)
    parser.add_argument('--no_time_dvfs', action='store_const', const=True, default=False)
    parser.add_argument('--no_energy_dvfs', action='store_const', const=True, default=False)
    parser.add_argument('--never_stop', action='store_const', const=True, default=False)
    parser.add_argument('--no_output', action='store_const', const=True, default=False)
    parser.add_argument('--pc', action='store_const', const=True, default=False)
    parser.add_argument('--encoder_file', type=str, default='')
    parser.add_argument('--time_dvfs_file', type=str, default='')
    parser.add_argument('--pow_dvfs_file', type=str, default='')
    parser.add_argument('--energy_dvfs_file', type=str, default='')

    args = vars(parser.parse_args())
    print(args)

    benchs_data_path = args['benchs_data_path']
    gpu_name = args['gpu_name']
    test_data_path = args['test_data_path']
    benchs_file = args['benchs_file']
    benchs_test_file = args['benchs_test_file']
    tdp = args['tdp']
    device_arg = args['device']
    device_id = args['device_id']
    num_epochs = args['num_epochs']
    fast = args['fast']
    verbose = args['v']
    never_stop = args['never_stop']
    no_output = args['no_output']
    test_number = args['test_number']
    performance_counters = args['pc']
    encoder_config_file = args['encoder_file']
    timedvfs_config_file = args['time_dvfs_file']
    powdvfs_config_file = args['pow_dvfs_file']
    energydvfs_config_file = args['energy_dvfs_file']

    outputs_to_model = {'time_dvfs': args['no_time_dvfs'], 'pow_dvfs': args['no_pow_dvfs'], 'energy_dvfs': args['no_energy_dvfs']}

    ISA = rf.readISA(isa_file)
    gls.ISA_size = len(ISA)
    state_spaces = rf.readISA(state_spaces_file)
    gls.state_spaces_size = len(state_spaces)
    inst_types = rf.readISA(inst_types_file)
    gls.inst_types_size = len(inst_types)

    orig_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = gls.createTestOutputFolder(output_dir_train, orig_dir, test_number, device_id)

    device = initPytorch(device_arg, device_id)
    printing(device, no_output)
    np.random.seed(40)

    # printing('Model type: %s' %model_name, no_output)
    printing('Data from GPU: %s' %gpu_name, no_output)

    dataset_ubench, clocks = rf.readDataSet(benchs_data_path, gpu_name, tdp, performance_counters)
    ubenchmarks = dataset_ubench['names']
    num_ubenchmarks = len(ubenchmarks)
    printing('Number of microbenchmarks: %d' %(num_ubenchmarks), no_output)
    printing('Benchs file: %s' %benchs_file, no_output)

    #if testing set is provided
    if test_data_path != '':
        use_test = True

        dataset_test, _ = rf.readDataSet(test_data_path, gpu_name, tdp, performance_counters)
        test_benchmarks = dataset_test['names']
        num_test_benchmarks = len(test_benchmarks)
        printing('Number of testing benchmarks: %d' %(num_test_benchmarks), no_output)
        printing('Test Benchs file: %s' %benchs_test_file, no_output)

    vocab_size = gls.ISA_size*(gls.state_spaces_size+1)*(gls.inst_types_size+1)*(gls.max_operands+1)*(gls.buffer_max_size+1)*(gls.dependencies_types+1) + 3
    if verbose == True:
        print('Vocab size: %d' %vocab_size)

    index_limit_training = int(gls.percentage_training * num_ubenchmarks)

    random_ordering = np.arange(num_ubenchmarks)
    np.random.shuffle(random_ordering)

    index_train = random_ordering[0:index_limit_training]
    index_train.sort()
    index_val = random_ordering[index_limit_training:]
    index_val.sort()

    data_train = arrangeDataset(dataset_ubench, index_train, performance_counters)
    data_val = arrangeDataset(dataset_ubench, index_val, performance_counters)

    if encoder_config_file != '':
        config = rf.readISA('model_configs/encoder/%s.txt' %(encoder_config_file))
        encoder_params = {'embed_size': int(config[0]), 'learning_rate': float(config[1]), 'dropout_prob': float(config[2]), 'optimizer_name': config[3], 'num_layers': int(config[4]), 'hidden_size': int(config[5]), 'batch_size': int(config[6])}
    else:
        encoder_params = {'embed_size': embed_size, 'learning_rate': learning_rate_encoder, 'dropout_prob': dropout_prob_encoder, 'optimizer_name': optimizer_encoder, 'num_layers': num_layers_encoder, 'hidden_size': hidden_size_encoder, 'batch_size': batch_size}
    encoder_params['vocab_size'] = vocab_size


    # PREPARE THE PARAMETERS BEFORE MODEL TRAINING
    nn_params = {}
    config_files = [timedvfs_config_file,  powdvfs_config_file, energydvfs_config_file]
    for model_name, model_config_file in zip(possible_outputs, config_files):
        if outputs_to_model[model_name] == False:
            if model_config_file != '':
                config = rf.readISA('model_configs/%s/%s.txt' %(model_name, model_config_file))
                nn_params[model_name] = {'learning_rate': float(config[0]), 'dropout_prob': float(config[1]), 'optimizer_name': config[2], 'num_layers': int(config[3])}

                hidden_sizes_list = []
                for hidden_size_aux in config[4:]:
                    hidden_sizes_list.append(int(hidden_size_aux))

                nn_params[model_name]['hidden_sizes'] = np.asarray(hidden_sizes_list)
                if len(nn_params[model_name]['hidden_sizes']) != nn_params[model_name]['num_layers']:
                    print('ERROR: Hidden sizes dont match with number of hidden layers in file: \'model_configs/encoder/%s' %(encoder_config_file))
                    sys.exit()
            else:
                nn_params[model_name] = {'learning_rate': learning_rate, 'dropout_prob': dropout_prob, 'optimizer_name': optimizer, 'num_layers': 2, 'hidden_sizes': [hidden_size, hidden_size_2]}

    model_params = {'model_name': model_name, 'max_epochs': num_epochs, 'encoder_params': encoder_params, 'nn_params': nn_params, 'outputs': outputs_to_model, 'never_stop': never_stop, 'no_output': no_output}


    # TRAIN THE MODELS
    if use_test == True:
        data_test = arrangeDataset(dataset_test, np.arange(len(dataset_test['names'])), performance_counters)
        trainingTime, trainedModels, results_train, results_val, results_test = pytor.trainPytorchModel(device, clocks, verbose, fast, performance_counters, model_params, test_output_dir, data_train, data_val, data_test)
    else:
        trainingTime, trainedModels, results_train, results_val, _  = pytor.trainPytorchModel(device, clocks, verbose, fast, performance_counters, model_params, test_output_dir, data_train, data_val)
    if verbose == True:
        print(test_output_dir)


    if fast == 0:
        predicted_values = {}
        measured_values = {}
        errors_values = {}

        #save the predictions to an output .csv file
        for model_type in trainedModels['output_types']:
            predicted_values[model_type] = {'Training': results_train['last_epoch_predictions'][model_type], 'Validation': results_val['last_epoch_predictions'][model_type]}
            measured_values[model_type] = {'Training': data_train[model_type], 'Validation': data_val[model_type]}
            errors_values[model_type] = {'Training': results_train['abs_error_per_epoch'][-1][model_type], 'Validation': results_val['abs_error_per_epoch'][-1][model_type]}

            np.savetxt('%s/last_epoch_prediction_train_%s_%s_%s.csv' %(test_output_dir, model_type, model_name, benchs_file[:-4]), predicted_values[model_type]['Training'], delimiter=",")
            np.savetxt('%s/last_epoch_prediction_val_%s_%s_%s.csv' %(test_output_dir, model_type, model_name, benchs_file[:-4]), predicted_values[model_type]['Validation'], delimiter=",")

        #if using the testing benchmarks to validate model
        if use_test == True:
            for model_type in trainedModels['output_types']:
                predicted_values[model_type]['Testing'] =  results_test['last_epoch_predictions'][model_type]
                measured_values[model_type]['Testing'] = data_test[model_type]
                errors_values[model_type]['Testing'] = results_test['abs_error_per_epoch'][-1][model_type]
                np.savetxt('%s/last_epoch_prediction_test_%s_%s_%s.csv' %(test_output_dir, model_type, model_name, benchs_file[:-4]), predicted_values[model_type]['Testing'], delimiter=",")

    #save model to an output file
    torch.save(trainedModels['encoder'].state_dict(), '%s/%s_%s' %(test_output_dir, 'encoder', benchs_file))
    for model in trainedModels['output_types']:
        torch.save(trainedModels[model].state_dict(), '%s/%s_%s' %(test_output_dir, model, benchs_file))
#
    #print last epoch results
    LEF = open('%s/last_epoch.txt' %(test_output_dir), 'w')
    LEF.write("num_epochs,%d\n" %len(results_train['abs_error_per_epoch']))
    if use_test == True:
        datasets_names = ['Training', 'Validation', 'Testing']
    else:
        datasets_names = ['Training', 'Validation']

    for model_type in trainedModels['output_types']:
        for dataset in datasets_names:
            LEF.write('%s,%s,%.4f\n' %(dataset, model_type, errors_values[model_type][dataset]))
    LEF.close()

    closeOutputLogFile()
if __name__ == "__main__":
    main()
