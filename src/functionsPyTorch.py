import numpy as np
import sys
import time
import torch
from torch.autograd import Variable

import src.modelsPytorch as mopy
from src.globalStuff import printing, threshold_consecutive_epochs, max_consecutive_epochs_no_improvement, reorderData, IND_PAD, timeSince, asMinutes, list_event_names, metrics_to_use, possible_outputs, SAVE_EVERY_TH_EPOCH, SAVE_STATE

def saveIntermediateState(test_output_dir, epoch, trainedModels, model_params, pred_train, pred_val, pred_test, data_train, data_val, data_test):
    import os

    try:
        os.makedirs(os.path.join(test_output_dir, 'epoch_number_%d' %epoch))
    except OSError:
        pass # already exists

    #save pytorch models
    torch.save(trainedModels['encoder'].state_dict(), '%s/epoch_number_%d/%s_optenc%s_lrenc%s_layrenc%s_hiddenenc%s_dropoutenc%s_epochnum%s_embed%s_batch%s' %(test_output_dir, epoch, 'encoder', model_params['encoder_params']['optimizer_name'], model_params['encoder_params']['learning_rate'], model_params['encoder_params']['num_layers'], model_params['encoder_params']['hidden_size'], model_params['encoder_params']['dropout_prob'], epoch, model_params['encoder_params']['embed_size'], model_params['encoder_params']['batch_size']))
    for model_type in trainedModels['output_types']:
        hidden_sizes_str = '_'.join(map(str, model_params['nn_params'][model_type]['hidden_sizes']))
        torch.save(trainedModels[model_type].state_dict(), '%s/epoch_number_%d/%s_opt%s_epochnum%s_lr%s_layr%s_hidden%s_dropout%s' %(test_output_dir, epoch, model_type, model_params['nn_params'][model_type]['optimizer_name'], epoch, model_params['nn_params'][model_type]['learning_rate'], model_params['nn_params'][model_type]['num_layers'], hidden_sizes_str, model_params['nn_params'][model_type]['dropout_prob']))

    #save most recent epoch predictions
    predicted_values = {}
    measured_values = {}
    errors_values = {}
    for model_type in trainedModels['output_types']:
        predicted_values[model_type] = {'Training': pred_train['last_epoch_predictions'][model_type], 'Validation': pred_val['last_epoch_predictions'][model_type]}
        measured_values[model_type] = {'Training': data_train[model_type], 'Validation': data_val[model_type]}
        errors_values[model_type] = {'Training': pred_train['abs_error_per_epoch'][-1][model_type], 'Validation': pred_val['abs_error_per_epoch'][-1][model_type]}
        np.savetxt('%s/epoch_number_%d/last_epoch_prediction_train_%s.csv' %(test_output_dir, epoch, model_type), predicted_values[model_type]['Training'], delimiter=",")
        np.savetxt('%s/epoch_number_%d/last_epoch_prediction_val_%s.csv' %(test_output_dir, epoch, model_type), predicted_values[model_type]['Validation'], delimiter=",")
        if data_test != []:
            predicted_values[model_type]['Testing'] = pred_test['last_epoch_predictions'][model_type]
            measured_values[model_type]['Testing'] = data_test[model_type]
            errors_values[model_type]['Testing'] = pred_test['abs_error_per_epoch'][-1][model_type]
            np.savetxt('%s/epoch_number_%d/last_epoch_prediction_test_%s.csv' %(test_output_dir, epoch, model_type), predicted_values[model_type]['Testing'], delimiter=",")

def padSequence(sequence, max_length):
    sequence += [IND_PAD for i in range(max_length - len(sequence))]
    return sequence

def tensorFromWord(program, max_size_batch, device):
    if not len(program) == max_size_batch:
        program = padSequence(program, max_size_batch)
    return torch.tensor(program, dtype=torch.long).view(-1, 1).to(device)

def tensorFromBatch(program_insts, batch_index, batch_size, device):
    tensors_batch=[]
    len_list = [len(i) for i in program_insts[batch_index:batch_index+batch_size]]
    max_size_batch = np.max(len_list)
    for program_id in range(batch_size):
        tensors_batch.append(tensorFromWord(program_insts[batch_index+program_id], max_size_batch, device))
    tensors_batch = torch.stack(tensors_batch)
    tensors_batch = torch.squeeze(tensors_batch)
    return tensors_batch

def evaluateBatch(device, clocks, models, model_name, input, target_variable, batch_index, batch_size, loss_function, pc, metrics_batch):
    encoder = models['encoder']
    output_model = models[model_name]

    loss = 0
    pred_output = torch.empty([batch_size, clocks['num_mem_clocks'], np.max(clocks['num_core_clocks'])] , dtype=torch.float).to(device)

    with torch.no_grad():
        input_tensor = tensorFromBatch(input, batch_index, batch_size, device).requires_grad_(requires_grad=False).detach()
        if (batch_size == 1):
            input_variable = Variable(input_tensor)
        else:
            input_variable = Variable(input_tensor.transpose(0, 1))

        seq_length = input_variable.size()[0]
        target_length = target_variable.view(-1).size()[0]

        encoder.hidden = encoder.initHidden(batch_size)
        for ei in range(seq_length):
            lstm_output = encoder(input_variable[ei], batch_size)

        lstm_output = lstm_output.view(batch_size, -1)

        for bi in range(batch_size):
            #if using performance_counters
            if pc == True:
                metrics_bi = []
                for metric_id, metric in enumerate(metrics_to_use):
                    metrics_bi.append(metrics_batch[metric][bi])
                metrics_tensor = torch.tensor(metrics_bi, dtype=torch.float).to(device)

            for clock_mem_id, clock_mem in enumerate(clocks['mem_clocks']):
                for clock_core_id, clock_core in enumerate(clocks['core_clocks'][clock_mem_id]):
                    clock_tensor = torch.tensor([clock_mem/clocks['highest_mem_clock'], clock_core/clocks['highest_core_clock']], dtype=torch.float).to(device)
                    newfeatures = torch.cat((lstm_output[bi,:], clock_tensor), 0).to(device) #concatenate V-F level to the feature vector
                    if pc == True:
                        newfeatures = torch.cat((newfeatures, metrics_tensor), 0).to(device)
                    pred_output[bi][clock_mem_id, clock_core_id] = output_model(newfeatures)
                    loss += loss_function(pred_output[bi][clock_mem_id, clock_core_id], target_variable[bi][clock_mem_id,clock_core_id])

    return pred_output, loss.item() / target_length

def processBatch(device, clocks, verbose, models, model_name, batch_index, batch_size, args_batch, X, y, names, pc, metrics_batch):
    loss_function = torch.nn.SmoothL1Loss()
    errors = args_batch[0]
    predicted_values = args_batch[1]
    total_loss = args_batch[2]

    real_target =  torch.tensor(y[batch_index:batch_index+batch_size], dtype=torch.float).to(device).requires_grad_(requires_grad=False).detach()
    target_variable = real_target

    pred_target, loss_batch = evaluateBatch(device, clocks, models, model_name, X, target_variable, batch_index, batch_size, loss_function, pc, metrics_batch)

    total_loss += loss_batch

    for bench_i in range(len(real_target)):
        for clock_mem_id, clock_mem in enumerate(clocks['mem_clocks']):
            for clock_core_id, clock_core in enumerate(clocks['core_clocks'][clock_mem_id]):
                real = real_target[bench_i][clock_mem_id, clock_core_id].item()
                pred = pred_target[bench_i][clock_mem_id, clock_core_id].item()
                predicted_values.append(pred)
                if verbose == True:
                    if names == []:
                            print('\t\tValidation --- (%d, %d) Predicted factor (%s): %2.3f, Real factor: %2.3f (difference %5.2f%%)' %(clock_mem, clock_core, model_name, pred, real, 100.0*np.abs(real-pred)/real))
                    else:
                        maxwidth=len(max(names,key=len))
                        str_aux = '\t{message: >{width}}'.format(message=names[batch_index+bench_i], width=maxwidth+3)
                        str_aux += ': %5.2f%% --- (%d, %d) Predicted factor (%s): %2.3f, Real factor: %2.3f' %(100.0*np.abs(real-pred)/real, clock_mem, clock_core, model_name, pred, real)
                        print(str_aux)
                errors.append(np.absolute(pred-real)/real*100.0)

    output_batch = [errors, predicted_values, total_loss]
    return output_batch

def evaluate_lstm(device, clocks, verbose, models, X, y, model_name, pc, data, names=[]):
    total_programs = len(X)
    errors = []
    predicted_values = []
    total_loss=0.0

    batch_size = 4
    batch_index = 0
    while batch_index + batch_size <= total_programs:
        next_index = batch_index + batch_size
        args_batch = [errors, predicted_values, total_loss]
        metrics_batch={}
        if pc == True:
            for metric_id, metric in enumerate(metrics_to_use):
                metrics_batch[metric] = data[metric][batch_index:batch_index+batch_size]
        output_batch = processBatch(device, clocks, verbose, models, model_name, batch_index, batch_size, args_batch, X, y, names, pc, metrics_batch)
        errors = output_batch[0]
        predicted_values = output_batch[1]
        total_loss = output_batch[2]

        batch_index = next_index

    if batch_index < total_programs:
        batch_size_aux = total_programs - batch_index
        args_batch = [errors, predicted_values, total_loss]
        metrics_batch={}
        if pc == True:
            for metric_id, metric in enumerate(metrics_to_use):
                metrics_batch[metric] = data[metric][batch_index:batch_index+batch_size]
        output_batch = processBatch(device, clocks, verbose, models, model_name, batch_index, batch_size_aux, args_batch, X, y, names, pc, metrics_batch)
        errors = output_batch[0]
        predicted_values = output_batch[1]
        total_loss = output_batch[2]

    return total_loss, np.mean(errors), errors, predicted_values

def trainBatch(device, clocks, model_name, input_tensor, target_tensor, models, output_name, optimizers, criterion, batch_size, pc, metrics_batch):
    encoder = models['encoder']
    optimizer_encoder = optimizers['encoder']

    output_model = models[output_name]
    optimizer_output = optimizers[output_name]

    optimizer_encoder.zero_grad()
    optimizer_output.zero_grad()

    if (batch_size == 1):
        input_variable = Variable(input_tensor)
    else:
        input_variable = Variable(input_tensor.transpose(0, 1))

    target_variable = Variable(target_tensor)

    seq_length = input_variable.size()[0]
    target_length = target_variable.view(-1).size()[0]

    loss = 0

    encoder.hidden = encoder.initHidden(batch_size)
    for ei in range(seq_length):
        features = encoder(input_variable[ei], batch_size)

    features = features.view(batch_size, -1)

    for bi in range(batch_size):
        #if using performance_counters
        if pc == True:
            metrics_bi = []
            for metric_id, metric in enumerate(metrics_to_use):
                metrics_bi.append(metrics_batch[metric][bi])
            metrics_tensor = torch.tensor(metrics_bi, dtype=torch.float).to(device)

        for clock_mem_id, clock_mem in enumerate(clocks['mem_clocks']):
            for clock_core_id, clock_core in enumerate(clocks['core_clocks'][clock_mem_id]):
                clock_tensor = torch.tensor([clock_mem/clocks['highest_mem_clock'], clock_core/clocks['highest_core_clock']], dtype=torch.float).to(device)
                newfeatures = torch.cat((features[bi,:], clock_tensor), 0).to(device) #concatenate V-F level to the feature vector
                if pc == True:
                    newfeatures = torch.cat((newfeatures, metrics_tensor), 0).to(device)
                output = output_model(newfeatures)
                loss += criterion(output[0], target_variable[bi][clock_mem_id,clock_core_id])

    loss.backward()
    optimizer_encoder.step()
    optimizer_output.step()
    return loss.item() / target_length

def evaluate(device, clocks, verbose, no_output, epoch, dataset_name, models, data, pc):
    X = data['inst_seq']

    avg_loss = {}
    mean_abs_error = {}
    errors_per_bench = {}
    predicted_values = {}
    for model_name in models['output_types']:
        y = data[model_name]

        bench_names = data['names']

        aux_avg_loss, aux_mean_abs_error, aux_errors_per_bench, aux_predicted_values = evaluate_lstm(device, clocks, verbose, models, X, y, model_name, pc, data, bench_names)

        printing('\tEpoch %d - %s (%s) Absolute loss: %.4f' %(epoch, dataset_name, model_name, aux_avg_loss), no_output)
        printing('\tEpoch %d - %s (%s) Mean Absolute error: %.4f%%' %(epoch, dataset_name, model_name, aux_mean_abs_error), no_output)

        avg_loss[model_name] = aux_avg_loss
        mean_abs_error[model_name] = aux_mean_abs_error
        errors_per_bench[model_name] = np.array(aux_errors_per_bench)
        predicted_values[model_name] = aux_predicted_values

    return avg_loss, mean_abs_error, errors_per_bench, predicted_values

def trainPytorchModel(device, clocks, verbose, fast, pc, model_params, test_output_dir, original_data_train, original_data_val, original_data_test=[]):
    #unpacking arguments
    model_name = model_params['model_name']
    max_epochs = model_params['max_epochs']

    batch_size = model_params['encoder_params']['batch_size']
    never_stop = model_params['never_stop']
    no_output = model_params['no_output']

    #parameters for printing output
    output_dim = 1
    print_every = 2
    print_every = print_every*batch_size

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    counter_no_improvements = 0

    trainTime = time.time()
    torch.cuda.manual_seed(42)


    feature_vector_size = model_params['encoder_params']['hidden_size']

    #Pytorch Models
    encoder_lstm = mopy.EncoderLSTM(model_params['encoder_params']['vocab_size'], model_params['encoder_params']['embed_size'], feature_vector_size, model_params['encoder_params']['num_layers'], model_params['encoder_params']['dropout_prob'], batch_size, device).to(device)
    models = {'encoder': encoder_lstm, 'output_types': []}
    optimizers = {}
    if model_params['encoder_params']['optimizer_name'] == 'SGD':
        optimizers['encoder'] = torch.optim.SGD(encoder_lstm.parameters(), lr=model_params['encoder_params']['learning_rate'])
    elif model_params['encoder_params']['optimizer_name'] == 'Adam':
        optimizers['encoder'] = torch.optim.Adam(encoder_lstm.parameters(), lr=model_params['encoder_params']['learning_rate'])
    else:
        optimizers['encoder'] = torch.optim.Adagrad(encoder_lstm.parameters(), lr=model_params['encoder_params']['learning_rate'])

    for model_type in possible_outputs:
        if model_params['outputs'][model_type] == False:
            if pc == True:
                model_params['nn_params'][model_type]['input_size'] = feature_vector_size+2+len(metrics_to_use)
            else:
                model_params['nn_params'][model_type]['input_size'] = feature_vector_size+2

            #create neural network
            models[model_type] = mopy.multiNN2(model_params['nn_params'][model_type]['input_size'], 1, model_params['nn_params'][model_type]['hidden_sizes'], model_params['nn_params'][model_type]['dropout_prob']).to(device)
            models['output_types'].append(model_type)

            #select correct optimizer
            if model_params['nn_params'][model_type]['optimizer_name'] == 'SGD':
                optimizers[model_type] = torch.optim.SGD(models[model_type].parameters(), lr=model_params['nn_params'][model_type]['learning_rate'])
            elif model_params['nn_params'][model_type]['optimizer_name'] == 'Adam':
                optimizers[model_type] = torch.optim.Adam(models[model_type].parameters(), lr=model_params['nn_params'][model_type]['learning_rate'])
            else:
                optimizers[model_type] = torch.optim.Adagrad(models[model_type].parameters(), lr=model_params['nn_params'][model_type]['learning_rate'])

    num_output_models = len(models) - 1 - 1
    printing("Num output models: %d" %num_output_models, no_output)
    printing(models['output_types'], no_output)
    printing('Training model (vocab_size: %d, max_epochs: %d, batch_size: %d, num_output_models: %d, using_performance_metrics: %r)...' % (model_params['encoder_params']['vocab_size'], max_epochs, batch_size, num_output_models, pc), False)
    printing('\tLSTM Encoder -> eta: %f, drop_prob: %f, opt: %s, num_layers: %d, layer_size: %d, embed_size: %d' %(model_params['encoder_params']['learning_rate'], model_params['encoder_params']['dropout_prob'], model_params['encoder_params']['optimizer_name'], model_params['encoder_params']['num_layers'], model_params['encoder_params']['hidden_size'], model_params['encoder_params']['embed_size']), False)
    for model_type in possible_outputs:
        if model_params['outputs'][model_type] == False:
            printing('\tOutput \'%s\' -> eta: %f, drop_prob: %f, opt: %s, num_layers: %d, hidden_layer_sizes: %s' %(model_type, model_params['nn_params'][model_type]['learning_rate'], model_params['nn_params'][model_type]['dropout_prob'], model_params['nn_params'][model_type]['optimizer_name'], model_params['nn_params'][model_type]['num_layers'], model_params['nn_params'][model_type]['hidden_sizes']), False)

    loss_function = torch.nn.SmoothL1Loss()
    losses = []

    num_ubench_train = len(original_data_train['names'])
    printing("\tNumber programs training: %d" %num_ubench_train, no_output)
    # if fast == 0:
    num_ubench_val = len(original_data_val['names'])
    printing("\tNumber programs validation: %d" %num_ubench_val, no_output)

    num_bench_test = 0
    if original_data_test != []:
        num_bench_test = len(original_data_test['names'])
        printing("\tNumber programs testing: %d" %num_bench_test, no_output)

    num_clocks_mem = clocks['num_mem_clocks']
    num_clocks_core = clocks['num_core_clocks']

    results_train = {'loss_per_epoch': [], 'abs_error_per_epoch': [], 'errors_per_bench_all_epochs': [], 'last_epoch_predictions': []}
    results_val = {'loss_per_epoch': [], 'abs_error_per_epoch': [], 'errors_per_bench_all_epochs': [], 'last_epoch_predictions': []}
    results_test = {'loss_per_epoch': [], 'abs_error_per_epoch': [], 'errors_per_bench_all_epochs': [], 'last_epoch_predictions': []}

    data_train = reorderData(original_data_train, pc)

    data_val = reorderData(original_data_val, pc)
    if original_data_test != []:
        data_test = reorderData(original_data_test, pc)

    previous_mae = {}
    error_too_high = 0
    for epoch in range(max_epochs):
        printing("Starting Epoch %d" %epoch, no_output)
        batch_index = 0
        start = time.time()
        aux_iter=0
        model_type = epoch % num_output_models #to make sure all benchmarks are used to train the different output models

        while batch_index + batch_size <= num_ubench_train:
            next_index = batch_index + batch_size
            input_tensor =  tensorFromBatch(data_train['inst_seq'], batch_index, batch_size, device)

            target_train = data_train[models['output_types'][model_type]]
            target_tensor =  torch.tensor(target_train[batch_index:batch_index+batch_size], dtype=torch.float).to(device)

            metrics_batch={}
            if pc == True:
                for metric_id, metric in enumerate(metrics_to_use):
                    metrics_batch[metric] = data_train[metric][batch_index:batch_index+batch_size]

            loss = trainBatch(device, clocks, model_name, input_tensor, target_tensor, models, models['output_types'][model_type], optimizers, loss_function, batch_size, pc, metrics_batch)

            print_loss_total += loss
            plot_loss_total += loss

            batch_index = next_index

            model_type = (model_type + 1) % num_output_models
            aux_iter += batch_size
            if aux_iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                if verbose == True:
                    print('\t%s (%d: %.1f%%) %.4f' % (timeSince(start, aux_iter / num_ubench_train), aux_iter, aux_iter / num_ubench_train * 100, print_loss_avg))

        if batch_index < num_ubench_train:
            batch_size_aux = num_ubench_train - batch_index
            input_tensor =  tensorFromBatch(data_train['inst_seq'], batch_index, batch_size_aux, device)

            target_train = data_train[models['output_types'][model_type]]
            target_tensor =  torch.tensor(target_train[batch_index:batch_index+batch_size_aux], dtype=torch.float).to(device)

            metrics_batch={}
            if pc == True:
                for metric_id, metric in enumerate(metrics_to_use):
                    metrics_batch[metric] = data_train[metric][batch_index:batch_index+batch_size_aux]

            loss = trainBatch(device, clocks, model_name, input_tensor, target_tensor, models, models['output_types'][model_type], optimizers, loss_function, batch_size_aux, pc, metrics_batch)

            print_loss_total += loss
            plot_loss_total += loss

            aux_iter += batch_size_aux
            model_type = (model_type + 1) % num_output_models
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            if verbose == True:
                print('\t%s (%d: %.1f%%) %.4f' % (timeSince(start, aux_iter / num_ubench_train), aux_iter, aux_iter / num_ubench_train * 100, print_loss_avg))

        printing('\tEpoch %d Training Duration: %s' %(epoch, asMinutes(time.time()-start)), no_output)

        if fast == 0:
            #evaluate on training set
            temp_avg_loss,  temp_aux_mean_abs_error,  temp_errors_per_bench, results_train['last_epoch_predictions'] = evaluate(device, clocks, verbose, no_output, epoch, 'Training', models, data_train, pc)
            results_train['loss_per_epoch'].append(temp_avg_loss)
            results_train['abs_error_per_epoch'].append(temp_aux_mean_abs_error)
            results_train['errors_per_bench_all_epochs'].append(temp_errors_per_bench)

            if SAVE_STATE == True and (epoch+1) % SAVE_EVERY_TH_EPOCH == 0:
                temp_avg_loss,  temp_aux_mean_abs_error,  temp_errors_per_bench, results_val['last_epoch_predictions'] = evaluate(device, clocks, verbose, no_output, epoch, 'Validation', models, data_val, pc)
                results_val['loss_per_epoch'].append(temp_avg_loss)
                results_val['abs_error_per_epoch'].append(temp_aux_mean_abs_error)
                results_val['errors_per_bench_all_epochs'].append(temp_errors_per_bench)

                if original_data_test != []:
                    #evaluate on testing set
                    temp_avg_loss,  temp_aux_mean_abs_error,  temp_errors_per_bench, results_test['last_epoch_predictions'] = evaluate(device, clocks, verbose, no_output, epoch, 'Testing', models, data_test, pc)
                    results_test['loss_per_epoch'].append(temp_avg_loss)
                    results_test['abs_error_per_epoch'].append(temp_aux_mean_abs_error)
                    results_test['errors_per_bench_all_epochs'].append(temp_errors_per_bench)

                saveIntermediateState(test_output_dir, epoch, models, model_params, results_train, results_val, results_test, data_train, data_val, data_test)

            if epoch % 5 == 0:
                #evaluate on validation set
                temp_avg_loss,  temp_aux_mean_abs_error,  temp_errors_per_bench, results_val['last_epoch_predictions'] = evaluate(device, clocks, verbose, no_output, epoch, 'Validation', models, data_val, pc)
                results_val['loss_per_epoch'].append(temp_avg_loss)
                results_val['abs_error_per_epoch'].append(temp_aux_mean_abs_error)
                results_val['errors_per_bench_all_epochs'].append(temp_errors_per_bench)

                if epoch > 0:
                    for model_type in models['output_types']:
                        if np.absolute(temp_aux_mean_abs_error[model_type] - previous_mae[model_type]) < threshold_consecutive_epochs:
                            counter_no_improvements += 1
                            break
                        else:
                            counter_no_improvements = 0

                previous_mae = temp_aux_mean_abs_error

                if counter_no_improvements == max_consecutive_epochs_no_improvement:
                    printing('\tLimit of consecutive epochs without improvement was achieved', no_output)
                    error_too_high = 1
                else:
                    for model_type in models['output_types']:
                        if epoch > 5 and temp_aux_mean_abs_error[model_type] > 50:
                            printing('\tError is too high (%s) (>50%% after 5 epochs)' %(model_type), no_output)
                            error_too_high = 1

            elif epoch == max_epochs-1:
                temp_avg_loss,  temp_aux_mean_abs_error,  temp_errors_per_bench, results_val['last_epoch_predictions'] = evaluate(device, clocks, verbose, no_output, epoch, 'Validation', models, data_val, pc)
                results_val['loss_per_epoch'].append(temp_avg_loss)
                results_val['abs_error_per_epoch'].append(temp_aux_mean_abs_error)
                results_val['errors_per_bench_all_epochs'].append(temp_errors_per_bench)

            if epoch == max_epochs-1 or (error_too_high == 1 and never_stop == False): #if last epoch
                if original_data_test != []:
                    #evaluate on testing set
                    temp_avg_loss,  temp_aux_mean_abs_error,  temp_errors_per_bench, results_test['last_epoch_predictions'] = evaluate(device, clocks, verbose, no_output, epoch, 'Testing', models, data_test, pc)
                    results_test['loss_per_epoch'].append(temp_avg_loss)
                    results_test['abs_error_per_epoch'].append(temp_aux_mean_abs_error)
                    results_test['errors_per_bench_all_epochs'].append(temp_errors_per_bench)



            printing('\tEpoch %d Full Duration: %s' %(epoch, asMinutes(time.time()-start)), no_output)

        if never_stop == False and error_too_high == 1:
            break

    endTime = time.time()
    printing('Training Time: %s' %asMinutes(endTime-trainTime), False)

    return (endTime-trainTime), models, results_train, results_val, results_test
