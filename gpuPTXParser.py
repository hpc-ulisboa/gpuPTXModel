import numpy as np
import sys
import matplotlib.pyplot as plt
import re #regular expressions library
from src.readFiles import readISA
from src.globalStuff import buffer_max_size

def printBuffers(buffer, buffer_producers):
    str_aux = '\tBuffer produced values: ['
    for id, value in enumerate(buffer):
        if id > 0:
            str_aux += ", "
        str_aux += "(%s<-%s)" %(value, "Core" if buffer_producers[id] == 0 else ("Memory" if buffer_producers[id] == 1 else  "SharedMem"))
    print(str_aux)

def addToBuffer(buffer, buffer_producers, value, memory_produced):
    if len(buffer) == buffer_max_size:
        buffer.pop()
        buffer_producers.pop()
    buffer.insert(0, value)
    buffer_producers.insert(0, memory_produced)

def checkDependencyOperand(buffer, buffer_producers, operand, dependency_inst, dependency_inst_type):
    if operand in buffer:
        idx_dependency = buffer.index(operand)
        if dependency_inst == 0 or idx_dependency+1 < dependency_inst:
            dependency_inst = idx_dependency+1
            dependency_inst_type = buffer_producers[idx_dependency]
        if verbose == True:
            print('\tDependency: operand "%s" was written into by a %s operation, %d instruction(s) ago' %(operand, "Core" if dependency_inst_type == 0 else ("Memory" if dependency_inst_type == 1 else  "SharedMem"),idx_dependency+1))
    return dependency_inst, dependency_inst_type

def readInstructionCounterFile(filepath, ISA, state_spaces, inst_types, isa_type):
    f = open(filepath)
    occurrences_global=np.zeros(len(ISA), dtype=np.int32)
    occurrences_per_kernel=[]
    sequence_per_kernel = []
    sequence_per_kernel_coded = [] #sequence format: 000(instruction) 0(state_space) 00(inst_type) 0(num_operands)
    kernel_count=0
    kernel_names = []
    buffer_past_insts = list()
    buffer_past_producers = list() #0 - produced by SM operation, 1 - produced by memory operation, 2 - produced by a shared memory operation

    shared_id = state_spaces.index('shared')

    if isa_type == 'cubin':
        keyword = 'global'
    else:
        keyword = 'globl'

    for line in f:
        if keyword in line:
            aux = line.split()
            if isa_type == 'cubin':
                kernel_name = aux[1]
            else:
                kernel_name = aux[2]

            if (kernel_count>0):
                occurrences_per_kernel.append(occurrences)
                sequence_per_kernel.append(kernel_sequence_aux)
                sequence_per_kernel_coded.append(kernel_sequence_coded_aux)
            occurrences = np.zeros(len(ISA), dtype=np.int32)
            kernel_sequence_aux = []
            kernel_sequence_coded_aux = []
            kernel_names.append(kernel_name)
            kernel_count += 1
        elif kernel_count > 0:
            # print('Line: "%s"' %line)
            line = line.strip()
            if not line == "":
                if verbose == True:
                    print('Instruction: "%s"' %line)
                full_inst_aux = line.split()[0]
                full_inst = full_inst_aux.split('.')
                inst = full_inst[0]
                num_operands = line.count(',') + 1 # this ignores instruction like ret which has no operands

                if inst in ISA:
                    if verbose == True:
                        print("\t%s" %full_inst)

                    aux_str = '%s' %inst
                    inst_id = ISA.index(inst)
                    occurrences[inst_id] += 1
                    occurrences_global[inst_id] += 1

                    mod_id = 0 # there are 8 state spaces (1-8) and 0 is the case when not state space is specified
                    inst_type_id = 0 # there are 8 state spaces (1-18) and 0 is the case when not instruction type is specified

                    if len(full_inst) > 2:
                        for mod in full_inst[1:-1]:
                            if mod in state_spaces:
                                # aux_str += ', state_space: %s' %(mod)
                                aux_str += '.%s' %(mod)
                                mod_id = state_spaces.index(mod) + 1 #0 is for the no modifier case
                                break

                    inst_type = full_inst[-1]
                    if inst_type in inst_types:
                        aux_str += '.%s' %inst_type
                        inst_type_id  = inst_types.index(inst_type) + 1 #0 is for the no instruction type case

                    #looking for dependencies of the instruction operands
                    dependency_inst = 0
                    dependency_inst_type = 0
                    if inst == 'st':
                        # value is written into memory from 2 operands
                        if 'v2' in full_inst:
                            #destination (address + operand to address the memory)
                            address=line.split()[1].split(',')[0]
                            operand_dest = address.split('[')[1].split(']')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand_dest, dependency_inst, dependency_inst_type)

                            #operands
                            test = re.search('{(.+?)}', line)
                            if test:
                                found = test.group(1)
                                operands_v2_st = found.split(',')
                                op1 = operands_v2_st[0].strip()
                                op2 = operands_v2_st[1].strip()
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op1, dependency_inst, dependency_inst_type)
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op2, dependency_inst, dependency_inst_type)

                        # value is written into memory from 4 operands
                        elif 'v4' in full_inst:
                            #destination (address + operand to address the memory)
                            address=line.split()[1].split(',')[0]
                            operand_dest = address.split('[')[1].split(']')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand_dest, dependency_inst, dependency_inst_type)

                            #operands
                            test = re.search('{(.+?)}', line)
                            if test:
                                found = test.group(1)
                                operands_v4_st = found.split(',')
                                op1 = operands_v4_st[0].strip()
                                op2 = operands_v4_st[1].strip()
                                op3 = operands_v4_st[2].strip()
                                op4 = operands_v4_st[3].strip()
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op1, dependency_inst, dependency_inst_type)
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op2, dependency_inst, dependency_inst_type)
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op3, dependency_inst, dependency_inst_type)
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op4, dependency_inst, dependency_inst_type)

                        # value is written into memory from a sinle operand
                        else:
                            #destination (address + operand to address the memory)
                            address=line.split()[1].split(',')[0]
                            operand_dest = address.split('[')[1].split(']')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand_dest, dependency_inst, dependency_inst_type)

                            op1 = line.split()[2].split(';')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, op1, dependency_inst, dependency_inst_type)

                        if verbose == True:
                            print("\tAddress: %s" %address)

                    elif inst == 'ld':
                        # value from memory is written into 2 destinations
                        if mod_id == shared_id:
                            mem_type = 2
                        else:
                            mem_type = 1
                        if 'v2' in full_inst:
                            #operands
                            address = line.split()[3].split(';')[0]
                            operand = address.split('[')[1].split(']')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand, dependency_inst, dependency_inst_type)

                            #destination
                            test = re.search('{(.+?)}', line)
                            if test:
                                found = test.group(1)
                                destinations_v2_load = found.split(',')
                                dest1 = destinations_v2_load[0].strip()
                                dest2 = destinations_v2_load[1].strip()
                                addToBuffer(buffer_past_insts, buffer_past_producers, dest1, mem_type)
                                addToBuffer(buffer_past_insts, buffer_past_producers, dest2, mem_type)

                        # value from memory is written into 4 destinations
                        elif 'v4' in full_inst:
                            #operands
                            address = line.split()[5].split(';')[0]
                            operand = address.split('[')[1].split(']')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand, dependency_inst, dependency_inst_type)

                            #destination
                            test = re.search('{(.+?)}', line)
                            if test:
                                found = test.group(1)
                                destinations_v4_load = found.split(',')
                                dest1 = destinations_v4_load[0].strip()
                                dest2 = destinations_v4_load[1].strip()
                                dest3 = destinations_v4_load[2].strip()
                                dest4 = destinations_v4_load[3].strip()
                                addToBuffer(buffer_past_insts, buffer_past_producers, dest1, mem_type)
                                addToBuffer(buffer_past_insts, buffer_past_producers, dest2, mem_type)
                                addToBuffer(buffer_past_insts, buffer_past_producers, dest3, mem_type)
                                addToBuffer(buffer_past_insts, buffer_past_producers, dest4, mem_type)

                        # value from memory is written into a single destination
                        else:
                            #operands
                            address = line.split()[2].split(',')[0].split(';')[0]
                            operand = address.split('[')[1].split(']')[0]
                            dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand, dependency_inst, dependency_inst_type)

                            #destination
                            dest1 = line.split()[1].split(',')[0].split(';')[0]
                            addToBuffer(buffer_past_insts, buffer_past_producers, dest1, mem_type)

                        if verbose == True:
                            print("\tAddress: %s" %address)

                    elif num_operands > 0:
                        #operands
                        if num_operands > 1:
                            if len(line.split()) < num_operands+1:
                                #special case found where ex2.approx.ftz.f32 instruction had a different identation than ALL other instructions
                                operand = line.split()[1].split(',')[0]
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand, dependency_inst, dependency_inst_type)
                                operand = line.split()[1].split(',')[1].split(';')[0]
                                dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand, dependency_inst, dependency_inst_type)
                            else:
                                for operand_id in range(2,num_operands+1):
                                    operand = line.split()[operand_id].split(',')[0].split(';')[0]
                                    dependency_inst, dependency_inst_type = checkDependencyOperand(buffer_past_insts, buffer_past_producers, operand, dependency_inst, dependency_inst_type)

                        #destination
                        dest_reg = line.split()[1].split(',')[0].split(';')[0]
                        addToBuffer(buffer_past_insts, buffer_past_producers, dest_reg, 0)

                    if verbose == True:
                        if dependency_inst == 0:
                            print('\n\tThis instruction has no dependencies\n')
                        else:
                            print('\n\tThis instruction has predominant dependency to a value written by a %s operation, %d instruction(s) ago\n' %("Core" if dependency_inst_type == 0 else ("Memory" if dependency_inst_type == 1 else  "SharedMem"), dependency_inst))
                        printBuffers(buffer_past_insts, buffer_past_producers)

                    kernel_sequence_coded_aux.append('%03d%01d%02d%01d%01d%01d' % (inst_id, mod_id, inst_type_id, num_operands, dependency_inst, dependency_inst_type))
                    kernel_sequence_aux.append(aux_str)

    occurrences_per_kernel.append(occurrences)
    sequence_per_kernel_coded.append(kernel_sequence_coded_aux)
    sequence_per_kernel.append(kernel_sequence_aux)
    f.close()
    return occurrences_global, occurrences_per_kernel, sequence_per_kernel, sequence_per_kernel_coded, kernel_names

def printOccurrences(ISA, occurrences):
    for inst_id, inst in enumerate(ISA):
        if(occurrences[inst_id]>0):
            print('\t%s: %d' %(inst, occurrences[inst_id]))

def printOccurrencesPerKernel(ISA, occurrences_per_kernel, kernel_names):
    for kernel_id, occurrences in enumerate(occurrences_per_kernel):
        print('Kernel %d (%s) --- (%d instructions):' %(kernel_id, kernel_names[kernel_id], np.sum(occurrences)))
        printOccurrences(ISA, occurrences)

def writeOutputSequence(filename, sequences):
    for kernel_id, sequence in enumerate(sequences):
        f = open('%s_%d.csv' %(filename, kernel_id), 'w+')
        for line_id, line in enumerate(sequence):
            f.write('%s\n' %(line))
        f.close()

def writeOutputFiles(ISA, occurrences_per_kernel, sequence_per_kernel, sequence_per_kernel_coded, kernel_names):
    f = open('outputOccurrences_per_kernel.csv', 'w+')
    for kernel_id, occurrences in enumerate(occurrences_per_kernel):
        line_kernel=kernel_names[kernel_id]
        for inst_id, inst in enumerate(ISA):
            line_kernel += ',%d' %(occurrences[inst_id])
        line_kernel +='\n'
        f.write(line_kernel)
    f.close()

    writeOutputSequence('outputSequenceReadable_kernel', sequence_per_kernel)
    writeOutputSequence('outputSequence_kernel_depend', sequence_per_kernel_coded)

def showHistogramInstructions(ISA, occurrences_per_kernel, kernel_names):
    num_kernels=len(kernel_names)
    aux_rows = 2 if num_kernels>4 else 1

    plt.figure(1,figsize=(16.53,11.69))
    for kernel_id, occurrences in enumerate(occurrences_per_kernel):
        ax = plt.subplot(aux_rows, int(num_kernels/aux_rows), kernel_id+1)
        ax.set_title('Kernel %d' %(kernel_id))
        bar = ax.bar(np.arange(0,len(ISA)), occurrences, align='center')
        ax.axis([0, len(ISA), 0, int(np.max(occurrences)*1.1)])
        ax.set_xticks(np.arange(0,len(ISA)))
        ax.set_xticklabels(ISA, rotation=45, fontsize = 5)
        ax.set_ylabel('Number of instructions')

        ind = np.argwhere(occurrences > 0)
        for i in ind:
            j=i[0]
            h = bar[j].get_height()
            ax.text(bar[j].get_x()+bar[j].get_width()/2.0, h, '%s' %(ISA[j]), ha='center', va='bottom', rotation=90, fontsize=10)

    plt.tight_layout()
    plt.savefig('instructions_count_histogram.pdf')
    plt.close()

def main():
    """Main function."""
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    isa_type = 'PTX'

    # parser.add_argument('isa_type', type=str)
    parser.add_argument('isa_files_path', type=str)
    parser.add_argument('benchmark_source_file', type=str)
    parser.add_argument('--v', action='store_const', const=True, default=False)
    parser.add_argument('--histogram', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())
    print(args)

    # isa_type = args['isa_type']
    isa_files_path =  args['isa_files_path']
    sass_file_path = args['benchmark_source_file']
    global verbose
    verbose = args['v']
    histogram = args['histogram']

    isa_file_path = '%s/ptx_isa.txt' %isa_files_path
    state_spacesfile_path = '%s/ptx_state_spaces.txt' %isa_files_path
    instruction_types_file_path = '%s/ptx_instruction_types.txt' %isa_files_path

    ISA = readISA(isa_file_path)
    print('\n\n======================= ARCHITECTURE =======================\n')
    print("ISA has %d instructions" %(len(ISA)))
    state_spaces = readISA(state_spacesfile_path)
    print("state_spaces has %d spaces" %(len(state_spaces)))
    inst_types = readISA(instruction_types_file_path)
    print("inst_types has %d types" %(len(inst_types)))

    if verbose == True:
        print('\n\n================== BEGINNING PTX PARSING ==================\n')
    occurrences, occurrences_per_kernel, sequence_per_kernel, sequence_per_kernel_coded, kernel_names = readInstructionCounterFile(sass_file_path, ISA, state_spaces, inst_types, isa_type)

    print('\n===================== KERNEL(S) SUMMARY ====================\n')

    print('Source file has %d kernels, with a total of %d instructions\n' %(len(occurrences_per_kernel), np.sum(occurrences)))
    printOccurrencesPerKernel(ISA, occurrences_per_kernel, kernel_names)

    writeOutputFiles(ISA, occurrences_per_kernel, sequence_per_kernel, sequence_per_kernel_coded, kernel_names)

    if histogram == True:
        showHistogramInstructions(ISA, occurrences_per_kernel, kernel_names)

    print('\n========================== THE END ==========================\n')


if __name__ == "__main__":
    main()
