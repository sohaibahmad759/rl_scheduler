# This file takes in all the profiling results from a given folder
# and puts them all in a single file

import glob
import pandas as pd

def normalize_accuracy(accuracy_dict: dict, model_families: list[str]):
    '''
    Takes in an accuracy dictionary and list of models, normalizes
    the accuracy for each model in the list
    '''
    normalized_accuracy = {}
    for model_family in model_families:
        max_acc = 0
        for model in accuracy_dict:
            if model_family in model:
                accuracy = accuracy_dict[model]
                max_acc = max(max_acc, accuracy)
        for model in accuracy_dict:
            if model_family in model:
                normalized_accuracy[model] = accuracy_dict[model] / max_acc * 100
    return normalized_accuracy
        
accuracy_df = pd.read_csv('accuracy.csv')
accuracy = dict(zip(accuracy_df.Model, accuracy_df.Accuracy))
model_families = ['bert', 'yolo', 't5', 'gpt2', 'densenet', 'resnet', 'resnest',
                  'mobilenet', 'vgg', 'efficientnet']
normalized_accuracy = normalize_accuracy(accuracy, model_families)


input_folder = 'profiled'
readfiles = sorted(glob.glob(f'{input_folder}/*.csv'))

wf = open('aggregate_profiled.csv', mode='w')

header = 'Model,Accelerator,Trials,Batch_size,50th_pct,90th_pct,Average,Min,Max,Accuracy,Normalized_Accuracy'
wf.write(header + '\n')

print(readfiles)
for filename in readfiles:
    model_family, accelerator, _ = filename.split('/')[1].split('_', 2)
    with open(filename, mode='r') as rf:
        print(f'model_family: {model_family}, accelerator: {accelerator}')
        lines = rf.readlines()
        lines = list(map(lambda x: x.replace('PyTorch', accelerator), lines))
        lines = list(map(lambda x: x.replace('cpu', 'onnxruntime_cpu'), lines))
        lines = list(map(lambda x: x.replace('1080ti', 'onnxruntime_gpu_pascal'), lines))
        lines = list(map(lambda x: x.replace('v100', 'onnxruntime_gpu_ampere'), lines))
        lines = lines[1:]
        # # Converting from seconds to milliseconds

        # add the accuracy value
        newlines = []
        for line in lines:
            separated = line.rstrip('\n').split(',')
            # converting latency from seconds to milliseconds
            separated[4:9] = [str(float(i) * 1000) for i in separated[4:9]]

            # adding accuracy and normalized accuracy
            model = separated[0]
            separated.append(f'{accuracy[model]},{normalized_accuracy[model]}\n')
            newline = ','.join(separated)
            newlines.append(newline)

        wf.writelines(newlines)

# Join previous profiled results from BLIS as well
with open('blis/batch_size_n.csv', mode='r') as rf:
    lines = rf.readlines()
    lines = list(map(lambda x: x.replace('.onnx', ''), lines))
    lines = lines[1:]
    
    newlines = []
    for line in lines:
        separated = line.split(',')
        model = separated[0]
        accelerator = separated[2]
        trials = ''
        batch_size = separated[3]
        average = separated[4]
        _50th_pct = average
        _90th_pct = ''
        minimum = separated[5]
        maximum = separated[6]

        newline = (f'{model},{accelerator},{trials},{batch_size},{_50th_pct},'
                  f'{_90th_pct},{average},{minimum},{maximum},{accuracy[model]},'
                  f'{normalized_accuracy[model]}\n')
        wf.write(newline)

wf.close()
