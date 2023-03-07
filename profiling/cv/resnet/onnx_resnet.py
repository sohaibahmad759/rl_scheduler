import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from torchvision import models, datasets, transforms as T
import torch
from PIL import Image
import numpy as np
import onnxruntime as ort
from onnx import numpy_helper

# Adjust session options
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
# opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
# opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

ort.set_default_logger_severity(3)

models = ['models/resnet18_v1.onnx', 'models/resnet34_v1.onnx', 'models/resnet50_v1.onnx',
          'models/resnet101_v1.onnx', 'models/resnet152_v1.onnx']
batch_sizes = [1, 2, 4, 8, 16]
trials = 100
platform = 'onnxruntime'
num_models = len(models) * len(batch_sizes)
idx = 0

wf = open('resnet_profiled.csv', mode='w')
wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

# Read the categories
with open("images/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

filename = 'images/cat.jpg'

input_image = Image.open(filename)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

for model_name in models:
    for batch_size in batch_sizes:
        idx += 1
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.repeat(batch_size, 1, 1, 1)
        print(f'input_batch shape: {input_batch.shape}')
        # time.sleep(10)

        # move the input and model to GPU for speed if available
        print("GPU Availability: ", torch.cuda.is_available())
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        session_fp32 = ort.InferenceSession(model_name, sess_options=opts, providers=['CPUExecutionProvider'])
        # session_fp32 = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])
        # session_fp32 = InferenceSession("models/resnest269.onnx", providers=['CUDAExecutionProvider'])

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        latencies = []
        def run_sample(session, image_file, categories, inputs):
            start_time = time.time()
            input_arr = inputs.cpu().detach().numpy()
            ort_outputs = session.run([], {'data':input_arr})[0]
            end_time = time.time()
            latencies.append(end_time - start_time)
            # # flattening only works for batch size of 1
            # output = ort_outputs.flatten()
            # output = softmax(output) # this is optional
            # print(f'shape: {output.shape}, output: {output}')
            # top5_catid = np.argsort(-output)[:5]
            # print(f'top5_catid: {top5_catid}')
            # for catid in top5_catid:
            #     print(categories[catid], output[catid])
            return ort_outputs

        for i in range(trials):
            ort_output = run_sample(session_fp32, 'cat.jpg', categories, input_batch)

        fifty_pct = sorted(latencies)[int(len(latencies)/2)]
        ninety_pct = sorted(latencies)[int(len(latencies)/10*9)]
        avg = sum(latencies)/len(latencies)
        minimum = min(latencies)
        maximum = max(latencies)

        print("ONNX Runtime CPU/GPU/OpenVINO Inference time = {} ms".format(format(sum(latencies) * 1000 / len(latencies), '.2f')))
        print()
        print('------------------------------------------')
        print(f'{idx}/{num_models}, Model: {model_name}, batch size: {batch_size}')
        print('------------------------------------------')
        print()
        print('50th percentile:')
        print(fifty_pct)
        print('90th percentile:')
        print(ninety_pct)
        print('average:')
        print(avg)
        print('minimum:')
        print(minimum)
        print('maximum:')
        print(maximum)
        print()

        wf.write(f'{model_name},{platform},{trials},{batch_size},{fifty_pct},'
                 f'{ninety_pct},{avg},{minimum},{maximum}\n')

wf.close()
