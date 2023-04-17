import json
import logging
import time
import torch
from PIL import Image

import sys
sys.path.append('../../')
from cluster_scripts.clusterpredictor import ClusterPredictor
from core.common import Event, EventType


batching_algorithm = 'aimd'
# batching_algorithm = 'infaas'
# batching_algorithm = 'proteus'

print()
print('----------------------------')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')
print('----------------------------')
print()

if 'cuda' in device:
    cache_dir = '/work/pi_rsitaram_umass_edu/sohaib/profiling/cache'
else:
    cache_dir = '~/.cache'

# wf = open('yolo_profiled.csv', mode='w')
# wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

# models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
# models = ['yolov5n']
model_name = 'yolov5n'
batch_sizes = [1, 2, 4, 8, 16]
# batch_sizes = [1]
# trials = 100

num_models = 1 * len(batch_sizes)
idx = 0

img_batch_1 = [Image.open('images/cat.jpg')]
img_batch_2 = [Image.open('images/cat.jpg')] * 2
img_batch_4 = [Image.open('images/cat.jpg')] * 4
img_batch_8 = [Image.open('images/cat.jpg')] * 8
img_batch_16 = [Image.open('images/cat.jpg')] * 16

platform = 'PyTorch'

model = torch.hub.load('ultralytics/yolov5', model_name).to(device)
# for i in range(10):
#     model(img_batch_1)
#     model(img_batch_2)
#     model(img_batch_4)
#     model(img_batch_8)
trace_file = open('../../logs/per_predictor/300ms/clipper_ht/5df0845ca00c424ca28d3f66aea70a77.txt')
trace = trace_file.readlines()
wf = open('trace.txt', mode='w')

variant_name, acc_type, max_batch_size, executor_name = trace[0].strip('\n').split(',')
# profiled_latencies = json.loads(trace[1].strip('\n').replace("'", '"'))
profiled_latencies = {('bert', 'prajjwal1/bert-tiny', 1): 0.9672641754150392, ('bert', 'prajjwal1/bert-tiny', 2): 1.0769367218017578, ('bert', 'prajjwal1/bert-tiny', 4): 1.341104507446289, ('bert', 'prajjwal1/bert-tiny', 8): 1.4221668243408203, ('densenet', 'densenet121', 1): 79.1618824005127, ('densenet', 'densenet121', 2): 156.43072128295898, ('densenet', 'densenet121', 4): 318.0832862854004, ('densenet', 'densenet121', 8): 641.7300701141357, ('efficientnet', 'efficientnet-b0', 1): 26.496171951293945, ('efficientnet', 'efficientnet-b0', 2): 53.8938045501709, ('efficientnet', 'efficientnet-b0', 4): 104.85291481018066, ('efficientnet', 'efficientnet-b0', 8): 210.1168632507324, ('gpt2', 'gpt2', 1): 263.30113410949707, ('gpt2', 'gpt2', 2): 365.9720420837402, ('gpt2', 'gpt2', 4): 357.96523094177246, ('gpt2', 'gpt2', 8): 410.3550910949707, ('mobilenet', 'mobilenet0.25', 1): 2.629995346069336, ('mobilenet', 'mobilenet0.25', 2): 5.167961120605469, ('mobilenet', 'mobilenet0.25', 4): 10.316848754882812, ('mobilenet', 'mobilenet0.25', 8): 20.428895950317383, ('resnest', 'resnest14', 1): 70.31989097595215, ('resnest', 'resnest14', 2): 133.44883918762207, ('resnest', 'resnest14', 4): 266.7930126190185, ('resnest', 'resnest14', 8): 532.6881408691406, ('resnet', 'resnet18_v1', 1): 47.19305038452149, ('resnet', 'resnet18_v1', 2): 92.91386604309082, ('resnet', 'resnet18_v1', 4): 184.28301811218265, ('resnet', 'resnet18_v1', 8): 367.50268936157227, ('t5', 't5-small', 1): 99.8671054840088, ('t5', 't5-small', 2): 174.64971542358398, ('t5', 't5-small', 4): 207.02290534973145, ('t5', 't5-small', 8): 247.06578254699707, ('yolo', 'yolov5n', 1): 37.5816822052002, ('yolo', 'yolov5n', 2): 74.18704032897949, ('yolo', 'yolov5n', 4): 116.99581146240234, ('yolo', 'yolov5n', 8): 226.62830352783203}

exp_start_time = time.time()
print(f'experiment started at: {exp_start_time}')

predictor = ClusterPredictor(logging_level=logging.DEBUG,
                             max_batch_size=int(max_batch_size),
                             batching_algo='aimd',
                             task_assignment='canary',
                             model_assignment='ilp',
                             profiled_latencies=profiled_latencies,
                             variant_name=variant_name)

current_clock = 0
for line in trace:
    if 'enqueued' in line:
        _, clock = line.strip('\n').split(',')
        clock = int(float(clock))
        event = Event(start_time=clock,
                      type=EventType.START_REQUEST,
                      deadline=300,
                      desc=executor_name)
        predictor.enqueue_request(event, clock)
        print(f'enqueued request at {clock}')
    # if 'process_batch' in line:
    #     _, clock, batch_size = line.strip('\n').split(',')
    #     clock = int(float(clock))
    #     batch_size = int(batch_size)

    #     if clock > current_clock:
    #         difference = clock - current_clock
    #         # time.sleep(difference / 1000)
    #         current_clock += difference
    #     else:
    #         print(f'oops, clock ({clock}) <= current_clock ({current_clock})')

    #     start_time = time.time()
    #     wf.write(f'process_batch,{current_clock},{batch_size}\n')
    #     if batch_size == 1:
    #         model(img_batch_1)
    #     elif batch_size == 2:
    #         model(img_batch_2)
    #     elif batch_size == 4:
    #         model(img_batch_4)
    #     elif batch_size == 8:
    #         model(img_batch_8)
    #     end_time = time.time()

    #     batch_process_time = (end_time - start_time) * 1000
    #     current_clock += batch_process_time
    #     wf.write(f'finish_batch_callback,{current_clock}\n')
    #     # new_time = time.time()
    #     # file_write_time = new_time - end_time
    #     # wf.write(f'file_write_overhead: {file_write_time}')

exp_end_time = time.time()
exp_time = exp_end_time - exp_start_time
print(f'experiment ended at: {exp_end_time}')
print(f'experiment time: {exp_time:.2f} seconds')

def enqueue_request(clock):
    return
