import time
import torch
from PIL import Image

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

wf = open('yolo_profiled.csv', mode='w')
wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

# models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
models = ['yolov5n']
batch_sizes = [1, 2, 4, 8, 16]
# batch_sizes = [1]
trials = 100

num_models = len(models) * len(batch_sizes)
idx = 0

platform = 'PyTorch'

for model_name in models:
    for batch_size in batch_sizes:
        idx += 1
        model = torch.hub.load('ultralytics/yolov5', model_name).to(device)

        images = [Image.open('images/cat.jpg')] * batch_size

        latencies = []
        for i in range(trials):
            start_time = time.time()
            results = model(images)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

        fifty_pct = sorted(latencies)[int(len(latencies)/2)]
        ninety_pct = sorted(latencies)[int(len(latencies)/10*9)]
        avg = sum(latencies)/len(latencies)
        minimum = min(latencies)
        maximum = max(latencies)
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

print(results)
