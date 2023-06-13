import os
import time
import torch
from PIL import Image
# import tensorflow as tf
from transformers import AutoModelForImageClassification, AutoImageProcessor, MobileNetV1ForImageClassification, MobileNetV1ImageProcessor

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

# print(tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(True)

wf = open('mobilenet_profiled.csv', mode='w')
wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

models = ['mobilenet0.25', 'mobilenet0.5', 'mobilenet0.75', 'mobilenet1.0']
# text = "Hello, my dog is cute"
batch_sizes = [1, 2, 4, 8]
trials = 100

num_models = len(models) * len(batch_sizes)
idx = 0

platform = 'PyTorch'

for model_name in models:
    for batch_size in batch_sizes:
        idx += 1

        name_conversion = {'mobilenet0.25': 'google/mobilenet_v2_0.25_224',
                                  'mobilenet0.5': 'google/mobilenet_v2_0.5_224',
                                  'mobilenet0.75': 'google/mobilenet_v2_0.75_224',
                                  'mobilenet1.0': 'google/mobilenet_v2_1.0_224'}
        model_variant_name = name_conversion[model_name]

        image_processor = MobileNetV1ImageProcessor.from_pretrained(model_variant_name)
        model = MobileNetV1ForImageClassification.from_pretrained(model_variant_name)

        input_batch = image_processor([Image.open('images/cat.jpg')]*batch_size, return_tensors="pt")

        # if 'roberta' in model_name:
        #     tokenizer = RobertaTokenizer.from_pretrained(model_name)
        #     model = RobertaForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir,
        #                                                              return_dict=True).to(device)
        # elif 'albert' in model_name:
        #     tokenizer = AlbertTokenizer.from_pretrained(model_name)
        #     model = AlbertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir,
        #                                                             return_dict=True).to(device)
        # elif 'bert' in model_name:
        #     tokenizer = BertTokenizer.from_pretrained(model_name)
        #     model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir,
                                                                #   return_dict=True).to(device)
        # model = TFRobertaModel.from_pretrained(model_name)
        # text = "Hello, my dog is cute"
        # encoded_input = tokenizer(text, return_tensors='pt').to(device)
        # # encoded_input = tokenizer(text, return_tensors='tf')

        # sequences = ["Hello, my dog is cute"] * batch_size
        # input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        # labels = torch.tensor([[1]] * batch_size).to(device) # Labels depend on the task

        latencies = []
        # with tf.device('/GPU:0'):
        for i in range(trials):
            start_time = time.time()
            # output = model(**encoded_input, labels=labels)
            # output = model(input_ids, labels=labels)
            output = model(**input)
            # output = model(encoded_input)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            # print(f'output: {output}')
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
