import time
# import torch
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel

# print()
# print('----------------------------')
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f'device: {device}')
# print('----------------------------')
# print()

print(tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(True)

models = ['roberta-base', 'roberta-large']

for model_name in models:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = TFRobertaModel.from_pretrained(model_name)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='tf')

    latencies = []
    with tf.device('/GPU:0'):
        for i in range(100):
            start_time = time.time()
            output = model(encoded_input)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
    print()
    print('----------------------------')
    print(f'Model: {model_name}')
    print('----------------------------')
    print()
    print(f'50th percentile: {sorted(latencies)[int(len(latencies)/2)]}')
    print(f'90th percentile: {sorted(latencies)[int(len(latencies)/10*9)]}')
    print(f'average: {sum(latencies)/len(latencies)}')
    print(f'minimum: {min(latencies)}')
    print(f'maximum: {max(latencies)}')
    print()