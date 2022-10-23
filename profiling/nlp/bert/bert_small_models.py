import time
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

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

wf = open('bert_small_profiled.csv', mode='w')
wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

models = ['prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-small',
          'prajjwal1/bert-medium']
text = 'Hello, my dog is cute'
batch_sizes = [1, 2, 4, 8, 16]
trials = 1000

num_models = len(models) * len(batch_sizes)
idx = 0

platform = 'PyTorch'

for model_name in models:
    for batch_size in batch_sizes:
        idx += 1
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny") # v1 and v2
        model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", cache_dir=cache_dir,
                                        return_dict=True).to(device) # v1 and v2

        sequences = [text] * batch_size
        input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)

        latencies = []
        # with tf.device('/GPU:0'):
        for i in range(trials):
            start_time = time.time()
            output = model(input_ids)
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
print(output)
