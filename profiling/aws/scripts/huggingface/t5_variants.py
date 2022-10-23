import time
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

print()
print('----------------------------')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')
print('----------------------------')
print()

cache_dir = '~/.cache'

wf = open('t5_profiled.csv', mode='w')
wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

# models = ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']
models = ['t5-small', 't5-base', 't5-large']
sentence = 'Hello, my dog is cute'
batch_sizes = [1, 2, 4, 8, 16]
trials = 1000

num_models = len(models) * len(batch_sizes)
idx = 0

platform = 'PyTorch'

for model_name in models:
    for batch_size in batch_sizes:
        idx += 1

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir,).to(device)

        model.eval()
        sequences = ['translate English to German: Hello, my dog is cute'] * batch_size
        # input_ids = tokenizer('translate English to German: ' + sentence, return_tensors='pt').input_ids.to(device)
        # input_ids = tokenizer('translate English to German: ' + sentence, return_tensors='pt').input_ids.to(device)
        input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)

        latencies = []
        for i in range(trials):
            start_time = time.time()
            outputs = model.generate(input_ids)
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

# print(outputs)
# outputs = miak xochitl istak
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(outputs)
