import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

wf = open('gpt2_profiled.csv', mode='w')
wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
sentence = 'Hello, my dog is cute'
batch_sizes = [1, 2, 4, 8, 16]
# batch_sizes = [1]
trials = 100

num_models = len(models) * len(batch_sizes)
idx = 0

platform = 'PyTorch'

for model_name in models:
    for batch_size in batch_sizes:
        idx += 1
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir,
                                                return_dict=True).to(device)

        model.eval()
        sequences = ["Hello, my dog is cute"] * batch_size
        input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)

        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

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

print(outputs)
