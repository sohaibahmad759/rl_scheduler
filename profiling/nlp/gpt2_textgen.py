import time
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

trials = 100
latencies = []
for i in range(trials):
    start_time = time.time()
    output = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    end_time = time.time()
    latency = end_time - start_time
    latencies.append(latency)

fifty_pct = sorted(latencies)[int(len(latencies)/2)]
ninety_pct = sorted(latencies)[int(len(latencies)/10*9)]
avg = sum(latencies)/len(latencies)
minimum = min(latencies)
maximum = max(latencies)
print()
# print('------------------------------------------')
# print(f'{idx}/{num_models}, Model: {model_name}, batch size: {batch_size}')
# print('------------------------------------------')
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

# print(output)
