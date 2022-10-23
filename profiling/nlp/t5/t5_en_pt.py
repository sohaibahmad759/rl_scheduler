import time
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('unicamp-dl/translation-en-pt-t5')
tokenizer = AutoTokenizer.from_pretrained('unicamp-dl/translation-en-pt-t5')

model.eval()
sentence = 'translate English to Portuguese: Hello, my dog is cute'
input_ids = tokenizer(sentence, return_tensors='pt').input_ids

latencies = []
trials = 100
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

# print(outputs)
# outputs = miak xochitl istak
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(outputs)
