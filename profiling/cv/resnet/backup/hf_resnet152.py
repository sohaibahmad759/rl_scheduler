import time
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")

inputs = feature_extractor(image, return_tensors="pt")

latencies = []
with torch.no_grad():
    for i in range(100):
        start_time = time.time()
        logits = model(**inputs).logits
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

print(f'latencies: {latencies}')
print()
print(f'50th percentile: {sorted(latencies)[int(len(latencies)/2)]}')
print(f'90th percentile: {sorted(latencies)[int(len(latencies)/10*9)]}')
print(f'average: {sum(latencies)/len(latencies)}')
print(f'minimum: {min(latencies)}')
print(f'maximum: {max(latencies)}')

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
