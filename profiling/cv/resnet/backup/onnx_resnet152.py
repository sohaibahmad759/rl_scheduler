from torchvision import models, datasets, transforms as T
import torch
from PIL import Image
import numpy as np

resnet50 = models.resnet50(pretrained=True)

# Read the categories
with open("images/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Export the model to ONNX
image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
torch_out = resnet50(x)
torch.onnx.export(resnet50,                     # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",              # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=12,             # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'])    # the model's output names

# Pre-processing for ResNet-50 Inferencing, from https://pytorch.org/hub/pytorch_vision_resnet/
resnet50.eval()  
filename = 'images/cat.jpg' # change to your filename

input_image = Image.open(filename)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
print("GPU Availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    resnet50.to('cuda')

# Inference with ONNX Runtime
from onnxruntime import *
from onnx import numpy_helper
import time

# session_fp32 = InferenceSession("resnet50.onnx", providers=['CPUExecutionProvider'])
session_fp32 = InferenceSession("resnet50.onnx", providers=['CUDAExecutionProvider'])
# session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['OpenVINOExecutionProvider'])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

latency = []
def run_sample(session, image_file, categories, inputs):
    start = time.time()
    input_arr = inputs.cpu().detach().numpy()
    ort_outputs = session.run([], {'input':input_arr})[0]
    latency.append(time.time() - start)
    output = ort_outputs.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    # for catid in top5_catid:
    #     print(categories[catid], output[catid])
    return ort_outputs

trials = 10000
for i in range(trials):
    ort_output = run_sample(session_fp32, 'cat.jpg', categories, input_batch)

fifty_pct = sorted(latency)[int(len(latency)/2)]
ninety_pct = sorted(latency)[int(len(latency)/10*9)]
avg = sum(latency)/len(latency)
minimum = min(latency)
maximum = max(latency)

print("ONNX Runtime CPU/GPU/OpenVINO Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))
print()
print('------------------------------------------')
# print(f'{idx}/{num_models}, Model: {model_name}, batch size: {batch_size}')
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


