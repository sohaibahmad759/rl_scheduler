# Note: We use this script to generate the EfficientNet models
#       used in the paper

import torch
from efficientnet_pytorch import EfficientNet

# Specify which model to use
model_name = 'efficientnet-b7'
image_size = EfficientNet.get_image_size(model_name)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(model_name)
model.set_swish(memory_efficient=False)
model.eval()
print('Model image size: ', model._global_params.image_size)

# Dummy input for ONNX
batch_size = 1
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model,
                dummy_input,
                'onnx/'+model_name+'.onnx',
                export_params=True,
                input_names=['input'],   # the model's input names
                output_names=['output'],  # the model's output names
                dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                            'output': {0: 'batch_size'}},
                verbose=False)
