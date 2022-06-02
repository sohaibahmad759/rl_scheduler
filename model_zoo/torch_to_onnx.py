import torch
from efficientnet_pytorch import EfficientNet

# Specify which model to use
model_name = 'resnet-18'
image_size = EfficientNet.get_image_size(model_name)
print('Image size: ', image_size)

# Load model
model = EfficientNet.from_pretrained(model_name)
model.set_swish(memory_efficient=False)
model.eval()
print('Model image size: ', model._global_params.image_size)

# Dummy input for ONNX
dummy_input = torch.randn(1, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, model_name+'.onnx', verbose=False)