import onnxruntime as ort
import os

model_architecture = 'vgg'
model_name = model_architecture + '19'
# model_name = 'resnet18_v1/resnet18_v1'
sess = ort.InferenceSession(os.path.join('onnx', model_architecture, model_name, model_name + '.onnx'))
# sess = ort.InferenceSession(os.path.join('onnx', model_name + '.onnx'))

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)
print(model_name)
