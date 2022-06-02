# Tutorial from https://chai-bapat.medium.com/5min-gluon-model-to-onnx-e8ed0eea754e

import os
from gluoncv import model_zoo
import numpy as np
import mxnet as mx

model_names = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

for model_name in model_names:
    directory = os.path.join('onnx', model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # download model
    model = model_zoo.get_model(model_name, pretrained=True)
    print(model_name+' downloaded')

    os.chdir(directory)

    # filename = os.path.join(directory, model_name)

    # convert to symbol
    model.hybridize()
    print(model_name+' hybridized')
    input_shape=(1,3,224,224)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    mx_data = mx.nd.array(data_array)
    model(mx_data)

    model.export(model_name)
    print(model_name+' exported')

    #convert using onnx
    # from mxnet.contrib import onnx as onnx_mxnet
    from mxnet.onnx import export_model
    onnx_file='./'+model_name+'.onnx'
    # onnx_file = filename + '.onnx'
    params = './'+model_name+'-0000.params'
    # params = filename + '-0000.params'
    #sym = mx.sym.load('./resnetfifty-symbol.json')
    sym='./'+model_name+'-symbol.json'
    # sym = filename + '-symbol.json'
    # onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)
    export_model(sym, params, [input_shape], np.float32, onnx_file)
    print('onnx export done')

    os.chdir('../..')