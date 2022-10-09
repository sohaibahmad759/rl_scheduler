# Tutorial from https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=3b6da290c21bbbbf418577f3e2c528986a2965c5&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6f6e6e782f6b657261732d6f6e6e782f336236646132393063323162626262663431383537376633653263353238393836613239363563352f7475746f7269616c2f54656e736f72466c6f775f4b657261735f456666696369656e744e65742e6970796e62&logged_in=false&nwo=onnx%2Fkeras-onnx&path=tutorial%2FTensorFlow_Keras_EfficientNet.ipynb&platform=android&repository_id=162340677&repository_type=Repository&version=98

# Note: This scipt is not used to export the EfficientNet models used in the paper

import numpy as np
import efficientnet.tfkeras as efn
# from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
# from efficientnet.preprocessing import center_crop_and_resize
# from skimage.io import imread
model = efn.EfficientNetB0(weights='imagenet')

import keras2onnx
output_model_path = "keras_efficientNet.onnx"
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, output_model_path)