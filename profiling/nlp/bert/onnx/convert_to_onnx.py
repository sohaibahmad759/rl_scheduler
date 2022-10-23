from pathlib import Path
import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, BertTokenizer, BertForSequenceClassification

# load model and tokenizer
model_id = "bert-base-uncased"
feature = "sequence-classification"
model = BertForSequenceClassification.from_pretrained(model_id)
tokenizer = BertTokenizer.from_pretrained(model_id, return_tensors="pt")

# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path("bert-base-uncased.onnx")
)

print(f'inputs: {onnx_inputs}')
print(f'outputs: {onnx_outputs}')
