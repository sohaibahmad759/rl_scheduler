import torch
import onnxruntime as ort
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", return_tensors="pt")
session = ort.InferenceSession('bert-base-uncased.onnx')

sequences = ['Hello, my dog is cute']
input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences])

_input = tokenizer(sequences[0])
print(f'input: {_input}')

print(f'input_ids: {input_ids}')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

input_dict = {'input_ids': _input["input_ids"],
                            'attention_mask': _input["attention_mask"],
                            # 'token_type_ids': torch.unsqueeze(torch.tensor(_input["token_type_ids"]), 0)}
                            'token_type_ids': to_numpy(torch.tensor(_input["token_type_ids"]))}

print(session.get_inputs()[2])

output = session.run(['logits'], input_dict)

print(f'output: {output}')
