from transformers import AutoModel

# model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
# out = model("Hello I'm a <mask> model.")
# print(f'out: {out}')

from transformers import pipeline
unmasker = pipeline('fill-mask', model='huawei-noah/TinyBERT_General_4L_312D')
out = unmasker("The man worked as a [MASK].")
print(f'out: {out}')
