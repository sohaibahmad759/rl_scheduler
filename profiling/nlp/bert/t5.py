from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
print(f'loss: {loss}, prediction_scores: {prediction_scores}')

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
outputs = model.generate(input_ids)
print(f'outputs: {outputs}')