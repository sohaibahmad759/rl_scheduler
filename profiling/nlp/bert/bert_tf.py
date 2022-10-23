import time
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

models = {'bert-tiny': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2',
          'bert-mini': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2',
          'bert-small': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2',
          'bert-medium': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2',
          'bert-base': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2'}
batch_sizes = [1, 2, 4, 8, 16]
batch_sizes = [1]
num_models = len(models) * len(batch_sizes)

idx = 0
trials = 100

with tf.device('/device:GPU:0'):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)

    for model_name in models:
        for batch_size in batch_sizes:
            idx += 1
            url = models[model_name]

            encoder = hub.KerasLayer(url, trainable=True)

            latencies = []
            for i in range(trials):
                start_time = time.time()
                outputs = encoder(encoder_inputs)
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
                # print(f'output: {output}')
        fifty_pct = sorted(latencies)[int(len(latencies)/2)]
        ninety_pct = sorted(latencies)[int(len(latencies)/10*9)]
        avg = sum(latencies)/len(latencies)
        minimum = min(latencies)
        maximum = max(latencies)
        print()
        print('------------------------------------------')
        print(f'{idx}/{num_models}, Model: {model_name}, batch size: {batch_size}')
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

    pooled_output = outputs["pooled_output"]      # [batch_size, 128].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 128].

    print(outputs)
