from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

args = TensorFlowBenchmarkArguments(
    models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
)
benchmark = TensorFlowBenchmark(args)

results = benchmark.run()
print(results)
