import os
import random
import numpy as np


trace_path = '/home/soahmad/blis/evaluation/scheduler/traces/synthetic'
trace_length = 50000000
# models = ['resnet-50', 'resnet-10', 'ssd', 'alexnet', 'mobilenet']
models = list(range(100))

for model in models:
    # lam = random.randint(5, 10)
    # lam = random.random()/100

    # 1/100 means 1 request every 100 milliseconds (10 requests per second)
    lam = 1/100
    trace = np.random.poisson(lam=lam, size=trace_length)
    filename = str(model) + '.txt'
    print(filename)
    with open(os.path.join(trace_path, filename), mode='w') as trace_file:
        idx = 0
        for num_requests in trace:
            for i in range(num_requests):
                trace_file.write(str(idx) + '\n')
            idx += 1
