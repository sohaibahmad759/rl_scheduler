import os
import random
import numpy as np

# TODO: parameterize this file argparse
# TODO: path should not be fixed
# TODO: comma-separated list of QoS values [3,4,5,7,7,7,2]. select from this list
trace_path = 'traces/synthetic/newly_generated'
# trace_length = 5000000000
trace_length = 5000
n_qos_levels = 2
# models = ['resnet-50', 'resnet-10', 'ssd', 'alexnet', 'mobilenet']
models = list(range(5))

for model in models:
    # lam = random.randint(5, 10)
    # lam = random.random()/100

    # 1/100 means 1 request every 100 milliseconds (10 requests per second)
    lam = 1/10
    trace = np.random.poisson(lam=lam, size=trace_length)
    print(trace)
    filename = str(model) + '.txt'
    print(filename)
    with open(os.path.join(trace_path, filename), mode='w') as trace_file:
        idx = 0
        for num_requests in trace:
            for i in range(num_requests):
                trace_file.write(str(idx) + ',' + str(random.randint(0, n_qos_levels-1)) + '\n')
            idx += 1
