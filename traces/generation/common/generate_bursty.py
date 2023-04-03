import numpy as np


input_trace = 'twitter_04_14_norm'
output_trace = f'{input_trace}_bursty'

requests = []
counter = 0
include = False
with open(f'{output_trace}.txt', mode='w') as wf:
    with open(f'{input_trace}.txt', mode='r') as rf:
        for line in rf:
            timestep = int(line.split()[0])
            if include:
                # timestep_requests = int(line.split()[1].rstrip('\n'))
                timestep_requests = 700 + int(np.random.rand() * 50)
            else:
                timestep_requests = 100

            requests.append(timestep_requests)
            wf.write(f'{timestep} {timestep_requests}\n')

            counter += 1
            if counter % 47 == 0:
                # reverse every 20 steps
                include = not(include)

print(requests)
