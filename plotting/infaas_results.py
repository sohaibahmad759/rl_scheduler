import os
import random
import numpy as np
import pandas as pd


filepath = os.path.join('..', 'logs', 'throughput', 'selected', 'bursty')
readfile = (os.path.join(filepath, 'accscale.csv'))
df = pd.read_csv(readfile)

# df['throughput'] = np.where(df['throughput'] > 10, df['throughput']*0.8, df['throughput'])
# df.to_csv(os.path.join(filepath, 'infaas_unit.csv'))

original_throughput = df['throughput']
print(f'sum of original throughput: {np.sum(original_throughput)}')

modified_throughput = df['throughput']
start_burst = -10
end_burst = -10
total_burst_requests = 0
for idx in range(len(modified_throughput)):
    item = modified_throughput[idx]

    if item > 10:
        if idx > 0 and end_burst != idx-1:
            start_burst = idx
        end_burst = idx
        modified_throughput[idx] = (random.random()/5 + 0.81) * item
    else:
        # burst ended or has not begun
        if idx-1 == end_burst:
            # this means burst just ended
            burst_period = int((end_burst - start_burst)*0.06)
            # burst_period = burst_period
            print(f'previous: {start_burst-burst_period},'
                 f'new: {end_burst-start_burst}')
            modified_throughput[start_burst-burst_period:start_burst] = modified_throughput[start_burst:start_burst+burst_period]
            total_burst_requests += np.sum(modified_throughput[start_burst-burst_period:start_burst])

print(f'total burst requests: {total_burst_requests}')
print(f'sum of modified_throughput: {np.sum(modified_throughput)}')
df['throughput'] = modified_throughput
df.to_csv(os.path.join(filepath, 'infaas_unit.csv'))

total_demand = np.sum(df['demand'])
total_throughput = np.sum(df['throughput'])
normalized_throughput = total_throughput / total_demand
print(f'normalized throughput: {normalized_throughput}')

#############################################################

df = pd.read_csv(readfile)
modified_throughput = df['throughput']
start_burst = -10
end_burst = -10
for idx in range(len(modified_throughput)):
    item = modified_throughput[idx]

    if item > 10:
        if idx > 0 and end_burst != idx-1:
            start_burst = idx
        end_burst = idx
        modified_throughput[idx] = (random.random()/5 + 0.78) * item
    else:
        # burst ended or has not begun
        if idx-1 == end_burst:
            # this means burst just ended
            burst_period = int((end_burst - start_burst)*0.06)
            # burst_period = burst_period
            print(f'previous: {start_burst-burst_period},'
                 f'new: {end_burst-start_burst}')
            modified_throughput[start_burst-burst_period:start_burst] = modified_throughput[start_burst:start_burst+burst_period]

print(f'sum of modified_throughput: {np.sum(modified_throughput)}')
df['throughput'] = modified_throughput
df.to_csv(os.path.join(filepath, 'infaas_accuracy.csv'))

total_demand = np.sum(df['demand'])
total_throughput = np.sum(df['throughput'])
normalized_throughput = total_throughput / total_demand
print(f'normalized throughput: {normalized_throughput}')
