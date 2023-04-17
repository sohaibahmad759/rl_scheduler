import pprint
import pandas as pd


df = pd.read_csv('aggregate_profiled.csv')

pairs = set()

# For each model variant, we want the runtime of the fastest CPU variant
runtimes = {}

for index, row in df.iterrows():
    # print(row['Model'], row['Accel'])
    pair = (row['Model'], row['Accel'])

    latency = row['50th_pct']

    if 'cpu' in row['Accel']:
        pairs.add(pair)

        if pair not in runtimes:
            runtimes[pair] = latency
        elif latency < runtimes[pair]:
            runtimes[pair] = latency

for pair in runtimes:
    runtimes[pair] *= 3
pprint.pprint(runtimes)
