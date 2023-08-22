import os
import pprint
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../profiling/aggregate_profiled.csv')
accuracy_df = pd.read_csv('../profiling/accuracy.csv')

pairs = set()

# For each model variant, we want the runtime of the fastest CPU variant
runtimes = {}
accuracies = {}
variants = set()
accelerators = set()

for index, row in df.iterrows():
    # print(row['Model'], row['Accel'])
    pair = (row['Model'], row['Accel'])

    latency = row['50th_pct']

    if row['batchsize'] == 1 and 'yolo' in row['Model']:
        variants.add(row['Model'])
        accelerators.add(row['Accel'])
        pairs.add(pair)

        if pair not in runtimes:
            runtimes[pair] = latency
            accuracies[pair] = row['Normalized_Accuracy']
        elif latency < runtimes[pair]:
            print('oops')
            runtimes[pair] = latency
            accuracies[pair] = row['Normalized_Accuracy']

# for pair in runtimes:
#     runtimes[pair] *= 2.1

variants = sorted(list(variants))
accelerators = sorted(list(accelerators))
runtimes_2d = np.zeros((len(variants), len(accelerators)))
accuracies_2d = np.zeros((len(variants), len(accelerators)))


for i in range(len(variants)):
    for j in range(len(accelerators)):
        variant = variants[i]
        accelerator = accelerators[j]
        runtime = runtimes[(variant, accelerator)]
        accuracy = accuracies[(variant, accelerator)]

        runtimes_2d[i, j] = runtime
        accuracies_2d[i, j] = accuracy


# accuracies = [[98.89, 98.89, 98.89],
#               [100, 100, 100],
#               [89.53, 89.53, 89.53],
#               [92.61, 92.61, 92.61],
#               [97.65, 97.65, 97.65],]
# 53.05039788
# 45.78754579
# 33.89830508
# 22.97266253
# 16.28134158
runtimes_2d = np.array([[55.58643691, 53.05039788, 15.15840534],
                        [52.63157895, 45.78754579, 13.33333333],
                        [47.55111745, 33.89830508, 10],
                        [32.55208333, 22.97266253, 5],
                        [25.23340903, 16.28134158, 1.357625784]])

accuracies_2d = np.array([[77.1, 77.1, 77.1],
              [79.1, 79.1, 79.1],
              [81.6, 81.6, 81.6],
              [83.6, 83.6, 83.6],
              [84.3, 84.3, 84.3],])

v100_thputs = np.array([55.58643691, 52.63157895, 47.55111745, 32.55208333, 25.23340903])
_1080ti_thputs = np.array([53.05039788, 45.78754579, 33.89830508, 22.97266253, 16.28134158])
cpu_thputs = np.array([15.15840534, 13.33333333, 10, 5, 1.357625784])
accuracies = np.array(['77.1', '79.1', '81.6', '83.6', '84.3'])

df = pd.DataFrame(np.zeros((1, 4)), columns=['Device', 'Throughput', 'Accuracy', 'Color'])
idx = 0
for throughput in v100_thputs:
    df = df.append({'Device': 'V100', 'Throughput': throughput, 'Accuracy': accuracies[idx],
                    'Color': 'tab:blue'},
                   ignore_index=True)
    idx += 1

idx = 0
for throughput in _1080ti_thputs:
    df = df.append({'Device': '1080 Ti', 'Throughput': throughput, 'Accuracy': accuracies[idx],
                    'Color': 'tab:orange'},
                   ignore_index=True)
    idx += 1

idx = 0
for throughput in cpu_thputs:
    df = df.append({'Device': 'CPU', 'Throughput': throughput, 'Accuracy': accuracies[idx],
                    'Color': 'tab:green'},
                   ignore_index=True)
    idx += 1

df = df[df['Throughput'] != 0]

print(df)

print(f'accuracies_2d: {accuracies_2d}')

color_2d = np.array([['tab:blue', 'tab:orange', 'tab:green'],
                     ['tab:blue', 'tab:orange', 'tab:green'],
                     ['tab:blue', 'tab:orange', 'tab:green'],
                     ['tab:blue', 'tab:orange', 'tab:green'],
                     ['tab:blue', 'tab:orange', 'tab:green']])

labels_2d = np.array([['V100', '1080 Ti', 'CPU'],
                      ['V100', '1080 Ti', 'CPU'],
                      ['V100', '1080 Ti', 'CPU'],
                      ['V100', '1080 Ti', 'CPU'],
                      ['V100', '1080 Ti', 'CPU']])

# colors = ['black', 'red', 'blue']
# color_2d = np.empty((len(variants), 3), dtype=np.str)
# print(f'shape: {color_2d.shape}')

# for i in range(color_2d.shape[0]):
#     color_2d[i, 0] = 'g'
#     color_2d[i, 1] = 'red'
#     color_2d[i, 2] = 'b'

print(f'eh: {np.ravel(np.transpose(runtimes_2d))}')
_dict = {'Peak Throughput Capacity': np.ravel(np.transpose(runtimes_2d)),
        'Accuracy (%)': np.ravel(np.transpose(accuracies_2d)),
        'labels': np.ravel(np.transpose(labels_2d))}
print(_dict)
new_df = pd.DataFrame(_dict)

# print(variants)
# pprint.pprint(runtimes)
# print(runtimes_2d)
# print(color_2d)
# print(runtimes.keys())
# print(runtimes.values())
# for i in range(accuracies_2d.shape[0]):
#     for j in range(runtimes_2d.shape[0]):
#         plt.scatter(accuracies_2d[i], runtime)
# matplotlib.rcParams.update({'font.size': 18})
# plt.scatter(y=np.transpose(accuracies_2d), x=np.transpose(runtimes_2d),
#             c=np.ravel(np.transpose(color_2d)), label=np.ravel(labels_2d),
#             s=100)

# fg = sns.FacetGrid(data=new_df, hue='labels')
# # fg = sns.FacetGrid(data=new_df)
# # fg.map(plt.scatter, 'Throughput', 'Accuracy').add_legend()
# fg.map(plt.scatter, 'Peak Throughput Capacity', 'Accuracy (%)')

plt.scatter(y=np.ravel(np.transpose(accuracies_2d)), x=np.ravel(np.transpose(runtimes_2d)),
            c=np.ravel(np.transpose(color_2d)), label=np.ravel(labels_2d),
            s=100)

plt.xlabel('Peak Throughput Capacity (QPS)')
# plt.xticks(np.arange(0, 61, 10))
plt.ylabel('Accuracy (%)')
# plt.yticks(np.arange(75, 86, 2))
plt.grid()
# plt.legend()
# plt.show()
plt.legend(loc='upper center', bbox_to_anchor=(0.85, 1), ncol=1, fontsize=8)
plt.savefig(os.path.join('..', 'figures', 'asplos', 'configurations', 'profiled.pdf'),
            dpi=500, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(4, 4))
plt.xlabel('System Throughput Capacity (QPS)', fontsize=14)
plt.ylabel('System Accuracy (%)', fontsize=14)
plt.grid()

v100_df = df[df['Device'] == 'V100']
plt.scatter(x=v100_df['Throughput'], y=v100_df['Accuracy'].astype(float),
            color='tab:blue', label='V100', s=100)

_1080ti_df = df[df['Device'] == '1080 Ti']
plt.scatter(x=_1080ti_df['Throughput'], y=_1080ti_df['Accuracy'].astype(float),
            color='tab:orange', label='1080 Ti', s=100)

cpu_df = df[df['Device'] == 'CPU']
plt.scatter(x=cpu_df['Throughput'], y=cpu_df['Accuracy'].astype(float),
            color='tab:green', label='CPU', s=100)

plt.xticks(np.arange(0, 61, 10), fontsize=12)
plt.yticks(np.arange(76, 86, 1), fontsize=12)

plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1), ncol=1, fontsize=12)
# plt.legend(fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join('..', 'figures', 'asplos', 'configurations', 'profiled.pdf'),
            dpi=500, bbox_inches='tight')
