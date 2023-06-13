import os
import pprint
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('aggregate_profiled.csv')
accuracy_df = pd.read_csv('accuracy.csv')

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
dict = {'Peak Throughput Capacity': np.ravel(np.transpose(runtimes_2d)),
        'Accuracy (%)': np.ravel(np.transpose(accuracies_2d)),
        'labels': np.ravel(np.transpose(labels_2d))}
print(dict)
new_df = pd.DataFrame(dict)

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
fg = sns.FacetGrid(data=new_df, hue='labels')
# fg = sns.FacetGrid(data=new_df)
# fg.map(plt.scatter, 'Throughput', 'Accuracy').add_legend()
fg.map(plt.scatter, 'Peak Throughput Capacity', 'Accuracy (%)')

plt.xlabel('Peak Throughput Capacity (QPS)')
# plt.xticks(np.arange(0, 61, 10))
plt.ylabel('Accuracy (%)')
# plt.yticks(np.arange(75, 86, 2))
plt.grid()
# plt.legend()
# plt.show()
plt.legend(loc='upper center', bbox_to_anchor=(0.85, 1), ncol=1, fontsize=8)
plt.savefig(os.path.join('..', 'figures', 'asplos', 'profiled.pdf'),
            dpi=500, bbox_inches='tight')



# resnet101_v1	78.34	ImageNet top-1			98.89			
# resnet152_v1	79.22	ImageNet top-1			100			
# resnet18_v1	70.93	ImageNet top-1				89.53		
# resnet34_v1	74.37	ImageNet top-1				92.61		
# resnet50_v1	77.36	ImageNet top-1				97.65		