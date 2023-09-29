import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

np_df = np.zeros((1, 7))

variants = ['EfficientNet-b0', 'EfficientNet-b1', 'EfficientNet-b3',
            'EfficientNet-b5', 'EfficientNet-b7']

count = 0
for v1 in variants:
    for v2 in variants:
        for v3 in variants:
            for v4 in variants:
                for v5 in variants:
                    # print(f'{v1}, {v2}, {v3}')
                    np_df = np.vstack((np_df, np.array([v1, v2, v3, v4, v5, 0, 0])))
                    count += 1
print(f'\n\ncount: {count}\n\n')

np_df = np_df[1:]
df = pd.DataFrame(data=np_df, columns=['Device 1', 'Device 2', 'Device 3', 'Device 4',
                                       'Device 5',
                                       'Throughput', 'Accuracy'])
print(df)

throughputs = {('EfficientNet-b0', 'Device 1'): 15.15840534,
               ('EfficientNet-b0', 'Device 2'): 53.05039788,
               ('EfficientNet-b0', 'Device 3'): 55.58643691,
               ('EfficientNet-b0', 'Device 4'): 15.15840534,
               ('EfficientNet-b0', 'Device 5'): 15.15840534,

               ('EfficientNet-b1', 'Device 1'): 13.33333333,
               ('EfficientNet-b1', 'Device 2'): 45.78754579,
               ('EfficientNet-b1', 'Device 3'): 52.63157895,
               ('EfficientNet-b1', 'Device 4'): 13.33333333,
               ('EfficientNet-b1', 'Device 5'): 13.33333333,

               ('EfficientNet-b3', 'Device 1'): 10,
               ('EfficientNet-b3', 'Device 2'): 33.89830508,
               ('EfficientNet-b3', 'Device 3'): 47.55111745,
               ('EfficientNet-b3', 'Device 4'): 10,
               ('EfficientNet-b3', 'Device 5'): 10,

               ('EfficientNet-b5', 'Device 1'): 5,
               ('EfficientNet-b5', 'Device 2'): 22.97266253,
               ('EfficientNet-b5', 'Device 3'): 32.55208333,
               ('EfficientNet-b5', 'Device 4'): 5,
               ('EfficientNet-b5', 'Device 5'): 5,

               ('EfficientNet-b7', 'Device 1'): 1.357625784,
               ('EfficientNet-b7', 'Device 2'): 16.28134158,
               ('EfficientNet-b7', 'Device 3'): 25.23340903,
               ('EfficientNet-b7', 'Device 4'): 1.357625784,
               ('EfficientNet-b7', 'Device 5'): 1.357625784}

accuracies = {'EfficientNet-b0': 77.1,
              'EfficientNet-b1': 79.1,
              'EfficientNet-b3': 81.6,
              'EfficientNet-b5': 83.6,
              'EfficientNet-b7': 84.3}

devices = ['Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5']
combinations = 0
for i in range(10):
    for j in range(10):
        for k in range(10):
            if i + j + k > 10 or i + j + k < 10:
                continue

            # device1 = devices[1]
            # device2 = devices[2]
            # device3 = devices[3]
            # configuration = []

            combinations += 1

# print(combinations)
# exit()

# configurations = ?

# for pair in throughputs:
#     for model in accuracies:
#         (_model, device) = pair

# for i in range(10):
#     for j in range(10):
#         for k in range(10):


for index, row in df.iterrows():
    system_throughput = 0
    for device in devices:
        system_throughput += throughputs[(row[device], device)]
    
    total_accuracy = 0
    for device in devices:
        total_accuracy += accuracies[row[device]]*throughputs[(row[device], device)]

    system_effective_accuracy = total_accuracy / system_throughput

    df.at[index, 'Throughput'] = system_throughput
    df.at[index, 'Accuracy'] = system_effective_accuracy
    
#     eb0s = 0
#     eb3s = 0
#     eb7s = 0
    
#     if row['Device 1'] == 'EfficientNet-b0':
#         eb0s += 1
#     elif row['Device 1'] == 'EfficientNet-b3':
#         eb3s += 1
#     else:
#         eb7s += 1

#     if row['Device 2'] == 'EfficientNet-b0':
#         eb0s += 1
#     elif row['Device 2'] == 'EfficientNet-b3':
#         eb3s += 1
#     else:
#         eb7s += 1

#     if row['Device 3'] == 'EfficientNet-b0':
#         eb0s += 1
#     elif row['Device 3'] == 'EfficientNet-b3':
#         eb3s += 1
#     else:
#         eb7s += 1

#     df.at[index, 'configuration'] = f'{eb0s} eb0, {eb3s} eb3, {eb7s} eb7'


# df = df.replace('EfficientNet-b0', 'eb0')
# df = df.replace('EfficientNet-b7', 'eb7')
# df = df.round(2)

print(df)

# hide axes
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')

# the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', fontsize=20)
# # the_table = axs.table(cellText=data, colLabels=columns, loc='center')
# the_table.auto_set_font_size(False)
# the_table.set_fontsize(12)

# df.plot.scatter(x='Throughput',
#                 y='Accuracy',
#                 c='configuration')

# # fg = sns.FacetGrid(data=df, hue='configurations', hue_order=_genders, aspect=1.61)
# fg = sns.FacetGrid(data=df, hue='configuration')
# # fg.map(plt.scatter, 'Throughput', 'Accuracy').add_legend()
# fg.map(plt.scatter, 'Throughput', 'Accuracy')

fig = plt.figure(figsize=(4, 4))
plt.scatter(x=df['Throughput'], y=df['Accuracy'], facecolors='None', edgecolors='gray')


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.75, 1), ncol=1, fontsize=8)
plt.xlabel('System Throughput Capacity (QPS)', fontsize=14)
plt.ylabel('System Accuracy (%)', fontsize=14)

def find_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    '''Plotting process'''
    # plt.scatter(Xs,Ys)
    # pf_X = [pair[0] for pair in pareto_front]
    # pf_Y = [pair[1] for pair in pareto_front]
    # plt.plot(pf_X, pf_Y)
    # plt.xlabel("Objective 1")
    # plt.ylabel("Objective 2")
    # plt.show()
    return pareto_front

pareto_frontier = find_pareto_frontier(df['Throughput'], df['Accuracy'])
pareto_frontier = np.array(pareto_frontier)
# print(f'pareto_frontier: {pareto_frontier}')
plt.scatter(x=pareto_frontier[:, 0], y=pareto_frontier[:, 1], facecolors='r',
            edgecolors='r', label='Pareto Frontier')


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.xticks(np.arange(40, 180, 20), fontsize=12)
plt.yticks(np.arange(76, 86, 1), fontsize=12)

# ax.scatter(x=df['Throughput'], y=df['Accuracy'])

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# plt.show()
plt.legend(loc='upper center', bbox_to_anchor=(0.63, 1), fontsize=14)
fig.tight_layout()
plt.savefig('../figures/asplos/configurations/configurations.pdf')
