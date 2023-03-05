import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = ['../logs/throughput/selected/bursty/clipper_lowacc.csv',
                '../logs/throughput/selected/bursty/clipper_highacc.csv',
                '../logs/throughput/selected/bursty/infaas_unit.csv',
                '../logs/throughput/selected/bursty/infaas_accuracy.csv',
                '../logs/throughput/selected/bursty/accscale.csv']

bursty_accuracies = [70.28495378, 79.67558044, 73.31185987, 77.64658593, 77.87]

labels = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
        'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C']
hatch = ['/', '\\', '|', 'xx', '+']

fig, axs = plt.subplots(1, 1)

for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    algorithm = logfile.split('/')[-1].rstrip('.csv')
    df = pd.read_csv(logfile)
    start_cutoff = 0

    time = df['wallclock_time'].values[start_cutoff:]
    demand = df['demand'].values[start_cutoff:]
    throughput = df['throughput'].values[start_cutoff:]
    capacity = df['capacity'].values[start_cutoff:]

    time = time
    time = [x - time[0] for x in time]
    print(time[-1])
    time = [x / time[-1] * 24 for x in time]
    print(time[0])
    print(time[-1])

    plt.plot(time, throughput, label=labels[idx], color=colors[idx])
plt.grid()
y_cutoff = max(demand) + 50
y_cutoff = 200

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fontsize=15)
plt.xlabel('Time (min)', fontsize=20)
plt.ylabel('Requests per second', fontsize=20)
plt.xticks(np.arange(0, 25, 4), fontsize=15)
plt.yticks(np.arange(0, y_cutoff, 20), fontsize=15)
# plt.savefig(os.path.join('..', 'figures', 'throughput.pdf'), dpi=500, bbox_inches='tight')

# # fig, ax = plt.subplots()
# axs[1].grid(linestyle='--', axis='y')
# axs[1].bar([1, 2, 3, 4, 5], bursty_accuracies, color=colors, hatch=hatch, label=labels)
# axs[1].set_axisbelow(True)
# axs[1].set_xlabel('Strategy', fontsize=15)
# axs[1].set_ylabel('Effective Accuracy (%)', fontsize=15)
# # axs[1].set_xticks(rotation=30)
# # axs[1].set_xticks([1, 2, 3, 4, 5], fontsize=15)
# # axs[1].set_yticks(np.arange(50, 85, 5), fontsize=15)
# axs[1].set_ylim(50, 85)
# axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=2, fontsize=13.5)

# fig.tight_layout(pad=0.5)
plt.savefig('bursty_throughput.pdf', dpi=500, bbox_inches='tight')
