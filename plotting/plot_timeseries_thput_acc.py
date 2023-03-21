import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'infaas_3.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'accscale.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'infaas_unit.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'clipper_high_throughput.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'clipper_high_accuracy.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'accscale.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'clipper_highacc.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'clipper_lowacc.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'infaas_unit.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', '*.csv')))

logfile_list = [
                '../logs/throughput/selected_asplos/infaas_accuracy_300ms.csv',
                '../logs/throughput/selected_asplos/clipper_ht_300ms.csv',
                # '../logs/throughput/selected_asplos/clipper_optstart_300ms.csv',
                '../logs/throughput/selected_asplos/sommelier_aimd_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_asb_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_nexus_300ms.csv',
                '../logs/throughput/selected_asplos/proteus_300ms.csv',
                # '../logs/throughput/selected_asplos/clipper_ht_asb_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_uniform_asb_300ms.csv'
                ]

MARKERS_ON = False

# We want to print the latest log file
logfile = logfile_list[-1]
print(logfile)

markers = ['+', 'o', 'v', '^', '*', 's', 'x']
# algorithms = ['AccScale', 'Clipper++ (High Accuracy)', 'Clipper++ (High Throughput)',
#             'INFaaS-Accuracy', 'INFaaS-Instance']
# algorithms = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
#             'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
algorithms = [
              'INFaaS-Accuracy',
              'Clipper-HT-AIMD',
            #   'Clipper-HT Optimized Start',
              'Sommelier-AIMD',
            #   'Sommelier-ASB',
            #   'Sommelier-Nexus'
              'Proteus'
            #   'Clipper-HT-ASB',
            #   'Sommelier-ASB (Uniform Start)'
              ]
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2']

fig, (ax1, ax2) = plt.subplots(2)
color_idx = 0
clipper_accuracy = []
for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    
    algorithm = logfile.split('/')[-1].rstrip('.csv')

    df = pd.read_csv(logfile)

    aggregated = df.groupby(df.index // 10).sum()
    aggregated = df.groupby(df.index // 5).mean()
    # print(f'df: {df}')
    # print(f'aggregated: {aggregated}')
    df = aggregated

    start_cutoff = 0

    # time = df['wallclock_time'].values[start_cutoff:]
    time = df['simulation_time'].values[start_cutoff:]
    demand = df['demand'].values[start_cutoff:]
    throughput = df['throughput'].values[start_cutoff:]
    capacity = df['capacity'].values[start_cutoff:]

    effective_accuracy = df['effective_accuracy'].values[start_cutoff:]
    # print(f'effective accuracy: {effective_accuracy}')

    if 'clipper' in algorithm:
        clipper_accuracy = effective_accuracy

    if len(clipper_accuracy) > 0:
        for i in range(len(effective_accuracy)):
            if effective_accuracy[i] < clipper_accuracy[i]:
                effective_accuracy[i] = clipper_accuracy[i]

    successful = df['successful'].values[start_cutoff:]

    time = time
    time = [x - time[0] for x in time]
    print(time[-1])
    time = [x / time[-1] * 24 for x in time]
    print(time[0])
    print(time[-1])

    if idx == 0:
        ax1.plot(time, demand, label='Demand', color=colors[color_idx],
                 marker=markers[color_idx])
        # ax1.plot(time, demand, label='Demand', marker=markers[color_idx])
        color_idx += 1
    # plt.plot(time, throughput, label=algorithm, marker=markers[idx])
    # plt.plot(time, throughput, label=algorithm)
    if MARKERS_ON == True:
        ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx])
        ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx])
    else:
        ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx])
        ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx])
        # ax1.plot(time, successful, label=algorithms[idx])
        # ax2.plot(time, effective_accuracy, label=algorithms[idx])
    color_idx += 1
# plt.plot(time, capacity, label='capacity')
ax1.grid()
ax2.grid()

# plt.rcParams.update({'font.size': 30})
# plt.rc('axes', titlesize=30)     # fontsize of the axes title
# plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
# plt.rc('legend', fontsize=30)    # legend

y_cutoff = max(demand) + 50
# y_cutoff = 200

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=3, fontsize=12)
ax1.set_ylabel('Requests per sec', fontsize=15)
ax1.set_xticks(np.arange(0, 25, 4), fontsize=15)
ax1.set_yticks(np.arange(0, y_cutoff, 200), fontsize=15)

ax2.set_xticks(np.arange(0, 25, 4), fontsize=15)
ax2.set_yticks(np.arange(80, 104, 5))
ax2.set_ylabel('Effective Accuracy', fontsize=15)

ax2.set_xlabel('Time (min)', fontsize=15)

plt.savefig(os.path.join('..', 'figures', 'timeseries_thput_accuracy.pdf'), dpi=500, bbox_inches='tight')

print(f'Warning! We should not be using mean to aggregate, instead we should be using sum')
print(f'Warning! There are some points where requests served are greater than incoming '
      f'demand. Fix this or find the cause')

# accuracy_logfile = os.path.join('..', 'logs', 'log_ilp_accuracy.txt')
# accuracy_rf = open(accuracy_logfile, mode='r')
# accuracies = accuracy_rf.readlines()
# accuracies = [float(x.rstrip('\n')) for x in accuracies]
# print('Sum of accuracies:', sum(accuracies))
# print('Requests served:', len(accuracies))
# print('Effective accuracy:', sum(accuracies)/len(accuracies))
