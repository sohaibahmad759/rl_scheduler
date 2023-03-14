import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'infaas_3.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'accscale.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'infaas_unit.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'clipper_high_throughput.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'clipper_high_accuracy.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'accscale.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'clipper_highacc.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'clipper_lowacc.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'infaas_unit.csv')))
logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', '*.csv')))
# logfile_list = ['../logs/throughput/selected/bursty/clipper_lowacc.csv',
#                 '../logs/throughput/selected/bursty/clipper_highacc.csv',
#                 '../logs/throughput/selected/bursty/infaas_unit.csv',
#                 '../logs/throughput/selected/bursty/infaas_accuracy.csv',
#                 '../logs/throughput/selected/bursty/accscale.csv']
# logfile_list = ['../logs/throughput/selected/clipper_high_throughput.csv',
#                 '../logs/throughput/selected/clipper_high_accuracy.csv',
#                 '../logs/throughput/selected/infaas_unit.csv',
#                 '../logs/throughput/selected/infaas_accuracy.csv',
#                 '../logs/throughput/selected/accscale.csv']
logfile_list = ['../logs/throughput/selected_asplos/proteus_300ms.csv']
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'bursty', 'infaas_accuracy.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'paper_jun13_onwards', 'infaas', '3', '*.csv')))
# print(logfile_list)
# We want the latest log file
logfile = logfile_list[-1]
print(logfile)

markers = ['o', 'v', '^', '*', 's']
# algorithms = ['AccScale', 'Clipper++ (High Accuracy)', 'Clipper++ (High Throughput)',
#             'INFaaS-Accuracy', 'INFaaS-Instance']
# algorithms = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
#             'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
algorithms = ['Proteus']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C']

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


    # plt.plot(time, demand, label='Demand')
    # plt.plot(time, throughput, label=algorithm, marker=markers[idx])
    # plt.plot(time, throughput, label=algorithm)
    plt.plot(time, throughput, label=algorithms[idx], color=colors[idx])
# plt.plot(time, capacity, label='capacity')
plt.grid()

# plt.rcParams.update({'font.size': 30})
# plt.rc('axes', titlesize=30)     # fontsize of the axes title
# plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
# plt.rc('legend', fontsize=30)    # legend

y_cutoff = max(demand) + 50
# y_cutoff = 200

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2, fontsize=15)
plt.xlabel('Time (min)', fontsize=25)
plt.ylabel('Requests per second', fontsize=25)
plt.xticks(np.arange(0, 25, 4), fontsize=15)
plt.yticks(np.arange(0, y_cutoff, 100), fontsize=15)
plt.savefig(os.path.join('..', 'figures', 'throughput.pdf'), dpi=500, bbox_inches='tight')

# accuracy_logfile = os.path.join('..', 'logs', 'log_ilp_accuracy.txt')
# accuracy_rf = open(accuracy_logfile, mode='r')
# accuracies = accuracy_rf.readlines()
# accuracies = [float(x.rstrip('\n')) for x in accuracies]
# print('Sum of accuracies:', sum(accuracies))
# print('Requests served:', len(accuracies))
# print('Effective accuracy:', sum(accuracies)/len(accuracies))
