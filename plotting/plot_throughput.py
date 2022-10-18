import os
import glob
from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', 'selected', 'infaas_3.csv')))
# logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'paper_jun13_onwards', 'infaas', '3', '*.csv')))
# print(logfile_list)
# We want the latest log file
logfile = logfile_list[-1]
print(logfile)
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


plt.plot(time, demand, label='Demand')
plt.plot(time, throughput, label='Throughput')
# plt.plot(time, capacity, label='capacity')
plt.grid()

# plt.rcParams.update({'font.size': 30})
# plt.rc('axes', titlesize=30)     # fontsize of the axes title
# plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
# plt.rc('legend', fontsize=30)    # legend

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Requests per second', fontsize=14)
plt.xticks(np.arange(0, 25, 2), fontsize=14)
plt.yticks(np.arange(0, 601, 100), fontsize=14)
plt.savefig(os.path.join('..', 'figures', 'throughput.pdf'), dpi=500)

accuracy_logfile = os.path.join('..', 'logs', 'log_ilp_accuracy.txt')
accuracy_rf = open(accuracy_logfile, mode='r')
accuracies = accuracy_rf.readlines()
accuracies = [float(x.rstrip('\n')) for x in accuracies]
print('Sum of accuracies:', sum(accuracies))
print('Requests served:', len(accuracies))
print('Effective accuracy:', sum(accuracies)/len(accuracies))
