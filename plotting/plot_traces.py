import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

END_TIME = 1441
END_TIME = 300


# logfile_list = glob.glob('../traces/azure/mlsys/250/*.txt')
logfile_list = glob.glob('../../datasets/infaas/asplos/trace_files/*.txt')

# # We want the latest log file
# logfile = logfile_list[-1]
print(logfile_list)

markers = ['o', 'v', '^', '*', 's']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C']

total_requests = np.zeros(END_TIME)

for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    
    model_family = logfile.split('/')[-1].split('.')[0]

    df = pd.read_csv(logfile, names=['time', 'accuracy', 'deadline'])

    print(f'df: {df}')
    aggregated = df.groupby(df.time // 1000).count()
    print(f'aggregated: {aggregated}')
    df = aggregated
    
    requests = df.time.to_list()
    if len(requests) < END_TIME:
        for i in range(END_TIME-len(requests)):
            requests.append(0)
    total_requests += requests
    # print(f'df.time: {len(df.time.to_list())}')

    start_cutoff = 0

    time = df['time'].values[start_cutoff:]    

    # time = time
    # time = [x - time[0] for x in time]
    # print(time[-1])
    # time = [x / time[-1] * 24 for x in time]
    # print(time[0])
    # print(time[-1])

    plt.plot(time, label=model_family)
    # plt.plot(time, demand, label='Demand')
    # plt.plot(time, throughput, label=algorithm, marker=markers[idx])
    # plt.plot(time, throughput, label=algorithm)
    # plt.plot(time, throughput, color=colors[idx])
# plt.plot(time, capacity, label='capacity')
plt.grid()

# plt.rcParams.update({'font.size': 30})
# plt.rc('axes', titlesize=30)     # fontsize of the axes title
# plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
# plt.rc('legend', fontsize=30)    # legend

# y_cutoff = max(demand) + 50
# # y_cutoff = 200

print(f'total requests: {total_requests}')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=10)
plt.xlabel('Time (min)', fontsize=25)
plt.ylabel('Requests per second', fontsize=25)
# plt.xticks(np.arange(0, 25, 4), fontsize=15)
# plt.yticks(np.arange(0, y_cutoff, 200), fontsize=15)
plt.savefig(os.path.join('../../datasets/infaas/asplos/trace_files/traces_by_model.pdf'), dpi=500, bbox_inches='tight')

plt.close()
plt.plot(total_requests)
plt.savefig(os.path.join('../../datasets/infaas/asplos/trace_files/traces_total.pdf'), dpi=500, bbox_inches='tight')
