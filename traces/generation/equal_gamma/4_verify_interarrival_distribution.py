import glob
import numpy as np
import matplotlib.pyplot as plt


files = glob.glob('trace_files/*.txt')

for file in files:
    with open(file, mode='r') as rf:
        lines = rf.readlines()
        arrivals = list(map(lambda x: int(x.split(',')[0]), lines))

        inter_arrival_times = np.diff(arrivals)
        # print(f'90th percentile: {np.percentile(inter_arrival_times, 95)}')
        # interval_arrival_times = inter_arrival_times[inter_arrival_times > np.percentile(inter_arrival_times,90)]
        _95pct = np.percentile(inter_arrival_times,95)
        inter_arrival_times = [ele for ele in inter_arrival_times if ele <= _95pct]
        print(inter_arrival_times)
        # print(f'maximum: {max(inter_arrival_times)}')
        # print(np.median(inter_arrival_times))

        # inter_arrival_times = np.random.exponential(101.123, 10000)

        count, bins, ignored = plt.hist(inter_arrival_times)
        plt.savefig(file + '.pdf')
        plt.close()
