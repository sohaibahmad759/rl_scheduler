import glob
import numpy as np
import matplotlib.pyplot as plt

path = '../../twitter/asplos'
trace = 'zipf_flat_uniform'
slo = '300'

files = glob.glob(f'{path}/{trace}/{slo}/*.txt')

overall_interarrival_times = []

for file in files:
    with open(file, mode='r') as rf:
        lines = rf.readlines()
        arrivals = list(map(lambda x: int(x.split(',')[0]), lines))

        inter_arrival_times = np.diff(arrivals)
        # print(f'90th percentile: {np.percentile(inter_arrival_times, 95)}')
        # interval_arrival_times = inter_arrival_times[inter_arrival_times > np.percentile(inter_arrival_times,90)]
        # _95pct = np.percentile(inter_arrival_times,95)
        # inter_arrival_times = [ele for ele in inter_arrival_times if ele <= _95pct]

        print(f'len: {len(inter_arrival_times)}, inter_arrival_times: {inter_arrival_times}')

        overall_interarrival_times.extend(inter_arrival_times)
        # print(inter_arrival_times)
        # print(f'maximum: {max(inter_arrival_times)}')
        # print(np.median(inter_arrival_times))

        # inter_arrival_times = np.random.exponential(101.123, 10000)

        model = file.split("/")[-1].split(".")[0]
        print(f'model: {model}')

        count, bins, ignored = plt.hist(inter_arrival_times)
        plt.savefig(f'{path}/{trace}/{slo}/{model}_interarrivals.pdf')
        plt.close()

print(f'overall len: {len(overall_interarrival_times)}')
count, bins, ignored = plt.hist(overall_interarrival_times)
plt.savefig(f'{path}/{trace}/{slo}/{trace}_interarrivals.pdf')
plt.close()
