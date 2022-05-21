import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', '*.csv')))
# print(logfile_list)
# We want the latest log file
logfile = logfile_list[-1]
print(logfile)
df = pd.read_csv(logfile)

time = df['wallclock_time']
demand = df['demand']
throughput = df['throughput']
capacity = df['capacity']

plt.plot(time, demand, label='demand')
plt.plot(time, throughput, label='throughput')
plt.plot(time, capacity, label='capacity')
plt.legend()
plt.savefig(os.path.join('..', 'figures', 'throughput.png'))
