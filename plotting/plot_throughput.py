import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'throughput', '*.csv')))
print(logfile_list)
# We want the latest log file
logfile = logfile_list[-1]
print(logfile)
df = pd.read_csv(logfile)

time = df['wallclock_time']
demand = df['demand']
throughput = df['throughput']

plt.plot(time, demand, label='demand')
plt.plot(time, throughput, label='throughput')
plt.legend()
plt.savefig(os.path.join('..', 'figures', 'throughput.png'))
