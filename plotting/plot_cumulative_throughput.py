import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = sorted(glob.glob(os.path.join('..', 'logs', 'temp', '*.csv')))
# print(logfile_list)
# We want the latest log file
# logfile = logfile_list[-1]
for logfile in logfile_list:
    print(logfile)
    df = pd.read_csv(logfile)

    if 'ilp' in logfile:
        label = ' (Our system)'
    elif 'infaas' in logfile:
        label = ' (INFaaS)'
    else:
        print('Error! Neither ilp nor INFaaS')
        sys.exit(0)

    time = df['wallclock_time']
    demand = df['demand']
    throughput = df['throughput']
    capacity = df['capacity']

    # print(time)
    first_elem = time[0]
    time = [x - first_elem for x in time]
    print(time)

    # if 'ilp' in logfile:
    plt.plot(time, demand, label='Demand')
    plt.plot(time, throughput, label='Throughput' + label)
    plt.xlabel('Time')
    # plt.xticks([])
    plt.ylabel('Requests per second')
    # plt.plot(time, capacity, label='capacity')
    plt.legend()

    if 'ilp' in logfile:
        plt.savefig(os.path.join('..', 'figures', 'paper_draft_figures', 'ilp_throughput_alpha0.png'))
    elif 'infaas' in logfile:
        plt.savefig(os.path.join('..', 'figures', 'paper_draft_figures', 'infaas_throughput.png'))
    plt.close()
