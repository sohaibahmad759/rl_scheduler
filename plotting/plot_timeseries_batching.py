import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# trace = 'normal-high_load'
# trace = 'normal_load'
trace = 'zipf_exponential'
# trace = 'zipf_gamma'
# trace = 'zipf_uniform'
# trace = 'zipf_flat_bursty'
# trace = 'zipf_flat_uniform'
# trace = 'zipf_uniform_random'
# trace = 'equal_exponential'
# trace = 'equal_gamma'

path = '../logs/throughput/selected_asplos'

logfile_list = [
                # f'{path}/{trace}/infaas_accuracy_300ms.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_interval2.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_interval10.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_normalhighload.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack3.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack2.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack1.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack1.5.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack0.15.csv',
                # f'{path}/{trace}/clipper_ht_aimd_300ms.csv', # this
                # f'{path}/{trace}/clipper_ht_nexus_300ms.csv',
                # f'{path}/{trace}/clipper_ht_asb_300ms.csv', # this
                # '../logs/throughput/selected_asplos/clipper_optstart_300ms.csv',
                # f'{path}/{trace}/sommelier_aimd_300ms.csv',
                # f'{path}/{trace}/sommelier_asb_300ms.csv', # this
                # f'{path}/{trace}/sommelier_nexus_300ms.csv',
                # '../logs/throughput/selected_asplos/proteus_aimd_300ms.csv',
                # '../logs/throughput/selected_asplos/proteus_nexus_300ms.csv',
                f'{path}/{trace}/proteus_300ms.csv',
                # f'{path}/{trace}/proteus_batchsize1_300ms.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.15.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.1.csv',
                # f'{path}/{trace}/proteus_aimd_300ms.csv',
                f'{path}/{trace}/proteus_aimd_lateallowed_300ms.csv',
                # f'{path}/{trace}/proteus_aimd_latedropped_300ms.csv',
                # f'{path}/{trace}/proteus_nexus_300ms.csv',
                f'{path}/{trace}/proteus_nexus_lateallowed_300ms.csv',
                # f'{path}/{trace}/proteus_nexus_latedropped_300ms.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.4.csv',
                # f'{path}/{trace}/proteus_300ms_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.4_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta1_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta2_gap1.1.csv',
                # f'{path}/{trace}/proteus_300ms_beta3_gap1.1_edwc.csv',
                # f'{path}/{trace}/proteus_aimd_300ms_beta1.05.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.05.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.05_lateallowed.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.1.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.05_interval2.csv',
                # f'{path}/{trace}/proteus_300ms_beta2_gap1.1_edwc.csv',
                # f'{path}/{trace}/proteus_300ms_beta3_gap1.1_aimd.csv', # this
                # f'{path}/{trace}/proteus_300ms_beta3_gap1.1_nexus.csv', # this
                # f'{path}/{trace}/proteus_300ms_beta2.0_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.05_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.8_15acc_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta2.1_20acc_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.5_20acc_proportional.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.4_coldstart.csv',
                # f'{path}/{trace}/proteus_300ms_accconstraint.csv',
                # f'{path}/{trace}/proteus_300ms_lawc.csv',
                # f'{path}/{trace}/proteus_300ms_edwc.csv',
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
              # 'INFaaS-Accuracy',
            #   'INFaaS-Accuracy Interval 2',
            #   'INFaaS-Accuracy Interval 10',
            #   'INFaaS-Accuracy NormalHighLoad',
            #   'INFaaS-Accuracy (Slack 3)',
            #   'INFaaS-Accuracy (Slack 2)',
            #   'INFaaS-Accuracy (Slack 1)',
            #   'INFaaS-Accuracy (Slack 1.5)',
            #   'INFaaS-Accuracy (Slack 0.15)',
            #   'Clipper-HT-AIMD',
            #   'Clipper-HT-Nexus',
            #   'Clipper-HT-ASB',
            #   'Clipper-HT Optimized Start',
            #   'Sommelier-AIMD',
              # 'Sommelier-ASB',
            #   'Sommelier-Nexus'
            #   'Proteus-Clipper',
            #   'Proteus-Nexus',
              'Proteus',
              # 'Proteus Batch Size 1',
            #   'Proteus (Beta 1.15)',
            #   'Proteus (Beta 1.1)',
            #   'Proteus AIMD',
              'Proteus w/ AIMD Batching',
              # 'Proteus w/ AIMD Batching LateAllowed',
              # 'Proteus w/ AIMD Batching LateDropped',
            #   'Proteus Nexus',
              'Proteus w/ Nexus Batching',
              # 'Proteus w/ Nexus Batching LateAllowed',
              # 'Proteus w/ Nexus Batching LateDropped',
            #   'Proteus (Beta 1.4)',
            #   'Proteus Proportional',
            #   'Proteus Proportional (Beta 1.4)',
            #   'Proteus Proportional (Beta 1)',
            #   'Proteus (Beta 2 Gap 1.1)',
            #   'Proteus (Beta 3 Gap 1.1 EDWC)',
            #   'Proteus (AIMD Beta 1.05)',
            #   'Proteus (Beta 1.05)',
            #   'Proteus (Beta 1.05 Late Allowed)',
            #   'Proteus (Beta 1.1)',
            #   'Proteus (Beta 1.05 Interval 2)',
            #   'Proteus (Beta 2 Gap 1.1 EDWC)',
            #   'Proteus (Beta 3 Gap 1.1 AIMD)',
            #   'Proteus (Beta 3 Gap 1.1 Nexus)',
            #   'Proteus Proportional (Beta 2.0)',
            #   'Proteus Proportional (Beta 1.05)',
            #   'Proteus Proportional (Beta 1.8) 15 Acc',
            #   'Proteus Proportional (Beta 2.1) 20 Acc',
            #   'Proteus Proportional (Beta 1.5) 20 Acc',
            #   'Proteus Proportional (Beta 1.4) Cold Start',
            #   'Proteus (Accuracy Constraint)',
            #   'Proteus (Late Allowed Work Conserving)',
            #   'Proteus (Early Drop Work Conserving)'
            #   'Clipper-HT-ASB',
            #   'Sommelier-ASB (Uniform Start)'
              ]
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2', 'tab:olive', 'tab:cyan']

fig, (ax1, ax2, ax3) = plt.subplots(3)
color_idx = 0
clipper_accuracy = []
y_cutoff = 0
for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    
    algorithm = logfile.split('/')[-1].rstrip('.csv')

    df = pd.read_csv(logfile)

    start_cutoff = 0

    aggregated = df.groupby(df.index // 10).sum()
    aggregated = df.groupby(df.index // 5).mean()
    df = aggregated
    # print(f'df: {df}')
    # print(f'aggregated: {aggregated}')

    if 'demand_ewma' in df:
        demand_ewma = df['demand_ewma'].values[start_cutoff:]
    else:
        demand_ewma = None

    # time = df['wallclock_time'].values[start_cutoff:]
    time = df['simulation_time'].values[start_cutoff:]
    demand = df['demand'].values[start_cutoff:]
    throughput = df['throughput'].values[start_cutoff:]
    capacity = df['capacity'].values[start_cutoff:]

    y_cutoff = max(y_cutoff, max(demand))

    dropped = df['dropped'].values[start_cutoff:]
    late = df['late'].values[start_cutoff:]
    total_slo_violations = dropped + late
    # total_slo_violations = dropped

    successful = df['successful'].values[start_cutoff:]

    # total_slo_violations = total_slo_violations / demand

    effective_accuracy = df['effective_accuracy'].values[start_cutoff:]
    total_accuracy = df['total_accuracy'].values[start_cutoff:]
    effective_accuracy = total_accuracy / successful
    # print(f'effective accuracy: {effective_accuracy}')

    if 'clipper' in algorithm:
        clipper_accuracy = effective_accuracy

    # if len(clipper_accuracy) > 0:
    #     for i in range(len(effective_accuracy)):
    #         if effective_accuracy[i] < clipper_accuracy[i]:
    #             effective_accuracy[i] = clipper_accuracy[i]

    difference = demand - successful - dropped
    # print(f'difference: {difference}')
    print(f'sum of difference: {sum(difference)}')

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

    if demand_ewma is not None:
        ax1.plot(time, demand_ewma, label='Demand EWMA', color='black')

    if MARKERS_ON == True:
        ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx])
        ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx])
        ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx])
    else:
        ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx])
        ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx])
        ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx])

        # if 'estimated_throughput' in df and sum(df['estimated_throughput'].values[start_cutoff:]) > 0 and algorithms[idx] == 'Proteus':
        #     estimated_throughput = df['estimated_throughput'].values[start_cutoff:]
        #     ax1.plot(time, estimated_throughput, label=f'Estimated throughput ({algorithms[idx]})',
        #              color='black')
        # ax1.plot(time, successful, label=algorithms[idx])
        # ax2.plot(time, effective_accuracy, label=algorithms[idx])
    color_idx += 1
# plt.plot(time, capacity, label='capacity')
ax1.grid()
ax2.grid()
ax3.grid()

# plt.rcParams.update({'font.size': 30})
# plt.rc('axes', titlesize=30)     # fontsize of the axes title
# plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
# plt.rc('legend', fontsize=30)    # legend

y_cutoff += 50
# y_cutoff = 200

ax1.set_title(trace)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.75), ncol=3, fontsize=12)

ax1.set_xticks(np.arange(0, 25, 4), fontsize=15)
ax1.set_yticks(np.arange(0, y_cutoff + 50, 200), fontsize=12)
ax1.set_ylabel('Requests per\nsecond', fontsize=11)

ax2.set_xticks(np.arange(0, 25, 4), fontsize=15)
ax2.set_yticks(np.arange(80, 104, 5), fontsize=12)
ax2.set_ylabel('Effective\nAccuracy', fontsize=11)

ax3.set_xticks(np.arange(0, 25, 4), fontsize=15)
# ax3.set_yticks(np.arange(0, 0.6, 0.1), fontsize=12)
# ax3.set_ylabel('SLO Violation\nRatio', fontsize=11)
ax3.set_yticks(np.arange(0, 560, 100), fontsize=12)
ax3.set_ylabel('SLO Violations', fontsize=11)

ax3.set_xlabel('Time (min)', fontsize=12)

plt.savefig(os.path.join('..', 'figures', 'batching', f'timeseries_batching_{trace}.pdf'), dpi=500, bbox_inches='tight')

print(f'Warning! We should not be using mean to aggregate, instead we should be using sum')
print(f'Warning! There are some points where requests served are greater than incoming '
      f'demand. Fix this or find the cause')
