import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# trace = 'normal-high_load'
# trace = 'normal_load'
trace = 'medium-normal_load'
# trace = 'zipf_exponential'
# trace = 'zipf_exponential_bursty'
# trace = 'zipf_gamma'
# trace = 'zipf_uniform'
# trace = 'zipf_uniform_random'
# trace = 'equal_exponential'
# trace = 'equal_gamma'

path = '../logs/throughput/selected_asplos'
# slo = '150ms'
# slo = '1x'
# slo = '300ms'
# slo = '2.1x'
# slo = '3x'
slo = '300ms_cluster'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

logfile_list = [
                f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.05_0.6.csv', # 1x, 3x
                f'{path}/{trace}/{slo}/clipper_ht_aimd_300ms.csv', # this
                f'{path}/{trace}/{slo}/clipper_ha_aimd_300ms.csv', # this
                f'{path}/{trace}/{slo}/sommelier_asb_ewma1.6_beta1.5_300ms.csv', # 300ms, 2.1x, 3x
                f'{path}/{trace}/{slo}/proteus_ewma1.6_300ms.csv', # 300ms, 2.1x, 3x

                # f'{path}/{trace}/{slo}/proteus.csv'

                # f'{path}/{trace}/infaas_accuracy_300ms.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_0.1_0.4.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_0.1_0.5.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_0.05_0.6.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_0.05_0.8.csv',
                # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.01_0.6.csv', # 300ms
                # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.05_0.8.csv', # 2.1x
                # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.05_0.5.csv',
                # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.05_0.3.csv',
                # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.01_0.8.csv',
                # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_aimd.csv',
                # f'{path}/{trace}/50ms/infaas_accuracy_50ms.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_interval2.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_interval10.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_normalhighload.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack3.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack2.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack1.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack1.5.csv',
                # f'{path}/{trace}/infaas_accuracy_300ms_slack0.15.csv',
                # f'{path}/{trace}/clipper_ht_asb_300ms.csv', # this
                # f'{path}/{trace}/clipper_ht_aimd_lateallowed_300ms.csv',
                # f'{path}/{trace}/clipper_ht_nexus_300ms.csv',
                # f'{path}/{trace}/clipper_ht_asb_300ms.csv', # this
                # '../logs/throughput/selected_asplos/clipper_optstart_300ms.csv',
                # f'{path}/{trace}/sommelier_aimd_300ms.csv',
                # f'{path}/{trace}/sommelier_asb_300ms.csv', # this
                # f'{path}/{trace}/sommelier_asb_ewma1.1_300ms.csv',
                # f'{path}/{trace}/sommelier_aimd_ewma1.1_beta1.5_300ms.csv',
                # f'{path}/{trace}/sommelier_asb_ewma1.1_beta1.5_300ms.csv',
                # f'{path}/{trace}/sommelier_asb_300ms_intervaladaptive2.csv',
                # f'{path}/{trace}/sommelier_nexus_300ms.csv',
                # '../logs/throughput/selected_asplos/proteus_aimd_300ms.csv',
                # '../logs/throughput/selected_asplos/proteus_nexus_300ms.csv',
                # f'{path}/{trace}/proteus_300ms.csv',
                # f'{path}/{trace}/proteus_ewma1.1_300ms.csv',
                # f'{path}/{trace}/proteus_ewma2.1_300ms.csv',
                # f'{path}/{trace}/proteus_lessbatching_ewma1.1_earlydrop_300ms.csv',
                # f'{path}/{trace}/50ms/proteus_50ms.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.15.csv',
                # f'{path}/{trace}/proteus_300ms_beta1.1.csv',
                # f'{path}/{trace}/proteus_aimd_300ms.csv',
                # f'{path}/{trace}/proteus_aimd_lateallowed_300ms.csv',
                # f'{path}/{trace}/proteus_nexus_300ms.csv',
                # f'{path}/{trace}/proteus_nexus_lateallowed_300ms.csv',
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

MARKERS_ON = True

# We want to print the latest log file
logfile = logfile_list[-1]
print(logfile)

markers = ['.', 's', 'v', '^', 'x', '+', '*']
# markersizes = [7, 3, 4, 4, 6, 5, 6]
markersizes = [7, 3, 4, 4, 5, 6, 5]
# algorithms = ['AccScale', 'Clipper++ (High Accuracy)', 'Clipper++ (High Throughput)',
#             'INFaaS-Accuracy', 'INFaaS-Instance']
# algorithms = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
#             'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
algorithms = [
            #   'INFaaS-Accuracy',
            #   'INFaaS-Accuracy 0.1 0.4',
            #   'INFaaS-Accuracy 0.1 0.5',
            #   'INFaaS-Accuracy 0.05 0.6',
            #   'INFaaS-Accuracy 0.05 0.8',
              'INFaaS-Accuracy',
            #   'INFaaS-Accuracy Interval 2',
            #   'INFaaS-Accuracy Interval 10',
            #   'INFaaS-Accuracy NormalHighLoad',
            #   'INFaaS-Accuracy (Slack 3)',
            #   'INFaaS-Accuracy (Slack 2)',
            #   'INFaaS-Accuracy (Slack 1)',
            #   'INFaaS-Accuracy (Slack 1.5)',
            #   'INFaaS-Accuracy (Slack 0.15)',
              'Clipper-HT',
            #   'Clipper-HT-ASB',
              'Clipper-HA',
            #   'Clipper-HT-AIMD LateAllowed',
            #   'Clipper-HT-Nexus',
            #   'Clipper-HT-ASB',
            #   'Clipper-HT Optimized Start',
            #   'Sommelier-AIMD',
            #   'Sommelier-ASB',
            #   'Sommelier-ASB EWMA1.1',
            #   'Sommelier-AIMD EWMA1.1 Beta1.5',
            #   'Sommelier-ASB EWMA1.1 Beta1.5',
              'Sommelier',
            #   'Sommelier-ASB Interval2',
            #   'Sommelier-Nexus'
            #   'Proteus-Clipper',
            #   'Proteus-Nexus',
            #   'Proteus EWMA1.1',
              'Proteus',
            #   'Proteus EWMA2.1',
            #   'Proteus (Beta 1.15)',
            #   'Proteus (Beta 1.1)',
            #   'Proteus AIMD',
            #   'Proteus AIMD LateAllowed',
            #   'Proteus Nexus',
            #   'Proteus Nexus LateAllowed',
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

logfile_list = [logfile_list[2], logfile_list[0], logfile_list[1], logfile_list[3], logfile_list[4]]
algorithms = [algorithms[2], algorithms[0], algorithms[1], algorithms[3], algorithms[4]]

fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 1]})
fig.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.2)

# fig = plt.figure()
# gs = fig.add_gridspec(3,4)
# ax1 = fig.add_subplot(gs[0, :-1])
# ax2 = fig.add_subplot(gs[1, :-1])
# ax3 = fig.add_subplot(gs[2, :-1])
# ax4 = fig.add_subplot(gs[:, -1])

color_idx = 0
clipper_accuracy = []
slo_violation_ratios = []
throughputs = []
effective_accuracies = []
accuracy_drops = []
y_cutoff = 0

infaas_y_intercept = 0
sommelier_y_intercept = 0
proteus_y_intercept = 0

for idx in range(len(logfile_list)):
    logfile = logfile_list[idx]
    
    algorithm = logfile.split('/')[-1].rstrip('.csv')

    df = pd.read_csv(logfile)

    original_df = copy.deepcopy(df)

    aggregated = df.groupby(df.index // 10).sum(numeric_only=True)
    aggregated = df.groupby(df.index // 10).mean(numeric_only=True)
    df = aggregated
    # print(f'df: {df}')
    # print(f'aggregated: {aggregated}')

    start_cutoff = 0

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

    for i in range(len(successful)):
        if successful[i] > demand[i]:
            successful[i] = demand[i]

    # if len(clipper_accuracy) > 0:
    #     for i in range(len(effective_accuracy)):
    #         if effective_accuracy[i] < clipper_accuracy[i]:
    #             effective_accuracy[i] = clipper_accuracy[i]

    difference = demand - successful - dropped
    # print(f'difference: {difference}')
    print(f'sum of difference: {sum(difference)}')

    slo_violation_ratio = (sum(original_df['demand']) - sum(original_df['successful']) + sum(original_df['late'])) / sum(original_df['demand'])
    slo_violation_ratios.append(slo_violation_ratio)

     # throughputs.append((sum(original_df['successful']) - sum(original_df['late'])) / len(original_df['successful']))
    throughputs.append(sum(original_df['successful']) / len(original_df['successful']))

    overall_effective_accuracy = sum(original_df['total_accuracy']) / sum(original_df['successful'])
    effective_accuracies.append(overall_effective_accuracy)

    max_accuracy_drop = 100 - min(effective_accuracy)
    accuracy_drops.append(max_accuracy_drop)

    time = time
    time = [x - time[0] for x in time]
    # print(time[-1])
    time = [x / time[-1] * 24 for x in time]
    # print(time[0])
    # print(time[-1])

    if idx == 0:
        axs[0, 0].plot(time, demand, label='Demand', color=colors[color_idx],
                 marker=markers[color_idx])
        # ax1.plot(time, demand, label='Demand', marker=markers[color_idx])
        color_idx += 1
    # plt.plot(time, throughput, label=algorithm, marker=markers[idx])
    # plt.plot(time, throughput, label=algorithm)
    if MARKERS_ON == True:
        axs[0, 0].plot(time, successful, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        axs[1, 0].plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        axs[2, 0].plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
    # else:
    #     ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx])
    #     ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx])
    #     ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx])

        print(f'algorithm: {algorithm}, slo_violation_ratio: {slo_violation_ratio}')
        # if 'estimated_throughput' in df and sum(df['estimated_throughput'].values[start_cutoff:]) > 0 and algorithms[idx] == 'Proteus':
        #     estimated_throughput = df['estimated_throughput'].values[start_cutoff:]
        #     ax1.plot(time, estimated_throughput, label=f'Estimated throughput ({algorithms[idx]})',
        #              color='black')
        # ax1.plot(time, successful, label=algorithms[idx])
        # ax2.plot(time, effective_accuracy, label=algorithms[idx])

    if 'infaas' in algorithm:
        infaas_y_intercept = min(effective_accuracy)
    elif 'sommelier' in algorithm:
        sommelier_y_intercept = min(effective_accuracy)
    elif 'proteus' in algorithm:
        proteus_y_intercept = min(effective_accuracy)
    
    color_idx += 1
# plt.plot(time, capacity, label='capacity')
axs[0, 0].grid()
axs[1, 0].grid()
axs[2, 0].grid()


y_cutoff += 50
# y_cutoff = 200


axs[0, 0].legend(loc='upper center', bbox_to_anchor=(0.65, 1.65), ncol=3, fontsize=14)

axs[0, 0].set_xticklabels([])
axs[0, 0].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[0, 0].set_yticks(np.arange(0, y_cutoff + 50, 200), fontsize=12)
axs[0, 0].set_ylabel('Throughput', fontsize=11)

axs[1, 0].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[1, 0].set_xticklabels([])
axs[1, 0].set_yticks(np.arange(80, 104, 5), fontsize=12)
axs[1, 0].set_ylabel('Effective Acc.', fontsize=11)

axs[2, 0].set_xticks(np.arange(0, 25, 4), fontsize=15)
# ax3.set_yticks(np.arange(0, 0.6, 0.1), fontsize=12)
# ax3.set_ylabel('SLO Violation\nRatio', fontsize=11)
axs[2, 0].set_yticks(np.arange(0, 510, 100), fontsize=12)
axs[2, 0].set_ylabel('SLO Violations', fontsize=11)

axs[2, 0].set_xlabel('Time (min)', fontsize=12)

print(f'infaas_y_intercept: {infaas_y_intercept}')
print(f'sommelier_y_intercept: {sommelier_y_intercept}')
print(f'proteus_y_intercept: {proteus_y_intercept}')

# ax2.plot(time, np.repeat(infaas_y_intercept, len(time)), color=colors[1],
#          linestyle='--', linewidth=1)
# ax2.plot(time, np.repeat(sommelier_y_intercept, len(time)), color=colors[4],
#          linestyle='--', linewidth=1)
# ax2.plot(time, np.repeat(proteus_y_intercept, len(time)), color=colors[5],
#          linestyle='--', linewidth=1)
# axs[1, 0].plot(time, np.repeat(infaas_y_intercept, len(time)), color='black',
#          linestyle='--', linewidth=1)
# axs[1, 0].plot(time, np.repeat(sommelier_y_intercept, len(time)), color='black',
#          linestyle='--', linewidth=1)
# axs[1, 0].plot(time, np.repeat(proteus_y_intercept, len(time)), color='black',
#          linestyle='--', linewidth=1)

# plt.savefig(os.path.join('..', 'figures', 'asplos', 'endtoend_comparison',
#                          f'timeseries_{trace}_{slo}.pdf'), dpi=500, bbox_inches='tight')

print(f'Warning! We should not be using mean to aggregate, instead we should be using sum')
print(f'Warning! There are some points where requests served are greater than incoming '
      f'demand. Fix this or find the cause')


print(f'slo_violation_ratios: {slo_violation_ratios}')
print(f'throughputs: {throughputs}')
# print(f'accuracy drop')

# plt.close()
# plt.xlabel('Algorithm', fontsize=13)
# plt.ylabel('SLO Violation Ratio', fontsize=13)
# plt.bar(algorithms, slo_violation_ratios, color=colors[1:])
# plt.yticks(np.arange(0, 0.38, 0.05), fontsize=12)
# plt.savefig(os.path.join('..', 'figures', 'asplos', 'endtoend_comparison',
#             f'slo_bar_{trace}_{slo}.pdf'), dpi=500, bbox_inches='tight')


hatches = ['-', '\\', '/', 'x', '+']

# idx_map = [2, 0, 1, 3, 4]
# print(algorithms)
# algorithms = [algorithms[2], algorithms[0], algorithms[1], algorithms[3], algorithms[4]]
# print(algorithms)
# # print(throughputs)
# throughputs = [throughputs[2], throughputs[0], throughputs[1], throughputs[3], throughputs[4]]
# print(colors)
# colors = [colors[3], colors[1], colors[2], colors[4], colors[5]]
# print(colors)
# slo_violation_ratios = [slo_violation_ratios[2], slo_violation_ratios[0],
#                         slo_violation_ratios[1], slo_violation_ratios[3],
#                         slo_violation_ratios[4]]

# axs[0, 1].set_xlabel('Algorithm', fontsize=12)
axs[0, 1].set_ylabel('Avg. Throughput', fontsize=11)
axs[0, 1].bar(algorithms, throughputs, label=algorithms, color=colors[1:],
              hatch=hatches, edgecolor='black')
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks(np.arange(0, 410, 100), fontsize=12)

# axs[1, 1].set_xlabel('Algorithm', fontsize=12)
axs[1, 1].set_ylabel('Max. Acc. Drop', fontsize=11)
axs[1, 1].bar(algorithms, accuracy_drops, label=algorithms, color=colors[1:],
              hatch=hatches, edgecolor='black')
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks(np.arange(0, 21, 5), fontsize=12)

axs[2, 1].set_xlabel('Algorithm', fontsize=12)
axs[2, 1].set_ylabel('SLO Violation Ratio', fontsize=11)
axs[2, 1].bar(algorithms, slo_violation_ratios, label=algorithms, color=colors[1:],
              hatch=hatches, edgecolor='black')
axs[2, 1].set_yticks(np.arange(0, 0.41, 0.1), fontsize=12)
# axs[2, 1].set_xticklabels(['INFaaS\nAccuracy', 'Clipper\nHT', 'Clipper\nHA', 'Sommelier', 'Proteus'])
# axs[2, 1]
axs[2, 1].set_xticks([])

# axs[2, 1].legend(loc='lower center', bbox_to_anchor=(-2, -1), ncol=3, fontsize=12)

plt.savefig(os.path.join('..', 'figures', 'asplos', 'endtoend_comparison',
            f'{trace}.pdf'), dpi=500, bbox_inches='tight')
plt.savefig(os.path.join('..', 'figures', 'asplos', 'endtoend_comparison',
            f'something.pdf'), dpi=500, bbox_inches='tight')
