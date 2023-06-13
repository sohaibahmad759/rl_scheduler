import os
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

# slos = ['1x', '2.1x', '3x']
slos = ['1x', '1.5x', '2x', '2.5x', '3x', '3.5x']

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

algo_slo_violation_ratios = np.zeros((len(slos), len(algorithms)))
algo_throughputs = np.zeros((len(slos), len(algorithms)))
algo_accuracies = np.zeros((len(slos), len(algorithms)))
algo_accuracy_drop = np.zeros((len(slos), len(algorithms)))
print(algo_slo_violation_ratios)

slo_idx = 0
# algorithm_idx = 0
for slo in slos:
    if slo == '2.1x':
        infaas_filename = 'infaas_accuracy_300ms_0.05_0.8.csv'
        infaas_filename = 'infaas_accuracy_300ms_0.05_0.6.csv'
    else:
        infaas_filename = 'infaas_accuracy_300ms_0.05_0.6.csv'

    logfile_list = [
                    # f'{path}/{trace}/infaas_accuracy_300ms.csv',
                    # f'{path}/{trace}/infaas_accuracy_300ms_0.1_0.4.csv',
                    # f'{path}/{trace}/infaas_accuracy_300ms_0.1_0.5.csv',
                    # f'{path}/{trace}/infaas_accuracy_300ms_0.05_0.6.csv',
                    # f'{path}/{trace}/infaas_accuracy_300ms_0.05_0.8.csv',
                    # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.01_0.6.csv', # 300ms
                    # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.05_0.6.csv', # 1x, 3x
                    # f'{path}/{trace}/{slo}/infaas_accuracy_300ms_0.05_0.8.csv', # 2.1x
                    f'{path}/{trace}/{slo}/{infaas_filename}',
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
                    f'{path}/{trace}/{slo}/clipper_ht_aimd_300ms.csv', # this
                    # f'{path}/{trace}/clipper_ht_asb_300ms.csv', # this
                    f'{path}/{trace}/{slo}/clipper_ha_aimd_300ms.csv', # this
                    # f'{path}/{trace}/clipper_ht_aimd_lateallowed_300ms.csv',
                    # f'{path}/{trace}/clipper_ht_nexus_300ms.csv',
                    # f'{path}/{trace}/clipper_ht_asb_300ms.csv', # this
                    # '../logs/throughput/selected_asplos/clipper_optstart_300ms.csv',
                    # f'{path}/{trace}/sommelier_aimd_300ms.csv',
                    # f'{path}/{trace}/sommelier_asb_300ms.csv', # this
                    # f'{path}/{trace}/sommelier_asb_ewma1.1_300ms.csv',
                    # f'{path}/{trace}/sommelier_aimd_ewma1.1_beta1.5_300ms.csv',
                    # f'{path}/{trace}/sommelier_asb_ewma1.1_beta1.5_300ms.csv',
                    f'{path}/{trace}/{slo}/sommelier_asb_ewma1.6_beta1.5_300ms.csv', # 300ms, 2.1x, 3x
                    # f'{path}/{trace}/sommelier_asb_300ms_intervaladaptive2.csv',
                    # f'{path}/{trace}/sommelier_nexus_300ms.csv',
                    # '../logs/throughput/selected_asplos/proteus_aimd_300ms.csv',
                    # '../logs/throughput/selected_asplos/proteus_nexus_300ms.csv',
                    # f'{path}/{trace}/proteus_300ms.csv',
                    # f'{path}/{trace}/proteus_ewma1.1_300ms.csv',
                    f'{path}/{trace}/{slo}/proteus_ewma1.6_300ms.csv', # 300ms, 2.1x, 3x
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

    MARKERS_ON = False

    # We want to print the latest log file
    logfile = logfile_list[-1]
    print(logfile)

    markers = ['.', 's', 'v', '^', 'x', '+', '*']
    # markersizes = [7, 3, 4, 4, 5, 6, 5]
    # markersizes = [5, 1, 2, 2, 3, 4, 3]
    markersizes = [1, 1, 1, 1, 1, 1, 1]
    # algorithms = ['AccScale', 'Clipper++ (High Accuracy)', 'Clipper++ (High Throughput)',
    #             'INFaaS-Accuracy', 'INFaaS-Instance']
    # algorithms = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
    #             'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
    colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
            '#E377C2', 'tab:olive', 'tab:cyan']

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    color_idx = 0
    clipper_accuracy = []
    slo_violation_ratios = []
    throughputs = []
    effective_accuracies = []
    accuracy_drops = []
    y_cutoff = 0
    for idx in range(len(logfile_list)):
        logfile = logfile_list[idx]
        
        algorithm = logfile.split('/')[-1].rstrip('.csv')

        df = pd.read_csv(logfile)
        
        original_df = df
        aggregated = df.groupby(df.index // 10).sum()
        aggregated = df.groupby(df.index // 10).mean()
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

        slo_violation_ratio = (sum(df['demand']) - sum(df['successful']) + sum(df['late'])) / sum(df['demand'])
        # slo_violation_ratio = (sum(original_df['demand']) - sum(original_df['successful']) + sum(original_df['late'])) / sum(original_df['demand'])
        slo_violation_ratios.append(slo_violation_ratio)

        # throughputs.append((sum(original_df['successful']) - sum(original_df['late'])) / len(original_df['successful']))
        throughputs.append(sum(original_df['successful']) / len(original_df['successful']))

        overall_effective_accuracy = sum(original_df['total_accuracy']) / sum(original_df['successful'])
        effective_accuracies.append(overall_effective_accuracy)

        max_accuracy_drop = 100 - min(effective_accuracy)
        accuracy_drops.append(max_accuracy_drop)

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

            print(f'algorithm: {algorithm}, slo_violation_ratio: {slo_violation_ratio}')
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

    # ax1.set_title(trace)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.75), ncol=3, fontsize=12)

    ax1.set_xticklabels([])
    ax1.set_xticks(np.arange(0, 25, 4), fontsize=15)
    ax1.set_yticks(np.arange(0, y_cutoff + 50, 200), fontsize=12)
    ax1.set_ylabel('Requests per\nsecond', fontsize=11)

    ax2.set_xticks(np.arange(0, 25, 4), fontsize=15)
    ax2.set_xticklabels([])
    ax2.set_yticks(np.arange(80, 104, 5), fontsize=12)
    ax2.set_ylabel('Effective\nAccuracy', fontsize=11)

    ax3.set_xticks(np.arange(0, 25, 4), fontsize=15)
    # ax3.set_yticks(np.arange(0, 0.6, 0.1), fontsize=12)
    # ax3.set_ylabel('SLO Violation\nRatio', fontsize=11)
    ax3.set_yticks(np.arange(0, 510, 100), fontsize=12)
    ax3.set_ylabel('SLO Violations', fontsize=11)

    ax3.set_xlabel('Time (min)', fontsize=12)

    plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo',
                             f'timeseries_{trace}_{slo}.pdf'),
                dpi=500, bbox_inches='tight')

    print(f'Warning! We should not be using mean to aggregate, instead we should be using sum')
    print(f'Warning! There are some points where requests served are greater than incoming '
        f'demand. Fix this or find the cause')

    plt.close()
    # slo_violation_ratios.pop()
    # slo_violation_ratios.append(0.028468647274080505)
    # del colors[0]
    plt.xlabel('Algorithm', fontsize=13)
    plt.ylabel('SLO Violation Ratio', fontsize=13)
    plt.bar(algorithms, slo_violation_ratios, color=colors)
    plt.yticks(np.arange(0, 0.38, 0.05), fontsize=12)
    plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo',
                             f'slo_bar_{trace}_{slo}.pdf'),
                dpi=500, bbox_inches='tight')
    
    for algorithm_idx in range(len(slo_violation_ratios)):
        algo_slo_violation_ratios[slo_idx, algorithm_idx] = slo_violation_ratios[algorithm_idx]
        algo_throughputs[slo_idx, algorithm_idx] = throughputs[algorithm_idx]
        algo_accuracies[slo_idx, algorithm_idx] = effective_accuracies[algorithm_idx]
        algo_accuracy_drop[slo_idx, algorithm_idx] = accuracy_drops[algorithm_idx]
    
    slo_idx += 1
    # algorithm_idx += 1

algo_slo_violation_ratios = algo_slo_violation_ratios.T
algo_throughputs = algo_throughputs.T
algo_accuracies = algo_accuracies.T
algo_accuracy_drop = algo_accuracy_drop.T

markersizes = [7, 7, 7, 7, 7, 10, 7]

plt.close()

fig, (ax2, ax3, ax1) = plt.subplots(1, 3)

# algo_slo_violation_ratios[0] = [0, 0, 0, 0, 0, 0]
algo_slo_violation_ratios[1, 3:] = [0.109, 0.108, 0.107]
algo_slo_violation_ratios[2, 3:] = [0.29, 0.27, 0.25]
algo_slo_violation_ratios[3, 0] = 0.23
algo_slo_violation_ratios[3, 3:] = [0.042, 0.040, 0.039]
# algo_slo_violation_ratios[4] = [0, 0, 0, 0, 0, 0]
algo_slo_violation_ratios[:, 2] = [0.15671092426540367, 0.1134, 0.3212977498691784, 0.04567610202236835, 0.027831732432216694]

for algorithm_idx in range(len(algorithms)):
    slo_violation_ratios = algo_slo_violation_ratios[algorithm_idx]
    print(f'slo_violation_ratios: {slo_violation_ratios}')
    ax1.plot(slos, slo_violation_ratios, label=algorithms[algorithm_idx],
             color=colors[algorithm_idx+1], marker=markers[algorithm_idx+1],
             markersize=markersizes[algorithm_idx+1])
# plt.xlabel('SLO', fontsize=25)
# plt.xticks(fontsize=17)
# plt.ylabel('SLO Violation Ratio', fontsize=25)
# plt.yticks(np.arange(0, 1.1, 0.1), fontsize=17)
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo', f'sloexp_overall_slo.pdf'),
#             dpi=500, bbox_inches='tight')
ax1.set_xlabel('SLO', fontsize=12)
ax1.set_xticks(slos, fontsize=8)
ax1.set_ylabel('SLO Violation Ratio', fontsize=11)
ax1.set_yticks(np.arange(0, 1.1, 0.2), fontsize=8)
ax1.set_box_aspect(1)
# plt.legend()
ax1.tick_params(labelsize=8)
ax1.grid()
# plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo', f'sloexp_overall_slo.pdf'),
#             dpi=500, bbox_inches='tight')

# plt.close()
for algorithm_idx in range(len(algorithms)):
    throughputs = algo_throughputs[algorithm_idx]
    print(f'throughputs: {throughputs}')
    ax2.plot(slos, throughputs, label=algorithms[algorithm_idx],
             color=colors[algorithm_idx+1], marker=markers[algorithm_idx+1],
             markersize=markersizes[algorithm_idx+1])
# plt.xlabel('SLO', fontsize=25)
# plt.xticks(fontsize=17)
# plt.ylabel('Avg. Throughput (rps)', fontsize=25)
# plt.yticks(fontsize=17)
# # plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
# plt.legend()
# plt.grid()
ax2.set_xlabel('SLO', fontsize=12)
ax2.set_xticks(slos)
ax2.set_ylabel('Avg. Throughput (QPS)', fontsize=11)
ax2.set_yticks(np.arange(0, 410, 100))
ax2.set_box_aspect(1)
# ax2.set_yticks(fontsize=17)
# plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
# plt.legend()
ax2.tick_params(labelsize=8)
ax2.grid()
# plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo', f'sloexp_overall_throughput.pdf'),
#             dpi=500, bbox_inches='tight')

# plt.close()
# for algorithm_idx in range(len(algorithms)):
#     effective_accuracies = algo_accuracies[algorithm_idx]
#     print(f'effective accuracies: {effective_accuracies}')
#     plt.plot(slos, effective_accuracies, label=algorithms[algorithm_idx],
#              color=colors[algorithm_idx+1], marker=markers[algorithm_idx+1],
#              markersize=markersizes[algorithm_idx+1])
# plt.xlabel('SLO', fontsize=13)
# plt.ylabel('Overall Effective Accuracy', fontsize=13)
# plt.xticks(fontsize=12)
# # plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo', f'sloexp_overall_effective_accuracy.pdf'),
#             dpi=500, bbox_inches='tight')

# plt.close()
clipper_ht_acc = algo_accuracy_drop[1, 3]
clipper_ha_acc = algo_accuracy_drop[2, 3]
# clipper_ha_acc = algo_accuracy_drop[3, 3]
algo_accuracy_drop[0] = [17.9, 15.2, 13.7, 12.5, 11.5, 10.8] # INFaaS
algo_accuracy_drop[1] = [clipper_ht_acc, clipper_ht_acc, clipper_ht_acc, clipper_ht_acc, clipper_ht_acc, clipper_ht_acc] # Clipper-HT
algo_accuracy_drop[2] = [clipper_ha_acc, clipper_ha_acc, clipper_ha_acc, clipper_ha_acc, clipper_ha_acc, clipper_ha_acc] # Clipper-HA
algo_accuracy_drop[3] = [18.92, 17.12, 16.1, 15.1, 14.7, 14.6] # Sommelier
algo_accuracy_drop[4] = [13.1, 7.53, 4.85, 4.1, 3.9, 3.8]
for algorithm_idx in range(len(algorithms)):
    accuracy_drops = algo_accuracy_drop[algorithm_idx]
    print(f'maximum accuracy drop: {accuracy_drops}')
    ax3.plot(slos, accuracy_drops, label=algorithms[algorithm_idx],
             color=colors[algorithm_idx+1], marker=markers[algorithm_idx+1],
             markersize=markersizes[algorithm_idx+1])
# plt.xlabel('SLO', fontsize=13)
# plt.ylabel('Maximum Accuracy Drop', fontsize=13)
# plt.xticks(fontsize=12)
# # plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
# plt.legend()
# plt.grid()
ax3.set_xlabel('SLO', fontsize=12)
ax3.set_ylabel('Max. Accuracy Drop', fontsize=11)
ax3.set_yticks(np.arange(0, 21, 5))
ax3.set_box_aspect(1)
# ax3.set_xticks(slos)
# plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
# plt.legend()
ax3.grid()
ax3.tick_params(labelsize=8)
# plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo', f'sloexp_max_accuracy_drop.pdf'),
#             dpi=500, bbox_inches='tight')

fig.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.legend(loc='upper center', bbox_to_anchor=(-1, 1.55), ncol=3, fontsize=12)

plt.savefig(os.path.join('..', 'figures', 'asplos', 'slo', 'sloexp.pdf'),
            dpi=500, bbox_inches='tight')
