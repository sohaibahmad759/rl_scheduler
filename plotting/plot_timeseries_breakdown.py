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

path = '../logs/throughput_per_model/selected_asplos'
# slo = '150ms'
# slo = '1x'
slo = '300ms'
# slo = '2.1x'
# slo = '3x'
# slo = '300ms_cluster'

# logfile = f'{path}/{trace}/{slo}/proteus_zipf_exponential.csv'
logfile = f'{path}/{trace}/{slo}/proteus.csv'
print(logfile)

MARKERS_ON = True


markers = ['.', 's', 'v', '^', 'x', '+', '*', 'o', '<']
# markersizes = [3, 3, 4, 4, 5, 6, 5, 3, 3]
# markersizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
# markersizes = [1, 1, 1, 1, 1, 1, 1, 1, 1]
markersizes = np.ones(9) * 2

colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2', 'tab:olive', 'tab:cyan']

# fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 1]})
fig, axs = plt.subplots(3, 1)
fig.tight_layout()
plt.subplots_adjust(wspace=0.25, hspace=0.2)

color_idx = 0
clipper_accuracy = []
slo_violation_ratios = []
throughputs = []
effective_accuracies = []
accuracy_drops = []
y_cutoff = 0


# ------------------------------------------
df = pd.read_csv(logfile)
unmodified_df = copy.deepcopy(df)

models = df['model'].unique()
print(models)
print(f'len(models): {len(models)}')

for model in models:

    df = unmodified_df[unmodified_df['model'] == model]
    original_df = copy.deepcopy(df)

    # aggregated = df.groupby(df.index // 10).sum(numeric_only=True)
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

    for i in range(len(successful)):
        if successful[i] > demand[i]:
            successful[i] = demand[i]


    difference = demand - successful - dropped
    # print(f'difference: {difference}')
    print(f'sum of difference: {sum(difference)}')

    slo_violation_ratio = (sum(original_df['demand']) - sum(original_df['successful']) \
                           + sum(original_df['late'])) / sum(original_df['demand'])
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

    # if idx == 0:
    #     axs[0, 0].plot(time, demand, label='Demand', color=colors[color_idx],
    #                 marker=markers[color_idx])
    #     color_idx += 1

    # normalized_throughput = 1 - (demand - throughput) / demand

    if MARKERS_ON == True:
        axs[0].plot(time, successful, label=model, color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        # axs[0].plot(time, normalized_throughput, label=model, color=colors[color_idx],
        #         marker=markers[color_idx], markersize=markersizes[color_idx])
        axs[1].plot(time, effective_accuracy, label=model, color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
        axs[2].plot(time, total_slo_violations, label=model, color=colors[color_idx],
                marker=markers[color_idx], markersize=markersizes[color_idx])
    # else:
    #     ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx])
    #     ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx])
    #     ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx])

        print(f'model: {model}, slo_violation_ratio: {slo_violation_ratio}')
        # if 'estimated_throughput' in df and sum(df['estimated_throughput'].values[start_cutoff:]) > 0 and algorithms[idx] == 'Proteus':
        #     estimated_throughput = df['estimated_throughput'].values[start_cutoff:]
        #     ax1.plot(time, estimated_throughput, label=f'Estimated throughput ({algorithms[idx]})',
        #              color='black')
        # ax1.plot(time, successful, label=algorithms[idx])
        # ax2.plot(time, effective_accuracy, label=algorithms[idx])

    # if 'infaas' in algorithm:
    #     infaas_y_intercept = min(effective_accuracy)
    # elif 'sommelier' in algorithm:
    #     sommelier_y_intercept = min(effective_accuracy)
    # elif 'proteus' in algorithm:
    #     proteus_y_intercept = min(effective_accuracy)

    color_idx += 1
# ------------------------------------------
# plt.plot(time, capacity, label='capacity')
axs[0].grid()
axs[1].grid()
axs[2].grid()


y_cutoff += 50
# y_cutoff = 200


# axs[0].legend(loc='upper center', bbox_to_anchor=(0.65, 1.65), ncol=3, fontsize=14)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.45, 1.85), ncol=3, fontsize=12)

axs[0].set_xticklabels([])
axs[0].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[0].set_yticks(np.arange(0, y_cutoff + 50, 200), fontsize=12)
# axs[0].set_yticks(np.arange(0, 1.1, 0.2), fontsize=12)
axs[0].set_ylabel('Throughput', fontsize=11)

axs[1].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[1].set_xticklabels([])
axs[1].set_yticks(np.arange(80, 104, 5), fontsize=12)
axs[1].set_ylabel('Effective Acc.', fontsize=11)

axs[2].set_xticks(np.arange(0, 25, 4), fontsize=15)
# ax3.set_yticks(np.arange(0, 0.6, 0.1), fontsize=12)
# ax3.set_ylabel('SLO Violation\nRatio', fontsize=11)
# axs[2, 0].set_yticks(np.arange(0, 510, 100), fontsize=12)
axs[2].set_yticks(np.arange(0, 110, 20), fontsize=12)
axs[2].set_ylabel('SLO Violations', fontsize=11)

axs[2].set_xlabel('Time (min)', fontsize=12)

# print(f'infaas_y_intercept: {infaas_y_intercept}')
# print(f'sommelier_y_intercept: {sommelier_y_intercept}')
# print(f'proteus_y_intercept: {proteus_y_intercept}')

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

# # Plotting the bar charts
# # ------------------------------------------
# axs[0, 1].set_ylabel('Avg. Throughput', fontsize=11)
# axs[0, 1].bar(algorithms, throughputs, label=algorithms, color=colors[1:],
#               hatch=hatches, edgecolor='black')
# axs[0, 1].set_xticks([])
# axs[0, 1].set_yticks(np.arange(0, 410, 100), fontsize=12)

# # axs[1, 1].set_xlabel('Algorithm', fontsize=12)
# axs[1, 1].set_ylabel('Max. Acc. Drop', fontsize=11)
# axs[1, 1].bar(algorithms, accuracy_drops, label=algorithms, color=colors[1:],
#               hatch=hatches, edgecolor='black')
# axs[1, 1].set_xticks([])
# axs[1, 1].set_yticks(np.arange(0, 21, 5), fontsize=12)

# axs[2, 1].set_xlabel('Algorithm', fontsize=12)
# axs[2, 1].set_ylabel('SLO Violation Ratio', fontsize=11)
# axs[2, 1].bar(algorithms, slo_violation_ratios, label=algorithms, color=colors[1:],
#               hatch=hatches, edgecolor='black')
# axs[2, 1].set_yticks(np.arange(0, 0.41, 0.1), fontsize=12)
# # axs[2, 1].set_xticklabels(['INFaaS\nAccuracy', 'Clipper\nHT', 'Clipper\nHA', 'Sommelier', 'Proteus'])
# # axs[2, 1]
# axs[2, 1].set_xticks([])
# # ------------------------------------------

# axs[2, 1].legend(loc='lower center', bbox_to_anchor=(-2, -1), ncol=3, fontsize=12)

plt.savefig(os.path.join('..', 'figures', 'asplos', 'breakdown', f'{trace}.pdf'),
            dpi=500, bbox_inches='tight')
