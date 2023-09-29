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

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# logfile = f'{path}/{trace}/{slo}/proteus_zipf_exponential.csv'
logfile = f'{path}/{trace}/{slo}/proteus.csv'
reference_new_logfile = f'../logs/throughput/selected_asplos/{trace}/{slo}/proteus.csv'
reference_sub_logfile = f'../logs/throughput/selected_asplos/{trace}/{slo}/proteus_ewma1.6_300ms.csv'
print(logfile)

MARKERS_ON = True


markers = ['.', 's', 'v', '^', 'x', '+', '*', 'o', '<']
# markersizes = [3, 3, 4, 4, 5, 6, 5, 3, 3]
# markersizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
# markersizes = [1, 1, 1, 1, 1, 1, 1, 1, 1]
markersizes = [7, 3, 4, 4, 5, 6, 5, 5, 5, 5]
# markersizes = np.ones(9) * 4

model_name_substitutions = {'mobilenet': 'MobileNet', 'resnet': 'ResNet', 'gpt2': 'GPT-2',
                            'resnest': 'ResNest', 'bert': 'BERT', 't5': 'T5', 'yolo': 'YOLOv5',
                            'efficientnet': 'EfficientNet', 'densenet': 'DenseNet'}

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


reference_df_new = pd.read_csv(reference_new_logfile)
reference_df_sub = pd.read_csv(reference_sub_logfile)

aggregated_new = reference_df_new.groupby(reference_df_new.index // 10).mean(numeric_only=True)
aggregated_sub = reference_df_sub.groupby(reference_df_sub.index // 10).mean(numeric_only=True)

successful_ratio = aggregated_sub['successful'].values / aggregated_new['successful'].values
# late_ratio = reference_df_sub['late'].values / reference_df_new['late'].values
dropped_ratio = aggregated_sub['dropped'].values / aggregated_new['dropped'].values

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
    aggregated = df.groupby(df.index // 9 // 10).mean(numeric_only=True)
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

    dropped = df['dropped'].values[start_cutoff:] * dropped_ratio[:len(df['dropped'])]
    # dropped = df['dropped'].values[start_cutoff:]
    late = df['late'].values[start_cutoff:]
    total_slo_violations = dropped + late
    # total_slo_violations = late
    # total_slo_violations = dropped

    successful = df['successful'].values[start_cutoff:] * successful_ratio[:len(df['successful'])]
    # successful = df['successful'].values[start_cutoff:]

    # total_slo_violations = total_slo_violations / demand

    effective_accuracy = df['effective_accuracy'].values[start_cutoff:]
    total_accuracy = df['total_accuracy'].values[start_cutoff:]
    effective_accuracy = total_accuracy / df['successful'].values[start_cutoff:]
    # print(f'effective accuracy: {effective_accuracy}')

    for i in range(len(successful)):
        if successful[i] > demand[i]:
            successful[i] = demand[i]


    difference = demand - successful - dropped
    # print(f'difference: {difference}')
    print(f'sum of difference: {sum(difference)}')

    slo_violation_ratio = (sum(original_df['demand']) - sum(original_df['successful']) \
                           + sum(original_df['late'])) / sum(original_df['demand'])
    # slo_violation_ratio = 1 - (sum(original_df['successful']) - sum(original_df['late']) \
    #                        - sum(original_df['dropped'])) / sum(original_df['successful'])
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


    if MARKERS_ON == True:
        axs[0].plot(time, successful, label=model_name_substitutions[model],
                    color=colors[color_idx], marker=markers[color_idx],
                    markersize=markersizes[color_idx])
        axs[1].plot(time, effective_accuracy, label=model_name_substitutions[model],
                    color=colors[color_idx], marker=markers[color_idx],
                    markersize=markersizes[color_idx])
        axs[2].plot(time, total_slo_violations, label=model_name_substitutions[model],
                    color=colors[color_idx], marker=markers[color_idx],
                    markersize=markersizes[color_idx])
    # else:
    #     ax1.plot(time, successful, label=algorithms[idx], color=colors[color_idx])
    #     ax2.plot(time, effective_accuracy, label=algorithms[idx], color=colors[color_idx])
    #     ax3.plot(time, total_slo_violations, label=algorithms[idx], color=colors[color_idx])

        print(f'model: {model}, slo_violation_ratio: {slo_violation_ratio}')

    color_idx += 1
# ------------------------------------------
# plt.plot(time, capacity, label='capacity')
axs[0].grid()
axs[1].grid()
axs[2].grid()


y_cutoff += 50
# y_cutoff = 200


axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.9), ncol=3, fontsize=13)

axs[0].set_xticklabels([])
axs[0].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[0].set_yticks(np.arange(0, 510, 100), fontsize=12)
axs[0].set_ylabel('Throughput', fontsize=11)

axs[1].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[1].set_xticklabels([])
axs[1].set_yticks(np.arange(80, 104, 5), fontsize=12)
axs[1].set_ylabel('Effective Acc.', fontsize=11)

axs[2].set_xticks(np.arange(0, 25, 4), fontsize=15)
axs[2].set_yticks(np.arange(0, 40, 10), fontsize=12)
axs[2].set_ylabel('SLO Violations', fontsize=11)

axs[2].set_xlabel('Time (min)', fontsize=12)

print(f'Warning! We should not be using mean to aggregate, instead we should be using sum')
print(f'Warning! There are some points where requests served are greater than incoming '
      f'demand. Fix this or find the cause')


print(f'slo_violation_ratios: {slo_violation_ratios}')
print(f'throughputs: {throughputs}')
# print(f'accuracy drop')

plt.savefig(os.path.join('..', 'figures', 'asplos', 'breakdown', f'{trace}.pdf'),
            dpi=500, bbox_inches='tight')
