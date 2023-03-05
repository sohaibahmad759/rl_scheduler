import os
import glob
import numpy as np
import matplotlib.pyplot as plt

filepath = os.path.join('..', 'logs', 'latency', 'selected')

cdf_cutoff_value = 5000

# fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
i = 0
j = 0

files = sorted(glob.glob(os.path.join(filepath, '250_ilp_scaled_max_batch_size_*.csv')))
# readfile = files[-1]
for readfile in files:
    response_times_all = {}
    with open(readfile, mode='r') as rf:
        response_times = []
        for line in rf.readlines():
            model, start_time, finish_time, response_time = line.rstrip('\n').split(',')
            response_time = float(response_time)
            # if 'response time=' in line:
            #     response_time = float(line.split('response time=')[1].split(',')[0])*1000
            #     if response_time < cdf_cutoff_value:
            #         response_times.append(response_time)
            if response_time > cdf_cutoff_value:
                continue
            if model in response_times_all:
                response_times_all[model].append(response_time)
            else:
                response_times_all[model] = [response_time]

    for model in response_times_all:
        # getting data of the histogram
        count, bins_count = np.histogram(response_times_all[model], bins=30)
        # finding the PDF of the histogram using count values
        pdf = count / sum(count)
        # using numpy np.cumsum to calculate the CDF
        cdf = np.cumsum(pdf)

        if 'resnet' in model:
            label = 'ResNet'
            marker = 'o'
        elif 'resnest' in model:
            label = 'ResNest'
            marker = 'v'
        elif 'mobilenet' in model:
            label = 'MobileNet'
            marker = '^'
        elif 'efficientnet' in model:
            label = 'EfficientNet'
            marker = '*'
        elif 'densenet' in model:
            label = 'DenseNet'
            marker = 's'

        # we do this if we only need the first and last graph, otherwise we comment this
        if i == 0 and j == 0 or i == 1 and j == 1:
            if i == 0:
                axs[i].plot(bins_count[1:], cdf, label=label, marker=marker)
            else:
                axs[i].plot(bins_count[1:], cdf, marker=marker)
        else:
            continue


        # if i == 0 and j == 0:
        #     axs[i, j].plot(bins_count[1:], cdf, label=label, marker=marker)
        # else:
        #     axs[i, j].plot(bins_count[1:], cdf, marker=marker)


    # axs[i, j].set_xticks(np.arange(0, 255, 50))
    # axs[i, j].set_yticks(np.arange(0.0, 1.1, 0.2))
    # if i == 1:
    #     axs[i, j].set_xlabel('Latency (ms)', fontsize=20)
    # if j == 0:
    #     axs[i, j].set_ylabel('Ratio of Requests', fontsize=20)
    # axs[i, j].tick_params(axis='both', which='major', labelsize=15)
    # axs[i, j].grid(linestyle='--')

    axs[i].set_xticks(np.arange(0, 255, 50))
    axs[i].set_yticks(np.arange(0.0, 1.1, 0.2))
    # if i == 1:
    axs[i].set_xlabel('Latency (ms)', fontsize=20)
    if i == 0:
        axs[i].set_ylabel('Ratio of Requests', fontsize=20)
    axs[i].tick_params(axis='both', which='major', labelsize=15)
    axs[i].grid(linestyle='--')
    
    j += 1
    if j == 2:
        i += 1
        j = 0

    # plot_file = os.path.join(filepath, 'cdf.pdf')
    # plt.savefig(plot_file, dpi=500)
# axs[0, 0].set_title('Max batch size = 1', fontsize=20)
# axs[0, 1].set_title('Max batch size = 2', fontsize=20)
# axs[1, 0].set_title('Max batch size = 4', fontsize=20)
# axs[1, 1].set_title('Max batch size = 8', fontsize=20)
axs[0].set_title('No batching', fontsize=17.5)
axs[1].set_title('Adaptive batching (ASB)', fontsize=17.5)
fig.legend(loc='upper center', bbox_to_anchor=(
    0.5, 1.15), ncol=3, prop={"size": 15})
fig.tight_layout(pad=2.0)

plt.savefig('cdf_all_batch_sizes.pdf', dpi=500, bbox_inches='tight')

print(files)
print(response_times)
