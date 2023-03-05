import os
import glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob(os.path.join('..', 'logs', 'paper_jun13_onwards', 'blis_ilp',
                    'results_20220613_scale_any_24min_ilp_1in6_0.25sec_alpha0_novgg_run2', '*.log'))

cdf_cutoff_value = 250

response_times_all = {}
for file in files:
    response_times = []
    with open(file, mode='r') as rf:
        for line in rf.readlines():
            # if 'requests->response time' in line:
            #     response_times.append(line)
            if 'response time=' in line:
                response_time = float(line.split('response time=')[1].split(',')[0])*1000
                if response_time < cdf_cutoff_value:
                    response_times.append(response_time)
    response_times_all[file] = response_times

    model = file
    # getting data of the histogram
    count, bins_count = np.histogram(response_times_all[model], bins=50)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
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
    plt.plot(bins_count[1:], cdf, label=label, marker=marker)

plt.xticks(np.arange(0, 260, 50), fontsize=14)
plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=14)
plt.xlabel('Response Time (ms)', fontsize=14)
plt.ylabel('Ratio of Requests', fontsize=14)
plt.grid(linestyle='--')
plt.legend(loc='upper center', bbox_to_anchor=(
    0.5, 1.10), ncol=5, prop={"size": 9})
plt.savefig('cdf.pdf', dpi=500)


print(files)
print(response_times)
