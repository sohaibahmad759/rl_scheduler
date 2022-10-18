import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logfile_list = sorted(
    glob.glob(os.path.join('..', 'logs', 'throughput_per_model', 'ilp_alpha', '10acc', '*.csv')))

models = 6
alphas = len(logfile_list)

x = np.zeros((models, alphas))
y = np.zeros((models, alphas))
print(x.shape)

# Repeat for different values of alpha
idx = 0
for logfile in logfile_list:
    print(logfile)
    alpha = logfile.split('/')[-1].rstrip('.csv')
    print('alpha:', alpha)

    df = pd.read_csv(logfile)

    # we want to figure out throughput per model
    start_idx = 2
    print(df.shape)
    requests = df.iloc[:, 2::4]
    succesful = df.iloc[:, 3::4]
    accuracy = df.iloc[:, 5::4]

    total_accuracy = accuracy.sum(axis=0).values

    total_requests_per_model = requests.sum(axis=0).values
    total_successful_per_model = succesful.sum(axis=0).values

    overall_throughput_per_model = total_successful_per_model / total_requests_per_model *  100

    effective_accuracy_per_model = total_accuracy / total_successful_per_model

    # print(total_requests_per_model)
    # print(total_successful_per_model)
    print(overall_throughput_per_model)

    # Now, get accuracy in for each model similarly
    print(effective_accuracy_per_model)

    for model in range(models):
        x[model, idx] = effective_accuracy_per_model[model]
        y[model, idx] = overall_throughput_per_model[model]

    idx += 1

# Plot
markers = ['o', 'v', '^', '*', 's', 'h']
y[:,-1] -= 20
for model in range(models):
    print('x[{}]: {}'.format(model, x[model]))
    print('y[{}]: {}'.format(model, y[model]))
    plt.scatter(x[model], y[model], label=model, marker=markers[model])
print('x:', x)
print('y:', y)
plt.show()

# If throughput values are too high, we can subtract a constant factor based on
# ILP throughput value (or from the overall throughput value graph as baseline)
