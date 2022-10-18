# Takes in a log file and generates numbers for the total system throughput
# and effective accuracy of that run

# Format of log file:
# wallclock_time,simulation_time,[demand_nth_model,throughput_nth_model,normalized_throughput_nth_model,accuracy_nth_model]*n

from bdb import effective
import glob
import numpy as np
import pandas as pd

log_path = '../logs/throughput_per_model/*.csv'

log_files = sorted(glob.glob(log_path))
most_recent_file = log_files[-1]

# We select the most recent log file to analyze
log_file = most_recent_file

# Alternatively, we could provide a direct path to the file we want to analyze
log_file = '../logs/selected/infaas.csv'
log_file = '../logs/selected/acc_scale.csv'

df = pd.read_csv(log_file)

# We have two columns at the start for wallclock time and simulation time
x_offset = 2
# 4 columns for each model family
cols_per_model = 4

num_models = int((df.shape[1] - x_offset) / cols_per_model)
print(f'Number of models: {num_models}')
print()

total_demand = 0
total_throughput = 0
total_accuracy = 0

for model in range(num_models):
    first_col_idx = x_offset + model * cols_per_model
    
    demand_idx = first_col_idx
    throughput_idx = first_col_idx + 1
    accuracy_idx = first_col_idx + 3

    demand = df.iloc[:, demand_idx].values.tolist()
    throughput = df.iloc[:, throughput_idx].values.tolist()
    accuracy = df.iloc[:, accuracy_idx].values.tolist()

    model_demand = sum(demand)
    total_demand += model_demand

    model_throughput = sum(throughput)
    total_throughput += model_throughput

    model_normalized_throughput = model_throughput / model_demand
    
    model_accuracy = sum(accuracy)
    total_accuracy += model_accuracy
    # effective_accuracy = np.divide(np.array(accuracy), np.array(throughput))
    # # print(f'effective accuracy: {effective_accuracy}')
    # effective_accuracy[np.isnan(effective_accuracy)] = 0
    # effective_accuracy = sum(effective_accuracy)
    model_effective_accuracy = model_accuracy / model_throughput

    print(f'Normalized throughput (model {model}): {model_normalized_throughput}')
    print(f'Effective accuracy (model {model}): {model_effective_accuracy}')
    print()

normalized_throughput = total_throughput / total_demand
effective_accuracy = total_accuracy / total_throughput
print(f'Normalized throughput (overall): {normalized_throughput}')
print(f'Effective accuracy (overall): {effective_accuracy}')
print()
