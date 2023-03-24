import numpy as np
import matplotlib.pyplot as plt


total_models = 9

counts = np.ones(total_models)
with open('distribution.txt', mode='w') as wf:
    distribution = counts / sum(counts)
    for model in range(total_models):
        wf.write(f'{distribution[model]}\n')

    print(f'ratio of requests for each model: {distribution}')
    
    plt.plot(distribution)
    plt.xlabel('Model architecture')
    plt.ylabel('Ratio of requests')
    plt.savefig('distribution.png')
