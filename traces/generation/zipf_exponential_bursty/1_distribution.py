import numpy as np
import matplotlib.pyplot as plt

alpha = 1.001
# alpha = 1.1
# alpha = 1.6
total_models = 9

num_requests = 139689
offset = 320000
# offset = 20000

dist = np.random.zipf(alpha, size=num_requests+offset)
print(f'length: {len(dist)}, dist: {dist}')
exceeding = np.sum(dist <= total_models)

counter = 0
counts = np.zeros(total_models)
with open('distribution.txt', mode='w') as wf:
    for sample in dist:
        if counter >= num_requests:
            print('Succeeded!')
            break
        if sample <= total_models:
            # wf.write(str(sample) + '\n')
            counter += 1
            counts[sample-1] += 1
            # print(f'sample: {sample}, written to file')
        # else:
        #     print(f'sample: {sample}, skipped because sample > total_models')

    print(f'exceeding: {exceeding}')
    print(f'counter: {counter}')
    print(f'ratio of requests for each model: {counts/np.sum(counts)}')

    distribution = counts/np.sum(counts)
    for model in range(total_models):
        wf.write(f'{distribution[model]}\n')
    
    plt.plot(counts/np.sum(counts))
    plt.xlabel('Model architecture')
    plt.ylabel('Ratio of requests')
    plt.savefig('zipf_distribution.png')
    print(np.sum(counts[0:4])/np.sum(counts))
