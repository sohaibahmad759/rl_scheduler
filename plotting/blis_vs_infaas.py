import os
import numpy as np
import matplotlib.pyplot as plt

filename = 'blis_vs_infaas.pdf'

blis_accuracy = [72.61937473, 79.13659519, 79.62678269, 79.76759679,
                 79.96766141, 79.99]

blis_throughput = np.array([92.40168539, 89.92977528, 89.02808989, 87.03835616,
                   82.69662921, 80.34269663])

infaas_accuracy = [73.3079, 76.3319, 78.3619]
infaas_throughput = np.array([83.93611111, 81.2914, 72.475])

plt.plot(blis_accuracy, blis_throughput, label='AccScale', marker='^')
plt.plot(infaas_accuracy, infaas_throughput, label='INFaaS', marker='v')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)
plt.xlabel('Effective Accuracy (%)', fontsize=14)
plt.ylabel('Normalized Throughput (%)', fontsize=14)
plt.xticks(np.arange(70, 83, 2), fontsize=14)
plt.yticks(np.arange(60, 101, 5), fontsize=14)
plt.grid(linestyle='--')
plt.savefig(os.path.join('..', 'figures', 'paper_draft_figures', filename), dpi=500)
