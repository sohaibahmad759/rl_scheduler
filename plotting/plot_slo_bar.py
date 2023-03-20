import os
import matplotlib.pyplot as plt


logfile_list = [
                # '../logs/throughput/selected_asplos/infaas_accuracy_300ms.csv',
                # '../logs/throughput/selected_asplos/clipper_ht_300ms.csv',
                # '../logs/throughput/selected_asplos/clipper_optstart_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_aimd_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_asb_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_nexus_300ms.csv',
                '../logs/throughput/selected_asplos/proteus_aimd_300ms.csv',
                '../logs/throughput/selected_asplos/proteus_nexus_300ms.csv',
                '../logs/throughput/selected_asplos/proteus_300ms.csv',
                # '../logs/throughput/selected_asplos/sommelier_uniform_asb_300ms.csv'
                ]

algorithms = [
              'AIMD',
              'Nexus',
              'Proteus',
              ]
slo_violations = [0.1314, 0.0845, 0.002722552]

colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2']

plt.xlabel('Batching Algorithm', fontsize=15)
plt.ylabel('SLO Violation Ratio', fontsize=15)

plt.bar(algorithms, slo_violations, color=colors)
plt.savefig(os.path.join('..', 'figures', 'slo_batching.pdf'), dpi=500, bbox_inches='tight')
