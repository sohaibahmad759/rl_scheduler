import glob
import matplotlib.pyplot as plt

accuracies = [68.79, 77.75, 70.24, 77.9, 77.35]
throughput = [1, 0.8562, 0.9201, 0.8914, 1]
throughput = [0.954181717, 0.894015286, 0.96186466, 0.917962124, 1]
slo_violation_ratio = [0.084512436, 0.196098121, 0.042567984, 0.067648587, 0.02]

labels = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
        'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C']
hatch = ['/', '\\', '|', 'xx', '+']

xticks = ['1', '2', '3', '4', '5']

fig, ax = plt.subplots(1, 3, figsize=(17,5))
ax[1].grid(linestyle='--', axis='y')
ax[1].bar(xticks, accuracies, color=colors, hatch=hatch, label=labels)
ax[1].set_axisbelow(True)
ax[1].set_xlabel('Strategy', fontsize=30)
ax[1].set_ylabel('Effective Accuracy (%)', fontsize=30)
# ax[0].xticks(fontize=20)
# ax.xticks(rotation=30)
# ax.yticks(fontsize=25)
ax[1].set_ylim(50, 80)
ax[1].tick_params(axis='both', which='major', labelsize=25)
# ax.savefig('accuracies.pdf', dpi=500, bbox_inches='tight')

ax[0].grid(linestyle='--', axis='y')
ax[0].bar(xticks, throughput, color=colors, hatch=hatch, label=labels)
ax[0].set_axisbelow(True)
ax[0].set_xlabel('Strategy', fontsize=30)
ax[0].set_ylabel('Normalized Throughput', fontsize=30)
ax[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# ax.xticks(rotation=30)
# ax.yticks(fontsize=25)
ax[0].set_ylim(0.5, 1.05)
# plt.savefig('throughput.pdf', dpi=500, bbox_inches='tight')
ax[0].tick_params(axis='both', which='major', labelsize=25)
# plt.xticks(fontsize=20)

# fig.tight_layout(pad=1.5)

# fig, ax = plt.subplots()
ax[2].grid(linestyle='--', axis='y')
ax[2].set_axisbelow(True)
ax[2].set_xlabel('Strategy', fontsize=30)
ax[2].set_ylabel('SLO Violation Ratio', fontsize=30)
# ax[2].set_xticks(fontsize=25)
ax[2].set_yticks([0, 0.05, 0.10, 0.15])
ax[2].set_ylim([0, 0.15])
ax[2].tick_params(axis='both', which='major', labelsize=25)
ax[2].bar(xticks, slo_violation_ratio, color=colors, hatch=hatch, label=labels)
# plt.autoscale()
fig.tight_layout(pad=1.5)
plt.legend(loc='upper center', bbox_to_anchor=(-1.05, 1.55), ncol=3, fontsize=25)
# plt.savefig('blis_accuracies_throughput.pdf', dpi=500, bbox_inches='tight')
# plt.savefig('blis_slo_violation_ratio.pdf', dpi=500, bbox_inches='tight')
plt.savefig('accuracies_throughput_slo.pdf', dpi=500, bbox_inches='tight')