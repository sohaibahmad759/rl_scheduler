import glob
import matplotlib.pyplot as plt

blis_accuracies = [70.0793, 78.2145, 71.0851, 77.2056, 77.8918]
blis_throughput = [0.98, 0.881, 0.965, 0.901, 1]
blis_slo_violation_ratio = [0.096646526, 0.115768104, 0.058334844, 0.081398964, 0.030438559]

labels = ['Clipper++ (High Throughput)', 'Clipper++ (High Accuracy)',
        'INFaaS-Instance', 'INFaaS-Accuracy', 'AccScale']
colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C']
hatch = ['/', '\\', '|', 'xx', '+']

xticks = ['1', '2', '3', '4', '5']

fig, ax = plt.subplots(1, 3, figsize=(17,5))
ax[1].grid(linestyle='--', axis='y')
ax[1].bar(xticks, blis_accuracies, color=colors, hatch=hatch, label=labels)
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
ax[0].bar(xticks, blis_throughput, color=colors, hatch=hatch, label=labels)
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
ax[2].bar(xticks, blis_slo_violation_ratio, color=colors, hatch=hatch, label=labels)
# plt.autoscale()
fig.tight_layout(pad=1.5)
plt.legend(loc='upper center', bbox_to_anchor=(-1.05, 1.55), ncol=3, fontsize=25)
# plt.savefig('blis_accuracies_throughput.pdf', dpi=500, bbox_inches='tight')
# plt.savefig('blis_slo_violation_ratio.pdf', dpi=500, bbox_inches='tight')
plt.savefig('blis_accuracies_throughput_slo.pdf', dpi=500, bbox_inches='tight')


# fig, ax = plt.subplots()
# plt.grid(linestyle='--', axis='y')
# plt.bar([1, 2, 3, 4, 5], bursty_accuracies, color=colors, hatch=hatch, label=labels)
# ax.set_axisbelow(True)
# plt.xlabel('Strategy', fontsize=20)
# plt.ylabel('Effective Accuracy (%)', fontsize=20)
# # axs[1].set_xticks(rotation=30)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# # plt.set_xticks([1, 2, 3, 4, 5])
# # axs[1].set_yticks(fontsize=25)
# plt.ylim(50, 85)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fontsize=12.5)
# plt.savefig('bursty_accuracies.pdf', dpi=500, bbox_inches='tight')

# # Batching comparison graphs
# batching_xticks = ['1', '2', '3', '4']
# batching_hatch = hatch
# del batching_hatch[-2]
# batching_colors = colors
# del batching_colors[-2]

# fig, ax = plt.subplots(1, 2, figsize=(15,5))
# ax[0].grid(linestyle='--', axis='y')
# ax[0].bar(batching_xticks, batching_throughputs, color=batching_colors,
#         hatch=batching_hatch, label=batching_labels)
# ax[0].set_axisbelow(True)
# ax[0].set_xlabel('Strategy', fontsize=30)
# ax[0].set_ylabel('Normalized Throughput', fontsize=27.5)
# ax[0].set_ylim(0.5, 1.05)
# ax[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# ax[0].tick_params(axis='both', which='major', labelsize=25)

# ax[1].grid(linestyle='--', axis='y')
# ax[1].bar(batching_xticks, batching_slo_violation_ratios, color=batching_colors,
#         hatch=batching_hatch, label=batching_labels)
# ax[1].set_axisbelow(True)
# ax[1].set_xlabel('Strategy', fontsize=27.5)
# ax[1].set_ylabel('SLO Violation Ratio', fontsize=30)
# ax[1].set_ylim(0.0, 0.2)
# ax[1].tick_params(axis='both', which='major', labelsize=25)

# fig.tight_layout(pad=1.5)
# plt.legend(loc='upper center', bbox_to_anchor=(-0.2, 1.55), ncol=2, fontsize=30)
# plt.savefig('batching_comparison.pdf', dpi=500, bbox_inches='tight')