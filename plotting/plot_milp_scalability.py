import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'profiling/ilp/profiled'

# markers = ['.', 's', 'v', '^', 'x', '+', '*', 'o', '<']
marker = '.'
# markersizes = [7, 3, 4, 4, 5, 6, 5, 5, 5, 5]
markersize = 5

colors = ['#729ECE', '#FF9E4A', '#ED665D', '#AD8BC9', '#67BF5C', '#8C564B',
          '#E377C2', 'tab:olive', 'tab:cyan']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# fig, axs = plt.subplots(3, 1, figsize=(5, 6))
fig, axs = plt.subplots(1, 3, figsize=(6, 2))
plt.subplots_adjust(wspace=0.25, hspace=0.4)


# d_df = pd.read_csv(os.path.join(path, 'devices/devices_all.csv'))
d_df = pd.read_csv(os.path.join(path, 'devices/devices_saved.csv'))

d_df = d_df[d_df['mean_runtime'] <= 60]
x = d_df['num_accelerators'].values
y = d_df['mean_runtime'].values
y_error = d_df['clm_upper'].values - d_df['clm_lower'].values

axs[0].errorbar(x, y, yerr=y_error, marker=marker, markersize=markersize)
# axs[0].plot(x, y, marker=marker, markersize=markersize)
# axs[0].set_xticklabels([])
# axs[0].set_xticks(np.arange(20, 161, 20), fontsize=15)
# axs[0].set_yticks(np.arange(0, 71, 10), fontsize=12)
axs[0].set_xticks(np.arange(0, 161, 40))
axs[0].set_yticks(np.arange(0, 71, 10))
axs[0].set_xlabel('Devices (d)', fontsize=11)
axs[0].set_ylabel('Time (sec)', fontsize=11)
axs[0].grid(True)
axs[0].set_box_aspect(1)
axs[0].tick_params(labelsize=8)


# m_df = pd.read_csv(os.path.join(path, 'model_variants/model_variants.bk.csv'))
m_df = pd.read_csv(os.path.join(path, 'model_variants/model_variants_saved.csv'))

m_df = m_df[m_df['mean_runtime'] <= 60]
x = m_df['model_variants'].values
y = m_df['mean_runtime'].values
y_error = m_df['clm_upper'].values - m_df['clm_lower'].values

axs[1].errorbar(x, y, yerr=y_error, marker=marker, markersize=markersize)
# axs[1].plot(x, y, marker=marker, markersize=markersize)
# axs[1].set_xticklabels([])
# axs[1].set_yticks(np.arange(80, 104, 5), fontsize=12)
# axs[1].set_xticks(np.arange(0, 460, 50), fontsize=15)
# axs[1].set_yticks(np.arange(0, 71, 10), fontsize=12)
axs[1].set_xticks(np.arange(0, 451, 150))
axs[1].set_yticks(np.arange(0, 71, 10))
axs[1].set_xlabel('Model Variants (m)', fontsize=11)
axs[1].set_ylabel('Time (sec)', fontsize=11)
axs[1].grid(True)
axs[1].set_box_aspect(1)
axs[1].tick_params(labelsize=8)

q_df = pd.read_csv(os.path.join(path, 'query_types/query_types_saved.csv'))

q_df = q_df[q_df['median_runtime'] <= 60]
x = q_df['query_types'].values
y = q_df['mean_runtime'].values
y_error = q_df['clm_upper'].values - q_df['clm_lower'].values

# axs[2].plot(x, y, marker=marker, markersize=markersize)
axs[2].errorbar(x, y, yerr=y_error, marker=marker, markersize=markersize)
# axs[2].set_yticks(np.arange(0, 40, 10), fontsize=12)
# axs[2].set_xticks(np.arange(1, 18, 2), fontsize=11)
# axs[2].set_yticks(np.arange(0, 71, 10), fontsize=12)
axs[2].set_xticks(np.arange(0, 19, 3))
axs[2].set_yticks(np.arange(0, 71, 10))
axs[2].set_xlabel('Query Types (q)', fontsize=11)
axs[2].set_ylabel('Time (sec)', fontsize=11)
axs[2].grid(True)
axs[2].set_box_aspect(1)
axs[2].tick_params(labelsize=8)

# axs[2].set_xlabel('Time (min)', fontsize=12)

fig.tight_layout()
plt.savefig(os.path.join('figures', 'asplos', 'milp', 'milp.pdf'),
            dpi=500, bbox_inches='tight')
