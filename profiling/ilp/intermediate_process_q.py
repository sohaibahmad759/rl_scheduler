import numpy as np
import pandas as pd


df = pd.read_csv('profiling/ilp/profiled/query_types/trials.csv', header=None,
                     names=['q', 'trial', 'runtime'])
# print(df)
df = df.applymap(lambda x: float(x.split(':')[-1]))
# print(df)

q_runtimes = np.zeros((1, 6))
q_values = df['q'].values

for q in range(int(min(q_values)), int(max(q_values)+1)):
    part_df = df[df['q'] == q]

    runtimes = part_df['runtime'].values

    q_runtimes = np.vstack((q_runtimes, np.array([q, np.median(runtimes),
                                                  np.min(runtimes), np.max(runtimes),
                                                  np.quantile(runtimes, 0.25),
                                                  np.quantile(runtimes, 0.75)])))
    
q_runtimes = q_runtimes[1:]
df_q_runtimes = pd.DataFrame(q_runtimes, columns=['query_types', 'median_runtime',
                                                  'min_runtime', 'max_runtime',
                                                  '25th_pct_runtime', '75th_pct_runtime'])
df_q_runtimes.to_csv('profiling/ilp/profiled/query_types/query_types.csv')
