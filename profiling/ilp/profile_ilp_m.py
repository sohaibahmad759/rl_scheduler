import os
import sys
import time
import logging
import numpy as np
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)

from algorithms.ilp_profiling import Ilp
from core.scheduling_env import SchedulingEnv
from core.utils import mean_confidence_interval


action_group_size = 15
logging_level = logging.ERROR
reward_window_length = 10
allocation_window = 1000
profiling_data = 'profiling/aggregate_profiled.csv'
allowed_variants_path = 'traces/model_variants/profiling/d'


m_runtimes = np.zeros((1, 9))

num_trials = 20
query_types = 5
# starting_m = 50
# maximum_m = 500
starting_m = 5
maximum_m = 6

for m in range(starting_m, maximum_m+1, 50):
    # trace_path = f'traces/twitter/asplos/zipf_exponential/profiling/q/{q}'
    trace_path = f'traces/twitter/asplos/zipf_exponential/profiling/d'
    job_scheduling = 'canary_routing'
    model_assignment = 'ilp'

    env = SchedulingEnv(trace_dir=trace_path, job_sched_algo=job_scheduling,
                        action_group_size=action_group_size, logging_level=logging_level,
                        reward_window_length=reward_window_length,
                        random_runtimes=False, fixed_seed=10,
                        allocation_window=allocation_window,
                        model_assignment=model_assignment,
                        batching=True, batching_algo='accscale',
                        profiling_data=profiling_data,
                        allowed_variants_path=allowed_variants_path,
                        max_batch_size=None, ewma_decay=1.6,
                        infaas_slack=None, infaas_downscale_slack=None)


    ilp = Ilp(allocation_window=allocation_window, beta=0.4, logging_level=logging.WARN,
            profiling_mode=True)

    ilp.set_simulator(env.simulator)

    action = env.action_space.sample()
    for i in range(10):
        observation, _, _, _ = env.step(action)

    # ----------------- Model Variants -----------------
    runtimes = []
    for i in range(num_trials):
        start_time = time.time()
        variants_per_model = int(m / query_types)
        actions = ilp.run(observation, env.n_accelerators, env.max_no_of_accelerators,
                          num_isi=query_types, variants_per_model=variants_per_model)
        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)

        for j in range(env.n_executors):
            isi = j % env.n_executors
            
            if actions is None:
                action = env.simulator.null_action(env.action_space.sample(), 1)
            else:
                action = env.action_space.sample()
                action[0] = isi
                action[1:5] = actions[isi]
            observation, _, _, _ = env.step(action)
        # env.render()

        print(f'm: {m}, trial: {i} runtime: {runtime}')
    mean, clm_lower, clm_upper = mean_confidence_interval(runtimes, confidence=0.95)
    m_runtimes = np.vstack((m_runtimes, np.array([m, np.median(runtimes),
                                                  np.min(runtimes), np.max(runtimes),
                                                  np.quantile(runtimes, 0.25),
                                                  np.quantile(runtimes, 0.75),
                                                  mean, clm_lower, clm_upper])))
    # ----------------- Model Variants -----------------

m_runtimes = m_runtimes[1:]
print(f'm_runtimes: {m_runtimes}')

df_m_runtimes = pd.DataFrame(m_runtimes, columns=['model_variants', 'median_runtime',
                                                  'min_runtime', 'max_runtime',
                                                  '25th_pct_runtime', '75th_pct_runtime',
                                                  'mean_runtime', 'clm_lower', 'clm_upper'])
print(f'df_m_runtimes: {df_m_runtimes}')

df_m_runtimes.to_csv('profiling/ilp/profiled/model_variants/model_variants.csv')

