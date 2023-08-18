import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import scipy

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


q_runtimes = np.zeros((1, 9))

num_trials = 20
starting_q = 1
maximum_q = 17

for q in range(starting_q, maximum_q+1):
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

    # ----------------- Query Types -----------------
    runtimes = []
    for i in range(num_trials):
        start_time = time.time()
        actions = ilp.run(observation, env.n_accelerators, env.max_no_of_accelerators,
                          num_isi=q, variants_per_model=3)
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

        print(f'q: {q}, trial: {i} runtime: {runtime}')
    mean, clm_lower, clm_upper = mean_confidence_interval(runtimes, confidence=0.95)
    q_runtimes = np.vstack((q_runtimes, np.array([q, np.median(runtimes),
                                                  np.min(runtimes), np.max(runtimes),
                                                  np.quantile(runtimes, 0.25),
                                                  np.quantile(runtimes, 0.75),
                                                  mean, clm_lower, clm_upper])))
# ----------------- Query Types -----------------

q_runtimes = q_runtimes[1:]
print(f'q_runtimes: {q_runtimes}')

df_q_runtimes = pd.DataFrame(q_runtimes, columns=['query_types', 'median_runtime',
                                                  'min_runtime', 'max_runtime',
                                                  '25th_pct_runtime', '75th_pct_runtime',
                                                  'mean_runtime', 'clm_lower', 'clm_upper'])
print(f'df_q_runtimes: {df_q_runtimes}')

df_q_runtimes.to_csv('profiling/ilp/profiled/query_types/query_types.csv')

