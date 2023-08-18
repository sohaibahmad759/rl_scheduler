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

from algorithms.ilp import Ilp
from core.scheduling_env import SchedulingEnv
from core.utils import mean_confidence_interval


action_group_size = 15
logging_level = logging.ERROR
reward_window_length = 10
allocation_window = 1000
profiling_data = 'profiling/aggregate_profiled.csv'
# allowed_variants_path = 'traces/model_variants/profiling/d'
allowed_variants_path = 'traces/model_variants'

# trace_path = 'traces/twitter/asplos/zipf_exponential/profiling/d'
trace_path = 'traces/twitter/asplos/zipf_exponential/300'
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


num_trials = 20
accelerator_types = 4
accelerators_per_type_list = np.arange(45, 46, 5)
d_runtimes = np.zeros((1, 9))

for accelerators_per_type in accelerators_per_type_list:
    runtimes = []
    for i in range(num_trials):
        start_time = time.time()
        actions = ilp.run(observation, accelerator_types, accelerators_per_type)
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

        print(f'accelators_per_type: {accelerators_per_type}, trial: {i} runtime: {runtime}')
    mean, clm_lower, clm_upper = mean_confidence_interval(runtimes, confidence=0.95)
    d_runtimes = np.vstack((d_runtimes, np.array([accelerators_per_type*accelerator_types, np.median(runtimes),
                                                  np.min(runtimes), np.max(runtimes),
                                                  np.quantile(runtimes, 0.25),
                                                  np.quantile(runtimes, 0.75),
                                                  mean, clm_lower, clm_upper])))

d_runtimes = d_runtimes[1:]
print(f'd_runtimes: {d_runtimes}')

df_d_runtimes = pd.DataFrame(d_runtimes, columns=['num_accelerators', 'median_runtime',
                                                  'min_runtime', 'max_runtime',
                                                  '25th_pct_runtime', '75th_pct_runtime',
                                                  'mean_runtime', 'clm_lower', 'clm_upper'])
print(f'df_d_runtimes: {df_d_runtimes}')

df_d_runtimes.to_csv('profiling/ilp/profiled/devices/devices.csv')

