import os
import sys
import glob
import logging
import time
import gym
import numpy as np
from core.simulator import Simulator


class SchedulingEnv(gym.Env):
    """Scheduling Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, trace_dir, job_sched_algo, action_group_size,
                 reward_window_length=10, random_runtimes=False, fixed_seed=0,
                 allocation_window=1000, model_assignment='', batching=False,
                 batching_algo=None, profiling_data=None):
        super(SchedulingEnv, self).__init__()

        logging.basicConfig(level=logging.INFO)
        np.set_printoptions(threshold=sys.maxsize)

        logfile_name = os.path.join('logs', 'reward', str(time.time()) + '.txt')
        self.logfile = open(logfile_name, mode='w')
        
        # TODO: move all parameters to a config file or cmd line argument, train.py should read that
        self.n_executors = len(glob.glob(os.path.join(os.getcwd(), trace_dir, '*.txt')))
        self.n_accelerators = 4
        self.n_qos_levels = 1
        self.action_group_size = action_group_size

        self.allocation_window = allocation_window
        
        # if we make this a vector, can have heterogeneous no. of accelerators for each type
        self.max_no_of_accelerators = 10
        self.max_runtime = 1000

        self.model_assignment = model_assignment

        # max total predictors for each accelerator type (CPU, GPU, VPU, FPGA) respectively
        self.predictors_max = self.max_no_of_accelerators * np.ones(self.n_accelerators * self.n_qos_levels)
        self.total_predictors = self.max_no_of_accelerators * np.ones(self.n_accelerators)
        
        # simulator environment that our RL agent will interact with
        self.simulator = Simulator(trace_path=trace_dir, mode='debugging',
                                   job_sched_algo=job_sched_algo,
                                   max_acc_per_type=self.max_no_of_accelerators,
                                   predictors_max=self.total_predictors,
                                   n_qos_levels=self.n_qos_levels,
                                   random_runtimes=random_runtimes,
                                   fixed_seed=fixed_seed,
                                   batching=batching,
                                   model_assignment=model_assignment,
                                   batching_algo=batching_algo,
                                   profiling_data=profiling_data)

        # number of steps that we play into the future to get reward
        # Note: this is a tunable parameter
        self.K_steps = reward_window_length

        # ----- with 1 QoS level -----
        # features = (CPU_predictors, GPU_predictors, VPU_predictors, FPGA_predictors,
        #             CPU_runtime,    GPU_runtime,    VPU_runtime,    FPGA_runtime,
        #             qos_level_1_requests, qos_level_1_missed,
        #             total_requests, failed_requests)    --> 1d array
        # ----- with 10 QoS levels -----
        # features = (CPU_predictorsx10, GPU_predictorsx10, VPU_predictorsx10, FPGA_predictorsx10,
        #             CPU_runtimex10,    GPU_runtimex10,    VPU_runtimex10,    FPGA_runtimex10,
        #             qos_level_1_requests, qos_level_1_missed, ..., qos_level_10_requests, qos_level_10_missed,
        #             QoS_total_requestsx10, QoS_failed_requestsx10, total_requests, failed_requests)    --> 1d array
        self.features = (self.n_accelerators * 2) * self.n_qos_levels + 2*self.n_qos_levels + 2

        # defining the upper bounds for each feature
        # high_1d = [10, 10, 10, 10, 1000, 1000, 1000, 1000, 10000, 10000]
        high_1d = np.ones((self.features))
        high_1d[-1] = 10000
        high_1d[-2] = 10000

        predictors_end_idx = self.n_accelerators * self.n_qos_levels
        runtimes_end_idx = self.n_accelerators * self.n_qos_levels * 2

        high_1d[:predictors_end_idx] = self.max_no_of_accelerators * np.ones((self.n_accelerators*self.n_qos_levels))
        high_1d[predictors_end_idx:runtimes_end_idx] = self.max_runtime * np.ones((self.n_accelerators*self.n_qos_levels))
        
        # we convert it into a 2d array by duplicating the same row, n_executors times
        high_2d = np.tile(high_1d, (self.n_executors+1, 1))

        # in the observation space, we have self.n_executors rows for each executor, and
        #   1 extra row at the end to denote the remaining available predictors in the system
        #   for that row, the rest of the columns will stay 0 throughout
        self.observation_space = gym.spaces.Box(low=np.zeros((self.n_executors+1, self.features)), high=high_2d,
                                                shape=(self.n_executors+1, self.features), dtype=np.int)

        # self.action_space = gym.spaces.Box(low=np.zeros((self.n_executors, self.n_accelerators)),
        #                             high=10*np.ones((self.n_executors, self.n_accelerators)), 
        #                             shape=((self.n_executors, self.n_accelerators)), dtype=np.int)
        action_space_max = np.concatenate(([self.n_executors], self.predictors_max))
        self.action_space = gym.spaces.MultiDiscrete(action_space_max)

        # initializing a random state
        self.state = self.reset()
        
        # the runtimes need to be accurate information that the simulator uses to run requests
        # otherwise, RL agent won't be able to learn
        self.populate_runtime_info()

        # initializing total and failed requests to be zero
        self.state[:, -2:] = np.zeros((self.n_executors+1, 2))

        self.actions_taken = 0
        self.clock = 0

        # State should have following components:
        # 1. number of ISIs (or executors)
        # 2. for each executor, number and AccType of predictors
        # 3. for each executor, number of total requests within last time period
        # 4. for each executor, number of failed requests within last time period
        # 5. profiled information! Should we include profiled info in observation/state or
        #    have a different way to represent it? ***It stays static over time***
        # 6. QoS?

    
    def step(self, action):
        # apply action on current state, update the state
        # we enforce a system wide constraint that total number of accelerators assigned cannot
        # exceed total number of accelerators in the system, so we scale the assignments proportionally
        # assignment = np.round(np.nan_to_num(action / np.sum(action, axis=0)) * self.predictors_max)
        # self.state[:, 1:5] = assignment

        logging.debug('')
        logging.debug('--------------------------')
        logging.debug('Applying action:{}'.format(action))
        logging.debug('--------------------------')
        logging.debug('')

        if not(self.model_assignment == 'ilp'):
            applied_assignment = self.simulator.apply_assignment_vector(action)
        # self.state[action[0], 0:4] = action[1:]
        # self.state[action[0], 0:4] = applied_assignment
            self.state[action[0], 0:self.n_accelerators*self.n_qos_levels] = applied_assignment
            logging.debug(self.state)

        available_predictors = self.simulator.get_available_predictors()
        # self.failed_requests += self.simulator.get_failed_requests()
        total_requests_arr = self.simulator.get_total_requests_arr()
        failed_requests_arr = self.simulator.get_failed_requests_arr()
        completed_requests = self.simulator.completed_requests
        qos_stats = self.simulator.get_qos_stats()

        self.state[-1, 0:4] = available_predictors
        self.state[:-1, -1] = failed_requests_arr
        self.state[:-1, -2] = total_requests_arr
        qos_start_idx = self.n_accelerators*self.n_qos_levels*2
        qos_end_idx = qos_start_idx + self.n_qos_levels*2
        self.state[:-1, qos_start_idx:qos_end_idx] = qos_stats

        # actions might be from gym.spaces.MultiDiscrete([x,y,z]) where x,y,z are the number
        # of possible discrete choices
        # can check with space.sample() to see possible values

        # IDEA: reward for negative of missed requests
        # simulating missed requests requires modeling the hardware, latency, (batch sizes)
        # we would also need a queue of requests for every predictor
        # do we model the executor? load balancing at the executor level? (can do round robin)
        # if we don't model executor, how are we increasing/decreasing # of predictors?
        # so actions = {increase/decrease # of predictors for an executor (where?),
        #               re-allocate predictor to different hardware,
        #               what others?}
        # reward = self.simulator.evaluate_reward(self.K_steps)
        # self.logfile.write(str(reward) + '\n')

        # TODO: verifying that state is being correctly modified and there is no overflow
        # logging.debug('self.state: {}'.format(self.state))
        # debug_total_assigned = np.sum(self.state[:-1, 0:4], axis=0)
        # debug_expected_assigned = self.predictors_max - self.state[-1, 0:4]
        # if not np.array_equal(debug_total_assigned, debug_expected_assigned):
        #     print('Total not equal')
        #     print('Expected: {}'.format(debug_expected_assigned))
        #     print('Actual: {}'.format(debug_total_assigned))
        

        self.actions_taken += 1
        if self.actions_taken == self.action_group_size:
            # after every 'group' of actions, wait `allocation_window` milliseconds before taking another action
            self.clock += self.allocation_window
            # print(f'scheduling env clock: {self.clock}, simulator clock: {self.simulator.clock}')
            # time.sleep(1)
            self.simulator.reset_request_count()
            self.simulator.simulate_until(self.clock)
            self.actions_taken = 0

        # experiment: show varying levels of performance for agent with different values of K

        observation = self.state

        # read done from simulator (if no more requests then we are done)
        done = self.simulator.is_done()

        # return observation, reward, done, self.failed_requests
        return observation, 0, done, {}

    
    def populate_runtime_info(self):
        predictors_end_idx = self.n_accelerators * self.n_qos_levels
        runtimes_end_idx = self.n_accelerators * self.n_qos_levels * 2

        for i in range(self.n_executors):
            # self.state[i, predictors_end_idx:runtimes_end_idx] = self.simulator.get_runtimes(i)
            # logging.debug(self.simulator.get_runtimes(i))
            logging.debug(self.state)
        return

    
    def trigger_infaas_upscaling(self):
        self.simulator.trigger_infaas_upscaling()

    
    def trigger_infaas_v2_upscaling(self):
        self.simulator.trigger_infaas_v2_upscaling()


    def trigger_infaas_downscaling(self):
        self.simulator.trigger_infaas_downscaling()
    

    def trigger_infaas_v2_downscaling(self):
        self.simulator.trigger_infaas_v2_downscaling()

    
    def reset(self):
        # observation = self.observation_space.sample()
        observation = np.zeros((self.n_executors+1, self.features))
        self.simulator.reset()
        self.state = observation
        self.failed_requests = 0
        self.populate_runtime_info()
        print('Resetting environment')
        return observation

    
    def render(self, mode='human'):
        # print(self.state)
        self.simulator.print_assignment()
        return

    
    def close(self):
        self.logfile.close()
        return
