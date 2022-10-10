import copy
import random
import numpy as np
import gym
from gym import spaces


class SchedulingEnv(gym.Env):
    """Scheduling Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, training_steps, n_jobs=4, m_servers=2):
        super(SchedulingEnv, self).__init__()

        self.n_jobs = n_jobs
        self.m_servers = m_servers
        self.training_steps = training_steps

        # self.action_space = spaces.MultiDiscrete([n_jobs, m_servers])
        # we can either fail the request (0) or place it on one of the servers (1...m)
        self.action_space = spaces.Discrete(self.m_servers + 1)

        # the shape of obs space is self.m_servers+1 since it needs to store which model the
        # m_servers are serving, and the +1 is for observation of the next job to schedule
        self.observation_space = spaces.Box(low=0, high=self.n_jobs, shape=(self.m_servers + 1,),
                                            dtype=np.uint8)

        # initially, all servers are assigned 0 (no model assigned)
        self.state = np.zeros(self.m_servers, dtype=np.uint8)

        self.request_trace = []
        for i in range(training_steps):
            self.request_trace.append(random.randint(1, self.n_jobs))

        self.requests = copy.deepcopy(self.request_trace)

        # # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=np.array([-1.0, 2.0]), high=np.array([1.0, 2.0]))

    def step(self, action):
        # if action -> 0, we don't do anything (the request fails)
        # if action is between 1 and m, we serve the model from the server number "action"
        if len(self.requests) == 0:
            self.remake()
            self.requests = copy.deepcopy(self.request_trace)

        model = self.requests.pop()

        reward = 0
        if action == 0:
            reward = 0
        else:
            # since servers are from 0 to m-1 instead of 1 to m, we subtract 1 from action
            selected_server = action - 1
            if self.state[selected_server] == model:
                reward = 5
            elif self.state[selected_server] == 0:
                reward = 1
            elif self.state[selected_server] != 0:
                reward = 0.5
            self.state[selected_server] = model

        if len(self.requests) == 0:
            done = True
        else:
            done = False

        observation = np.array(self.state)
        if len(self.requests) > 0:
            observation = np.append(observation, self.requests[-1])
        else:
            observation = np.append(observation, 0)
        return observation, reward, done, {}

    def reset(self):
        self.requests = copy.deepcopy(self.request_trace)
        observation = np.zeros(self.m_servers+1, dtype=np.uint8)
        return observation  # reward, done, info can't be included

    def remake(self):
        self.request_trace = []
        for i in range(self.training_steps):
            self.request_trace.append(random.randint(1, self.n_jobs))
        return

    def render(self, mode='human'):
        print('State: ' + str(self.state))
        print('Requests: ' + str(self.requests))
        print(str(len(self.requests)) + ' requests left.')
        print()
        return

    def close(self):
        return
