import time
import sys
import argparse
import random
import numpy as np
from stable_baselines3 import PPO
from scheduling_env import SchedulingEnv


if __name__=='__main__':

    # TODO: should be able to fix seed for random, so that results are reproducible (local scheduler)

    parser = argparse.ArgumentParser(description='Test scheduler on the simulated BLIS environment.')
    parser.add_argument('--static_runtimes', dest='random_runtimes', action='store_false', help='Initializes static runtimes if used. Otherwise, uses random runtimes.')
    parser.set_defaults(random_runtimes='True')

    args = parser.parse_args()
    print(args)

    model_training_steps = 8000
    testing_steps = 1000
    action_group_size = 15
    reward_window_length = 10

    modes = ['random', 'static', 'rl', 'lfu']
    mode = modes[1]

    env = SchedulingEnv(trace_dir='traces/synthetic/', action_group_size=action_group_size,
                        reward_window_length=reward_window_length, random_runtimes=args.random_runtimes)

    policy_kwargs = dict(net_arch=[128, 128, dict(pi=[128, 128, 128],
                                        vf=[128, 128, 128])])
    # model = PPO('MlpPolicy', env, verbose=2)
    model_name = 'train_' + str(model_training_steps) + '_action_size_' + str(action_group_size) + \
                    '_window_' + str(reward_window_length)
    model = PPO.load('saved_models/' + model_name)
    print(model)

    print()
    print('--------------------------')
    if mode == 'random':
        print('Testing with random actions')
    elif mode == 'static':
        print('Testing with static allocation of 1 predictor for each type')
    elif mode == 'rl':
        print('Testing with trained model: {}'.format(model_name))
    elif mode == 'lfu':
        print('Testing with LFU (Least Frequently Used) baseline')
    else:
        print('Undefined mode, exiting')
        sys.exit(0)
    print('--------------------------')
    print()

    logfile_name = 'log_' + mode + '.txt'
    logfile = open('logs/' + logfile_name, mode='w')

    rl_reward = 0
    observation = env.reset()
    failed_requests = 0
    total_requests = 0
    start = time.time()
    for i in range(testing_steps):
        if mode == 'random':
            action = env.action_space.sample()
        elif mode == 'static':
            action = env.action_space.sample()
            for j in range(1,5):
                action[j] = 1
            for j in range(5,len(action)):
                action[j] = 0
            action[0] = i % 5
        elif mode == 'rl':
            if i < 5:
                action = env.action_space.sample()
                for j in range(len(action)):
                    action[j] = 1
                action[0] = i % 5
            else:
                action, _states = model.predict(observation)
        elif mode == 'lfu':
            action = env.action_space.sample()
            if i < 5:
                for j in range(1,5):
                    action[j] = 1
                for j in range(5,len(action)):
                    action[j] = 0
                action[0] = i % 5
            else:
                # there are actually 2 sub-actions:
                # (i) stealing from an ISI, (ii) giving to another ISI
                missed_ratios = observation[:-1,-1]/observation[:-1,-2]
                # print(observation)
                receiving_isi = np.argmax(missed_ratios)
                # print('Receive candidate: {}'.format(receiving_isi))
                losing_acc_found = False
                losing_candidates = observation[:-1,-2]
                # while not losing_acc_found:
                losing_isi = np.argmin(observation[:-1,-2])
                # print('Losing ISI: {}'.format(losing_isi))
                losing_acc = np.argmax(observation[losing_isi,1:5])
                # print('Losing accelerator: {}'.format(losing_acc))
                # print('Observation: {}'.format(observation))
                # print(observation[losing_isi, losing_acc])
                if observation[losing_isi,losing_acc] == 0:
                    losing_acc = 0
                else:
                    # action[losing_acc] needs to be > 1
                    action = env.action_space.sample()
                    action[0] = losing_isi
                    action[1:5] = observation[losing_isi, 0:4]
                    action[losing_acc+1] -= 1
                    observation, reward, done, info = env.step(action)
                receiving_acc = losing_acc
                action = env.action_space.sample()
                action[0] = receiving_isi
                action[1:5] = observation[receiving_isi, 0:4]
                # action[receiving_acc] needs to be < 10
                if action[receiving_acc+1] >= 10:
                    continue
                action[receiving_acc+1] += 1
                # print(observation)
        if i % 100 == 0:
            print('Testing step: {} of {}'.format(i, testing_steps))
            print('Action taken: {}'.format(action))
        observation, reward, done, info = env.step(action)
        # read failed and total requests once every 5 actions
        if (i-1) % action_group_size == 0:
            failed_requests += np.sum(observation[:,-1])
            total_requests += np.sum(observation[:,-2])
            logfile.write(str(total_requests) + ',' + str(failed_requests) + '\n')
        rl_reward += reward
        env.render()
    end = time.time()
    print('Reward from RL model: ' + str(rl_reward))
    print('Requests failed by RL model: ' + str(failed_requests))
    print('Total requests: ' + str(total_requests))
    print('Percentage of requests failed: {}%'.format(failed_requests/total_requests*100))
    print('Test time: {} seconds'.format(end-start))
    print('Requests added: {}'.format(env.simulator.requests_added))
    logfile.close()
