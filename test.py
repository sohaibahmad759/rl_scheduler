import time
import sys
import argparse
import random
import numpy as np
from stable_baselines3 import PPO
from scheduling_env import SchedulingEnv


def getargs():
    parser = argparse.ArgumentParser(description='Test scheduler on the simulated BLIS environment.')
    parser.add_argument('--random_runtimes', '-r', required=False,
                        dest='random_runtimes', action='store_true',
                        help='Initializes random runtimes if used. Otherwise, uses static runtimes.')
    # parser.add_argument('--random_runtimes', required=False, default=False, type=bool,
    #                     dest='random_runtimes', help='Whether to randomize runtimes. If not, uses static runtimes.')
    parser.add_argument('--test_steps', '-t', required=False, default=1000,
                        dest='test_steps', help='Number of steps to test for. Default value is 1000')
    parser.add_argument('--action_size', '-a', required=False, default=15,
                        dest='action_size', help='Number of scheduling changes to make in each iteration. ' +
                        'Default value is 15')
    parser.add_argument('--window_length', '-w', required=False, default=10,
                        dest='window_length', help='The number of steps to look out into the future to ' +
                        'calculate the reward of an action. Default value is 10')
    parser.add_argument('--job_scheduling', '-js', required=True, choices=['1', '2', '3', '4'],
                        dest='job_sched_algo', help='The job scheduling algorithm. Select a number:\n' +
                        '1 - Random. 2 - Round robin. 3 - Earliest Finish Time with FIFO. ' +
                        '4 - Latest Finish Time with FIFO')
    parser.set_defaults(random_runtimes=False)

    return parser.parse_args()


def main(args):
    # TODO: should be able to fix seed for random, so that results are reproducible (local scheduler)
    model_training_steps = 8000
    testing_steps = args.test_steps
    action_group_size = args.action_size
    reward_window_length = args.window_length

    modes = ['random', 'static', 'rl', 'lfu']
    mode = modes[1]

    env = SchedulingEnv(trace_dir='traces/twitter/', job_sched_algo=int(args.job_sched_algo),
                        action_group_size=action_group_size, reward_window_length=reward_window_length,
                        random_runtimes=args.random_runtimes)

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
            action[0] = i % env.n_executors
        elif mode == 'rl':
            if i < 5:
                action = env.action_space.sample()
                for j in range(len(action)):
                    action[j] = 1
                action[0] = i % env.n_executors
            else:
                action, _states = model.predict(observation)
        elif mode == 'lfu':
            action = env.action_space.sample()
            if i < 5:
                for j in range(1,5):
                    action[j] = 1
                for j in range(5,len(action)):
                    action[j] = 0
                action[0] = i % env.n_executors
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
            # print('State: {}'.format(env.state))
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


if __name__=='__main__':
    main(getargs())
