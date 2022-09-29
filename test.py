import os
import time
import sys
import logging
import argparse
import random
import numpy as np
from stable_baselines3 import PPO
import utils
from scheduling_env import SchedulingEnv
from algorithms.ilp import Ilp
from algorithms.ilp_throughput import IlpThroughput


def getargs():
    parser = argparse.ArgumentParser(description='Test scheduler on the simulated BLIS environment.')
    parser.add_argument('--random_runtimes', '-r', required=False,
                        dest='random_runtimes', action='store_true',
                        help='Initializes random runtimes if used. Otherwise, uses static runtimes.')
    parser.add_argument('--fixed_seed', '-f', required=False,
                        dest='fixed_seed', default=0,
                        help='Fix a seed for random behavior.')
    parser.add_argument('--trace_path', '-p', required=False, default='traces/zipf_static_deadlines/',
                        dest='trace_path', help='Path for trace files. Default is traces/zipf_static/')
    parser.add_argument('--test_steps', '-t', required=False, default=1000,
                        dest='test_steps', help='Number of steps to test for. Default value is 1000')
    parser.add_argument('--action_size', '-a', required=False, default=10,
                        dest='action_size', help='Number of scheduling changes to make in each iteration. ' +
                        'Default value is 15')
    parser.add_argument('--reward_window_length', '-l', required=False, default=10,
                        dest='reward_window_length', help='The number of steps to look out into the future to ' +
                        'calculate the reward of an action. Default value is 10')
    parser.add_argument('--model_assignment', '-ma', required=True, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                        dest='model_asn_algo', help='The model assignment algorithm. Select a number:\n' +
                        '1 - Random. 2 - Static. 3 - Least Frequently Used (LFU). 4 - Load proportional. ' +
                        '5 - RL. 6 - RL with warm start (load proportional). 7 - ILP. 8 - ILP (Max throughput). ' +
                        '9 - INFaaS')
    parser.add_argument('--job_scheduling', '-js', required=True, choices=['1', '2', '3', '4', '5', '6'],
                        dest='job_sched_algo', help='The job scheduling algorithm. Select a number:\n' +
                        '1 - Random. 2 - Round robin. 3 - Earliest Finish Time with FIFO. ' +
                        '4 - Latest Finish Time with FIFO. 5 - INFAAS. 6 - Canary Routing')
    parser.add_argument('--allocation_window', '-w', required=False, default=1000,
                        dest='allocation_window', help='Milliseconds to wait before recalculating allocation. Default is 1000')
    parser.set_defaults(random_runtimes=False)

    return parser.parse_args()


def main(args):
    model_training_steps = 8000
    testing_steps = int(args.test_steps)
    fixed_seed = int(args.fixed_seed)
    action_group_size = int(args.action_size)
    reward_window_length = args.reward_window_length
    allocation_window = int(args.allocation_window)

    model_asn_algos = ['random', 'static', 'lfu', 'load_proportional', 'rl', 'rl_warm',
                        'ilp', 'ilp_throughput', 'infaas']
    model_assignment = model_asn_algos[int(args.model_asn_algo)-1]

    env = SchedulingEnv(trace_dir=args.trace_path, job_sched_algo=int(args.job_sched_algo),
                        action_group_size=action_group_size, reward_window_length=reward_window_length,
                        random_runtimes=args.random_runtimes, fixed_seed=fixed_seed,
                        allocation_window=allocation_window, model_assignment=model_assignment)

    policy_kwargs = dict(net_arch=[128, 128, dict(pi=[128, 128, 128],
                                        vf=[128, 128, 128])])
    # model = PPO('MlpPolicy', env, verbose=2)
    if 'rl' in model_assignment:
        model_name = 'train_' + str(model_training_steps) + '_action_size_' + str(action_group_size) + \
                        '_window_' + str(reward_window_length)
        model = PPO.load('saved_models/' + model_name)
        print(model)

    print()
    print('--------------------------')
    if model_assignment == 'random':
        print('Testing with random actions')
    elif model_assignment == 'static':
        print('Testing with static allocation of 1 predictor for each type')
    elif model_assignment == 'rl':
        print('Testing with trained RL model (cold start): {}'.format(model_name))
    elif model_assignment == 'rl_warm':
        print('Testing with trained RL model (warm start): {}'.format(model_name))
    elif model_assignment == 'lfu':
        print('Testing with LFU (Least Frequently Used) baseline')
    elif model_assignment == 'load_proportional':
        print('Testing with load proportional algorithm')
    elif model_assignment == 'ilp':
        ilp = Ilp(allocation_window=allocation_window)
        ilp_applied = True
        print('Testing with solution given by ILP')
    elif model_assignment == 'ilp_throughput':
        ilp = IlpThroughput(allocation_window=allocation_window)
        ilp_applied = True
        print('Testing with solution given by ILP (Max throughput version)')
    elif model_assignment == 'infaas':
        print('Testing with INFaaS model assignment policy')
    else:
        print('Undefined mode, exiting')
        sys.exit(0)
    print('--------------------------')
    print()

    logfile_name = 'log_' + model_assignment + '.txt'
    logfile = open('logs/' + logfile_name, mode='w')

    rate_logger = logging.getLogger('Rate logger')
    rate_loggerfile = os.path.join('logs', 'throughput', str(time.time()) + '_' +  model_assignment + '.csv')
    rate_logger.addHandler(logging.FileHandler(rate_loggerfile, mode='w'))
    rate_logger.setLevel(logging.DEBUG)
    rate_logger.info('wallclock_time,simulation_time,demand,throughput,capacity')

    rate_logger_per_model =logging.getLogger('Rate logger per model')
    rate_logger_per_model_file = os.path.join('logs', 'throughput_per_model', str(time.time()) + '_' + model_assignment + '.csv')
    rate_logger_per_model.addHandler(
        logging.FileHandler(rate_logger_per_model_file, mode='w'))
    rate_logger_per_model.setLevel(logging.DEBUG)

    rl_reward = 0
    observation = env.reset()
    failed_requests = 0
    total_requests = 0
    start = time.time()
    for i in range(testing_steps):
        if model_assignment == 'random':
            action = env.action_space.sample()
        elif model_assignment == 'static':
            action = env.action_space.sample()
            for j in range(1,5):
                action[j] = 1
            for j in range(5,len(action)):
                action[j] = 0
            action[0] = i % env.n_executors
        elif model_assignment == 'load_proportional':
            # we only need to apply the assignment once at the start
            if i == 0:
                # proportions = np.array([0.30, 0.14, 0.091, 0.066, 0.051, 0.042, 0.035, 0.030, 0.027, \
                #                0.024, 0.022, 0.020, 0.018, 0.016, 0.015, 0.014, 0.013, 0.012, \
                #                0.012, 0.011, 0.010, 0.009])
                proportions = np.array([0.37089186, 0.17539785, 0.11117586, 0.08196145, 0.0635107, \
                                        0.05204432, 0.04448577, 0.03854691, 0.03237666, 0.02960862])
                # we calculate the shares for each model architecture (ISI) and round to integer
                shares = np.rint(proportions * env.n_accelerators * env.max_no_of_accelerators)
                # print('this: {}'.format(env.n_accelerators * env.max_no_of_accelerators))
                # we use a counter to loop through the types of accelerators available, assigning
                # them in round robin fashion
                counter = 0
                for model in range(len(shares)):
                    action = env.action_space.sample()
                    action[0] = model
                    for j in range(1, len(action)):
                        action[j] = 0

                    to_assign = shares[model]
                    while to_assign > 0:
                        action[counter+1] += 1
                        to_assign -= 1
                        counter = (counter + 1) % 4
                    observation, reward, done, info = env.step(action)
                #     print(action)
                #     print(np.sum(action[1:]))
                # print(shares)
                # time.sleep(5)
                # print(np.sum(np.rint(acc_shares)))
                # print(env.n_accelerators * env.max_no_of_accelerators)
        elif model_assignment == 'rl':
            if i < 5:
                action = env.action_space.sample()
                for j in range(len(action)):
                    action[j] = 1
                action[0] = i % env.n_executors
            else:
                action, _states = model.predict(observation)
        elif model_assignment == 'rl_warm':
            if i == 0:
                proportions = np.array([0.30, 0.14, 0.091, 0.066, 0.051, 0.042, 0.035, 0.030, 0.027, \
                               0.024, 0.022, 0.020, 0.018, 0.016, 0.015, 0.014, 0.013, 0.012, \
                               0.012, 0.011, 0.010, 0.009])
                # proportions = np.array([0.49350343,0.16217454, 0.08506038, 0.05360479, 0.03783405, 0.02843459, 0.02187717, 0.01780384, 0.01448217, 0.01235602, 0.01037304, 0.0092348, 0.00851892, 0.0070657,  0.00659322, 0.00568406, 0.00491807, 0.004825, 0.00410913, 0.00432389, 0.00375835, 0.00346484])
                # we calculate the shares for each model architecture (ISI) and round to integer
                shares = np.rint(proportions * env.n_accelerators * env.max_no_of_accelerators)
                # we use a counter to loop through the types of accelerators available, assigning
                # them in round robin fashion
                counter = 0
                for idx in range(len(shares)):
                    action = env.action_space.sample()
                    action[0] = idx
                    for j in range(1, len(action)):
                        action[j] = 0

                    to_assign = shares[idx]
                    while to_assign > 0:
                        action[counter+1] += 1
                        to_assign -= 1
                        counter = (counter + 1) % 4
                    observation, reward, done, info = env.step(action)
            else:
                action, _states = model.predict(observation)
        elif model_assignment == 'lfu':
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
        elif model_assignment == 'ilp' or model_assignment == 'ilp_throughput':
            if ilp.is_simulator_set() is False:
                ilp.set_simulator(env.simulator)

            if ilp_applied == True:
                actions = ilp.run(observation, env.n_accelerators, env.max_no_of_accelerators)

            if actions is None:
                # action = env.action_space.sample()
                # action[0] = 1
                # for j in range(1, len(action)):
                #     action[j] = 
                action = env.simulator.null_action(env.action_space.sample(), 1)
            elif np.sum(actions) == 0:
                # Apply null action
                action = env.action_space.sample()
                for j in range(len(action)):
                    action[j] = 1
            else:
                ilp_applied = False
                # apply actions iteratively
                action = env.action_space.sample()
                isi = i % env.n_executors
                action[0] = isi
                action[1:5] = actions[isi]
                print('action:', action)

                if isi == env.n_executors - 1:
                    ilp_applied = True
            
            # action = env.action_space.sample()
            # for j in range(5, len(action)):
            #     action[j] = 0
        elif model_assignment == 'infaas':
            print()
            env.trigger_infaas_upscaling()
            env.trigger_infaas_downscaling()
            
            num_isi = observation.shape[0] - 1
            # Initially, we give all executors one CPU predictor to start with
            if i < num_isi:
                action = env.action_space.sample()
                action[0] = i
                action[1:5] = [1,0,0,0]
            else:
                # apply null action, as autoscaling is done by now
                action = env.action_space.sample()
                action[0] = 0
                action[1:5] = observation[0, 0:4]
            # print('observation:' + str(observation))
            # print('null action:' + str(action))
            # time.sleep(2)

        if i % 100 == 0:
            print('Testing step: {} of {}'.format(i, testing_steps))
            print('Action taken: {}'.format(action))
            # print('State: {}'.format(env.state))
        observation, reward, done, info = env.step(action)
        # read failed and total requests once every 5 actions
        if (i-1) % action_group_size == 0:
            failed_requests += np.sum(observation[:,-1])
            total_requests += np.sum(observation[:,-2])
            logfile.write(str(total_requests) + ',' +
                          str(failed_requests) + '\n')
            requests_per_model, failed_per_model, accuracy_per_model = env.simulator.get_thput_accuracy_per_model()
            utils.log_thput_accuracy_per_model(rate_logger_per_model, i, requests_per_model,
                                            failed_per_model, accuracy_per_model)
        rl_reward += reward
        env.render()


        utils.log_throughput(rate_logger, observation, i, allocation_window)
        print('observation:', observation)

    end = time.time()
    print('Reward from RL model: ' + str(rl_reward))
    print('Requests failed by RL model: ' + str(failed_requests))
    print('Total requests: ' + str(total_requests))
    print('Percentage of requests failed: {}%'.format(failed_requests/total_requests*100))
    print('Test time: {} seconds'.format(end-start))
    print('Requests added: {}'.format(env.simulator.requests_added))
    logfile.close()

    completed_requests = env.simulator.completed_requests
    sim_time_elapsed = env.simulator.clock / 1000
    overall_throughput = completed_requests / sim_time_elapsed
    print()
    print('Completed requests: {}'.format(completed_requests))
    print('Simulator time elapsed: {}'.format(sim_time_elapsed))
    print('Overall throughput (requests/sec): {}'.format(overall_throughput))

if __name__=='__main__':
    main(getargs())
