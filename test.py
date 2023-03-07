import os
import json
import time
import sys
import logging
import argparse
import numpy as np
from stable_baselines3 import PPO
import core.utils as utils
from core.scheduling_env import SchedulingEnv
from algorithms.clipper import Clipper
from algorithms.ilp import Ilp
from algorithms.ilp_alpha import IlpAlpha
from algorithms.ilp_throughput import IlpThroughput
from core.exceptions import ConfigException


def getargs():
    parser = argparse.ArgumentParser(description='Test scheduler on the simulated BLIS environment.')
    parser.add_argument('--random_runtimes', '-r', required=False,
                        dest='random_runtimes', action='store_true',
                        help='Initializes random runtimes if used. Otherwise, uses static runtimes.')
    parser.add_argument('--test_steps', '-t', required=False, default=1000,
                        dest='test_steps', help='Number of steps to test for. Default value is 1000')
    parser.add_argument('--action_size', '-a', required=False, default=10,
                        dest='action_size', help='Number of scheduling changes to make in each iteration. ' +
                        'Default value is 15')
    parser.add_argument('--reward_window_length', '-l', required=False, default=10,
                        dest='reward_window_length', help='The number of steps to look out into the future to ' +
                        'calculate the reward of an action. Default value is 10')
    parser.add_argument('--allocation_window', '-w', required=False, default=1000,
                        dest='allocation_window', help='Milliseconds to wait before recalculating allocation. Default is 1000')
    parser.add_argument('--config_file', required=True)

    parser.set_defaults(random_runtimes=False, batching=False)

    return parser.parse_args()


def validate_config(config: dict, filename: str):
    model_allocation_algos = ['random', 'static', 'lfu', 'load_proportional', 'rl',
                              'rl_warm', 'ilp_alpha', 'ilp_throughput', 'infaas',
                              'clipper', 'ilp', 'infaas_v2', 'sommelier']
    job_sched_algos = ['random', 'round_robin', 'eft_fifo', 'lft_fifo', 'infaas',
                       'canary_routing']
    batching_algorithms = ['disabled', 'accscale', 'aimd', 'infaas']

    if 'profiling_data' not in config:
        raise ConfigException(f'profiling_data not specified in config file: {filename}')

    if 'trace_path' not in config:
        raise ConfigException(f'trace_path not specified in config file: {filename}')

    if 'model_allocation' not in config:
        raise ConfigException(f'model_allocation algorithm not specified in config '
                              f'file: {filename}')

    if 'job_scheduling' not in config:
        raise ConfigException(f'job_scheduling algorithm (query assignment) not specified '
                              f'in config file: {filename}')
    
    if 'batching' not in config:
        raise ConfigException(f'batching algorithm not specified in config file: {filename}'
                              f'\nPossible choices: {batching_algorithms}')

    if 'allowed_variants' not in config:
        raise ConfigException(f'allowed_variants not specified in config file: {filename}')

    if not(os.path.exists(config['allowed_variants'])):
        raise ConfigException(f'allowed_variants path not found: {config["allowed_variants"]}')

    model_allocation = config['model_allocation']
    if model_allocation not in model_allocation_algos:
        raise ConfigException(f'invalid model_allocation algorithm specified: {model_allocation}. '
                              f'\nPossible choices: {model_allocation_algos}')

    if (model_allocation == 'ilp' or model_allocation == 'ilp_alpha') and 'beta' not in config:
        raise ConfigException(f'beta value for ILP model allocation not specified in '
                              f'config file: {filename}')

    if not(model_allocation == 'ilp' or model_allocation == 'ilp_alpha') and 'beta' in config:
        raise ConfigException(f'unexpected parameter beta specificed in config: {filename}'
                              f'beta is only needed for ILP or ILP-Alpha')

    if 'beta' in config and 1 > float(config['beta']) < 0:
        raise ConfigException(f'invalid value for beta parameter: {config["beta"]}'
                              f'Expected a value between 0 and 1')
    
    if model_allocation == 'ilp_alpha' and 'alpha' not in config:
        raise ConfigException(f'alpha value for ILP-alpha model allocation not specified '
                              f'in config file: {filename}')
    
    if model_allocation != 'ilp_alpha' and 'alpha' in config:
        raise ConfigException(f'unexpected parameter alpha specificed in config: {filename}'
                              f'alpha is only needed for ILP-Alpha')
    
    if 'alpha' in config and 1 > float(config['alpha']) < 0:
        raise ConfigException(f'invalid value for alpha parameter: {config["alpha"]}'
                              f'Expected a value between 0 and 1')

    job_scheduling = config['job_scheduling']
    if job_scheduling not in job_sched_algos:
        raise ConfigException(f'invalid job_scheduling algorithm specified: {job_scheduling}. '
                              f'\nPossible choices: {job_sched_algos}')

    batching = config['batching']
    if batching not in batching_algorithms:
        raise ConfigException(f'invalid batching algorithm specified: {batching}. '
                              f'\nPossible choices: {model_allocation_algos}')
    
    if 'fixed_seed' in config:
        if 'seed' not in config:
            raise ConfigException(f'fixed_seed is set to true but seed value is not '
                                  f'specified in config: {filename}')
    else:
        if 'seed' in config:
            raise ConfigException(f'unexpected parameter seed specified in config: '
                                  f'{filename} Seed is only expected when fixed_seed '
                                  f'is true in config')


def main(args):
    model_training_steps = 8000
    testing_steps = int(args.test_steps)
    action_group_size = int(args.action_size)
    reward_window_length = args.reward_window_length
    allocation_window = int(args.allocation_window)

    with open(args.config_file) as cf:
        config = json.load(cf)

    validate_config(config=config, filename=args.config_file)

    model_assignment = config['model_allocation']
    job_scheduling = config['job_scheduling']
    batching_algo = config['batching']

    trace_path = config['trace_path']
    fixed_seed = int(config['seed']) if 'seed' in config else 0
    alpha = float(config['alpha']) if 'alpha' in config else -1
    beta = float(config['beta']) if 'beta' in config else -1
    enable_batching = False if batching_algo == 'disabled' else True
    profiling_data = config['profiling_data']
    allowed_variants_path = config['allowed_variants']

    env = SchedulingEnv(trace_dir=trace_path, job_sched_algo=job_scheduling,
                        action_group_size=action_group_size, reward_window_length=reward_window_length,
                        random_runtimes=args.random_runtimes, fixed_seed=fixed_seed,
                        allocation_window=allocation_window, model_assignment=model_assignment,
                        batching=enable_batching, batching_algo=batching_algo,
                        profiling_data=profiling_data, allowed_variants_path=allowed_variants_path)

    policy_kwargs = dict(net_arch=[128, 128, dict(pi=[128, 128, 128],
                                        vf=[128, 128, 128])])
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
        ilp = Ilp(allocation_window=allocation_window, beta=beta)
        ilp_applied = True
        print(f'Testing with solution given by ILP (no alpha, beta={beta})')
    elif model_assignment == 'ilp_alpha':
        ilp = IlpAlpha(allocation_window=allocation_window, alpha=alpha, beta=beta)
        ilp_applied = True
        print(f'Testing with solution given by ILP-Alpha (alpha={alpha}, beta={beta})')
    elif model_assignment == 'ilp_throughput':
        ilp = IlpThroughput(allocation_window=allocation_window)
        ilp_applied = True
        print('Testing with solution given by ILP (Max throughput version)')
    elif model_assignment == 'infaas':
        print('Testing with INFaaS model assignment policy')
    elif model_assignment == 'infaas_v2':
        print('Testing with INFaaS model assignment policy (batching, cost=accuracy drop)')
    elif model_assignment == 'clipper':
        clipper = Clipper(simulator=env.simulator)
        print('Testing with Clipper model assignment policy')
    elif model_assignment == 'sommelier':
        ilp = Ilp(allocation_window=allocation_window, beta=beta,
                  starting_allocation='algorithms/sommelier_solutions/starting_uniform.txt',
                  spec_acc=True)
        ilp_applied = True
        print('Testing with Sommelier model switching policy (spec_acc)')
    else:
        print('Undefined mode, exiting')
        sys.exit(0)
    print('--------------------------')
    print()
    print('Starting in 5 seconds...')
    time.sleep(5)

    logfile_name = 'log_' + model_assignment + '.txt'
    logfile = open('logs/' + logfile_name, mode='w')

    rate_logger = logging.getLogger('Rate logger')
    rate_loggerfile = os.path.join('logs', 'throughput', str(time.time()) + '_' +  model_assignment + '.csv')
    rate_logger.addHandler(logging.FileHandler(rate_loggerfile, mode='w'))
    rate_logger.setLevel(logging.DEBUG)
    rate_logger.info('wallclock_time,simulation_time,demand,throughput,capacity')

    rate_logger_per_model = logging.getLogger('Rate logger per model')
    rate_logger_per_model_file = os.path.join('logs', 'throughput_per_model', str(time.time()) + '_' + model_assignment + '.csv')
    rate_logger_per_model.addHandler(
        logging.FileHandler(rate_logger_per_model_file, mode='w'))
    rate_logger_per_model.setLevel(logging.DEBUG)
    rate_logger_per_model.info(
        'wallclock_time,simulation_time,demand_nth_model,throughput_nth_model,normalized_throughput_nth_model,accuracy_nth_model,,,,,,,,,,,,,,,,')

    rl_reward = 0
    observation = env.reset()
    failed_requests = 0
    total_requests = 0
    successful_requests = 0
    ilp_rounds = 0
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
            # We only need to apply the assignment once at the start
            if i == 0:
                # proportions = np.array([0.30, 0.14, 0.091, 0.066, 0.051, 0.042, 0.035, 0.030, 0.027, \
                #                0.024, 0.022, 0.020, 0.018, 0.016, 0.015, 0.014, 0.013, 0.012, \
                #                0.012, 0.011, 0.010, 0.009])
                proportions = np.array([0.37089186, 0.17539785, 0.11117586, 0.08196145, 0.0635107, \
                                        0.05204432, 0.04448577, 0.03854691, 0.03237666, 0.02960862])
                # we calculate the shares for each model architecture (ISI) and round to integer
                shares = np.rint(proportions * env.n_accelerators * env.max_no_of_accelerators)

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
        elif model_assignment == 'ilp' or model_assignment == 'ilp_alpha' or model_assignment == 'ilp_throughput' or model_assignment == 'sommelier':
            if ilp.is_simulator_set() is False:
                ilp.set_simulator(env.simulator)

            if ilp_applied == True:
                period_tuning = 1
                # TODO: Tune how frequently the ILP is run by tuning 'period_tuning'
                #       The bigger it is, the less frequently the ILP is invoked
                #       Also, what are its implications on allocation window sizes and
                #       action group sizes?
                if i % period_tuning == 0:
                    actions = ilp.run(observation, env.n_accelerators, env.max_no_of_accelerators)
                    ilp_rounds += 1
                else:
                    actions = None

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
            
            # # This is used to get a solution for Clipper at midpoint
            # # of experiment
            # if i > 0 and i % 500 == 0:
            #     # ilp.print_cached_solution()
            #     time.sleep(10)

            # action = env.action_space.sample()
            # for j in range(5, len(action)):
            #     action[j] = 0
        elif 'infaas' in model_assignment:
            print()
            if model_assignment == 'infaas':
                env.trigger_infaas_upscaling()
                env.trigger_infaas_downscaling()
            elif model_assignment == 'infaas_v2':
                env.trigger_infaas_v2_upscaling()
                env.trigger_infaas_v2_downscaling()
            else:
                print(f'Invalid verison of INFaaS: {model_assignment}')
            
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
        elif model_assignment == 'clipper':
            clipper.apply_solution()
            action = env.action_space.sample()
            action[0] = 0
            action[1:5] = observation[0, 0:4]
            # time.sleep(2)

        if i % 100 == 0:
            print('Testing step: {} of {}'.format(i, testing_steps))
            print('Action taken: {}'.format(action))
            # print('State: {}'.format(env.state))
        observation, reward, done, info = env.step(action)
        # read failed and total requests once every 5 actions
        if (i-1) % action_group_size == 0 or done:
            failed_requests += np.sum(observation[:,-1])
            total_requests += np.sum(observation[:,-2])
            successful_requests += env.simulator.get_successful_requests()
            logfile.write(str(total_requests) + ',' +
                          str(failed_requests) + '\n')
            requests_per_model, failed_per_model, accuracy_per_model, successful_per_model = env.simulator.get_thput_accuracy_per_model()
            utils.log_thput_accuracy_per_model(rate_logger_per_model, i, requests_per_model,
                                            failed_per_model, accuracy_per_model)
        if done:
            break
        rl_reward += reward
        env.render()


        utils.log_throughput(rate_logger, observation, i, allocation_window)
        print('observation:', observation)

    end = time.time()

    print()
    print('---------------------------------')
    print('Printing overall statistics below')
    print('---------------------------------')
    print(f'Reward from RL model: {rl_reward}')
    print(f'Requests failed by RL model: {failed_requests}')
    print(f'Total requests: {total_requests}')
    print(
        f'Percentage of requests failed: {(failed_requests/total_requests*100)}%')
    print(
        f'Percentage of requests succeeded: {(successful_requests/total_requests*100)}%')
    print(f'Test time: {(end-start)} seconds')
    print(f'Requests added: {env.simulator.requests_added}')
    print(f'ILP made changes {ilp_rounds} times')
    logfile.close()

    completed_requests = env.simulator.completed_requests
    sim_time_elapsed = env.simulator.clock / 1000
    overall_throughput = completed_requests / sim_time_elapsed

    print()
    print(f'Completed requests: {completed_requests}')
    print(f'Simulator time elapsed: {sim_time_elapsed}')
    print(f'Overall throughput (requests/sec): {overall_throughput}')

    total_accuracy = env.simulator.total_accuracy
    effective_accuracy = total_accuracy / completed_requests
    print(f'Effective accuracy (over served requests): {effective_accuracy}')
    print()

    print('---------------')
    print('Logging details')
    print('---------------')
    print(f'Aggregate throughput and accuracy logs written to: {rate_loggerfile}')
    print(f'Per-model throughput and accuracy logs written to: {rate_logger_per_model_file}')
    print(f'Latency (response time) logs written to: {env.simulator.latency_logfilename}')

    bumped_succeeded = env.simulator.slo_timeouts['succeeded']
    bumped_failed = env.simulator.slo_timeouts['timeouts']
    bumped_total = bumped_failed + bumped_succeeded
    bumped_violation_ratio = bumped_failed / bumped_total
    print(f'SLO violation ratio based on bumped stats: {bumped_violation_ratio}')
    # total_slo = env.simulator.slo_timeouts['total']
    print(f'Total SLO: {bumped_total}, SLO timed out: {bumped_failed}, successful '
          f'SLO: {bumped_succeeded}, timeout ratio: {bumped_violation_ratio}')
    print()

    print(f'test stat: {env.simulator.test_stat}')

    # bumped_failed = env.simulator.failed_requests
    # bumped_succeeded = env.simulator.successful_requests
    # bumped_total = bumped_failed + bumped_succeeded
    # bumped_violation_ratio = bumped_failed / bumped_total
    # print(f'SLO violation ratio based on bumped stats: {bumped_violation_ratio}')

if __name__=='__main__':
    main(getargs())
