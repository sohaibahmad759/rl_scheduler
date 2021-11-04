import time
import numpy as np
from stable_baselines3 import PPO
# from stable_baselines.sac import SAC
# from stable_baselines.common.evaluation import evaluate_policy
from scheduling_env import SchedulingEnv


if __name__=='__main__':
    print('Starting the training process')

    # TODO: move all these cmd line parameters
    training_steps = 8000
    testing_steps = 500
    action_group_size = 15
    reward_window_length = 10

    random = False

    env = SchedulingEnv(trace_dir='traces/twitter/', action_group_size=action_group_size,
                        reward_window_length=reward_window_length)

                    #                    / 128 - 128 - 128 - policy (pi)
                    # input -- 128 - 128 
                    #                    \ 128 - 128 - 128 - value function

    policy_kwargs = dict(net_arch=[128, 128, dict(pi=[128, 128, 128],
                                        vf=[128, 128, 128])])

    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=2)
    model_name = 'train_' + str(training_steps) + '_action_size_' + str(action_group_size) + \
                    '_window_' + str(reward_window_length)
    # model = SAC('MlpPolicy', env, verbose=1)

    start = time.time()
    if not random:
        model.learn(total_timesteps=training_steps)
    end = time.time()
    model.save('saved_models/' + model_name)

    print()
    print('--------------------------')
    if random:
        print('No training done, testing with random actions')
    else:
        print('Training completed in {} seconds'.format(end-start))
    print('--------------------------')
    print()

    rl_reward = 0
    observation = env.reset()
    failed_requests = 0
    total_requests = 0
    start = time.time()
    for i in range(testing_steps):
        if random:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(observation)
        if i % 100 == 0:
            print('Testing step: {} of {}'.format(i, testing_steps))
            print('Action taken: {}'.format(action))
        observation, reward, done, info = env.step(action)
        # read failed and total requests once every 5 actions
        if (i-1) % action_group_size == 0:
            failed_requests += np.sum(observation[:,-1])
            total_requests += np.sum(observation[:,-2])
        rl_reward += reward
        env.render()
    end = time.time()
    print('Reward from RL model: ' + str(rl_reward))
    print('Requests failed by RL model: ' + str(failed_requests))
    print('Total requests: ' + str(total_requests))
    print('Percentage of requests failed: {}%'.format(failed_requests/total_requests*100))
    print('Test time: {} seconds'.format(end-start))
    print('Requests added: {}'.format(env.simulator.requests_added))
