
import random
import gym
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from old_env import SchedulingEnv

# 50,000 training steps perform pretty well, no need to go to 500,000 (5,000 may not be enough)
training_steps = 5000000
testing_steps = 5000

env = SchedulingEnv(training_steps)

model = PPO2('MlpPolicy', env, verbose=1)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# env.reset()

model.learn(total_timesteps=training_steps)
print('Done with training!')

# env.reset()
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

rl_reward = 0
env.remake()
observation = env.reset()
# env.render()
done = False
# while not done:
for i in range(testing_steps):
    # how do we know which model we are scheduling?
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    rl_reward += reward
    # env.render()
print('Total reward from RL model: ' + str(rl_reward))

guided_rl_reward = 0
observation = env.reset()
# env.render()
done = False
# while not done:
for i in range(testing_steps):
    # action, _states = model.predict(observation)
    state = observation[:-1]
    next_request = observation[-1]
    if state[0] == next_request:
        next_action = 1
    elif state[1] == next_request:
        next_action = 2
    else:
        next_action, _states = model.predict(observation)
    observation, reward, done, info = env.step(next_action)
    guided_rl_reward += reward
    # env.render()
print('Total reward from guided RL model: ' + str(guided_rl_reward))


random_reward = 0
env.reset()
done = False
# while not done:
for i in range(testing_steps):
    observation, reward, done, info = env.step(random.randint(0, 2))
    random_reward += reward
print('Total reward from random assignment: ' + str(random_reward))


lru_reward = 0
env.reset()
done = False
next_action = 1
flip = 1
last_used = 1
# while not done:
for i in range(testing_steps):
    observation, reward, done, info = env.step(next_action)
    # print(reward)
    lru_reward += reward

    # LRU policy
    state = observation[:-1]
    next_request = observation[-1]
    if state[0] == next_request:
        next_action = 1
    elif state[1] == next_request:
        next_action = 2
    else:
        # which one to evict? evict the one used least recently
        # print()
        # next_action = flip
        # if flip == 1:
        #     flip = 2
        # else:
        #     flip = 1
        if last_used == 1:
            next_action = 2
        elif last_used == 2:
            next_action = 1

    last_used = next_action


print('Total reward from LRU assignment: ' + str(lru_reward))
