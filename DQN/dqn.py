from src import network, replay_memory_agent,deep_q_agent, epsi_greedy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

import gym
env = gym.make("MountainCar-v0")

sess = tf.Session()
ac_function = network(2,3, sess)
target_function = network(2,3, sess)
target_function.set_weights(ac_function)
replay = replay_memory_agent(2, 4000)
prof_x = deep_q_agent(
    ac_function,
    target_function,
    3,
    replay.replay_memory,
    epsi_greedy)
state = env.reset()
# book keeping
done=False
reward_track = []
reward_sum = []
temp = []
reward_episode = []
episodes = 0
epsilon = 0.10
for i in range(0, 150000):
    # Pick action
    state = np.asarray(state)
    state = state.reshape(1,2)
    q_values = ac_function.predict_on_batch(state)
    action = epsi_greedy([0, 1, 2], q_values, epsilon)
    # implement action
    state_new, reward, done, _ = env.step(action)
    reward_episode.append(reward)
    # Update the replay memory
    replay.replay_memory_update(state, state_new, reward, action, done)
    # train
    if i > 1000:
        update = True if i%4000==0 else False
        prof_x.train_q(update)
        if update:
            print(np.mean(reward_episode))
            print(np.sum(reward_sum))
            temp.append(np.sum(reward_sum)/150000)
    state = state_new
    if done:
        state = env.reset()
        reward_track.append(np.mean(reward_episode))
        episodes += 1
        reward_sum.append(np.sum(reward_episode))
        reward_episode = []

plt.figure(1)
plt.plot(reward_track)

plt.figure(2)
plt.plot(temp)
