from src import network, replay_memory_agent,deep_q_agent, epsi_greedy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import gym
# Make the environemnt 
env = gym.make("CartPole-v0")
# Tensorflow Session
sess = tf.Session()
# Initalize the neural network
ac_function = network(4, 2, sess)
target_function = network(4, 2, sess)
# Make sure both networks start from the same weight
target_function.set_weights(ac_function)
# Replay memory
replay = replay_memory_agent(4, 10000)
# Deep Q learning agent
prof_x = deep_q_agent(
    ac_function,
    target_function,
    4,
    replay.replay_memory,
    epsi_greedy)
state = env.reset()
# book keeping
done=False
episodes = 0
reward_episode = []
reward_track = []
epsilon = 1.0
while episodes < 8000:
    # Pick action
    state = np.asarray(state)
    state = state.reshape(1, 4)
    q_values = ac_function.predict_on_batch(state)
    action = epsi_greedy([0, 1], q_values, epsilon)
    # implement action
    state_new, reward, done, _ = env.step(action)
    reward_episode.append(reward)
    # Update the replay memory
    replay.replay_memory_update(state, state_new, reward, action, done)
    # train
    if episodes > 1000:
        update = True if episodes%500==0 else False
        prof_x.train_q(update)
    state = state_new
    if done:
        epsilon = max(0.01, epsilon*0.99) 
        state = env.reset()
        reward_track.append(np.sum(reward_episode))
        episodes += 1
        reward_episode = []
plt.plot(reward_track)
plt.xlabel("Episodes")
plt.ylabel("Avg Reward")
plt.show()
