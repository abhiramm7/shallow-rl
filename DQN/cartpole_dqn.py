from src import network, replay_memory_agent, deep_q_agent, epsi_greedy
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

np.random.seed(42)
env = gym.make("CartPole-v0")

q_nn = network(input_states=4, num_layers=1, nurons_list=[10], output_states=2, session=tf.Session())
target_nn = network(input_states=4, num_layers=1, nurons_list=[10], output_states=2, session=q_nn.session)
target_nn.set_weights(q_nn.get_weights())

replay1 = replay_memory_agent(4, 10000)
dqn_controller = deep_q_agent(action_value_model=q_nn,
                              target_model=target_nn,
                              states_len=4,
                              replay_memory=replay1)


# Book keeping
avg_reward_episodes = []

# Exploration decay
epsilon = np.linspace(0.10, 0.01, 5001)

# Global time step
gt = 0

for episodes in range(0, 5000):

    # Initial State
    state = env.reset()
    done=False

    # Clear the reward buffer
    rewards = []

    while not(done):
        gt += 1

        # Reshape the state
        state = np.asarray(state)
        state = state.reshape(1,4)

        # Pick a action based on the state
        q_values = q_nn.predict_on_batch(state)
        action = epsi_greedy([0, 1], q_values, 0.10)#epsilon[episodes])

        # Implement action and observe the reward signal
        state_new, reward, done, _ = env.step(action)
        rewards.append(reward)

        state_new = np.asarray(state_new)
        state_new = state_new.reshape(1,4)

        # Update the replay memory
        replay1.replay_memory_update(state, state_new, reward, action, done)

        # Train
        update = True if gt%1000==0 else False
        dqn_controller.train_q(update)

        state = state_new

    avg_reward_episodes.append(sum(rewards))
    if episodes%25 == 0:
        print(sum(rewards), "Episode Count :" ,episodes)

q_nn.save_weights("model_1")
np.save("sum_rewards", avg_reward_episodes)

plt.plot(avg_reward_episodes)
plt.show()
