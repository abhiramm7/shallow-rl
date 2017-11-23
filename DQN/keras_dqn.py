from src import replay_memory_agent, deep_q_agent, epsi_greedy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import gym

def build_network(input_states,
                  output_states,
                  hidden_layers,
                  nuron_count,
                  activation_function,
                  dropout):
    """
    Build and initialize the neural network with a choice for dropout
    """
    model = Sequential()
    model.add(Dense(nuron_count, input_dim=input_states))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))
    for i_layers in range(0, hidden_layers - 1):
        model.add(Dense(nuron_count))     
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

q_nn = build_network(4, 2, 2, 20, "relu", 0.0);
#q_nn.load_weights("model_1")
target_nn = build_network(4, 2, 2, 20, "relu", 0.0);


target_nn.set_weights(q_nn.get_weights())

replay1 = replay_memory_agent(4, 5000)

dqn_controller = deep_q_agent(action_value_model=q_nn,
                              target_model=target_nn,
                              states_len=4,
                              replay_memory=replay1)

env = gym.make("CartPole-v1")

# Book keeping
avg_reward_episodes = []
# Global time step
gt = 0

for episodes in range(0, 10000):

    # Initial State
    state = env.reset()
    done=False
    
    # Clear the reward buffer
    rewards = []
    if gt > 10000:
        epsilon = max(0.01, epsilon-0.0001)
    else:
        epsilon = 1.0
    
    episode_time = 0

    while not(done):
        gt += 1

        # Reshape the state
        state = np.asarray(state)
        state = state.reshape(1,4)

        # Pick a action based on the state
        q_values = q_nn.predict_on_batch(state)

        if np.random.rand() <= epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(q_values)


        # Implement action and observe the reward signal
        state_new, reward, done, _ = env.step(action)
        rewards.append(reward)

        # Update the replay memory
        replay1.replay_memory_update(state, state_new, reward, action, done)

        if gt > 10000:
            # Train
            update = True if gt%5000==0 else False
            dqn_controller.train_q(update)
            if update:
                print("updated ", gt)

        state = state_new

        episode_time += 1
        if episode_time > 200:
            break

    avg_reward_episodes.append(sum(rewards))
    if episodes%100 == 0:
        print(sum(rewards), "Episode Count :" ,episodes)
        q_nn.save_weights("model"+str(episodes))

np.save("sum_rewards", avg_reward_episodes)
plt.plot(avg_reward_episodes)
plt.show()
