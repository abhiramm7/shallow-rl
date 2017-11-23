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
env = gym.make("CartPole-v1")

# Book keeping
model_reward = []
# Global time step
gt = 0

for weights in np.linspace(0, 8500, 86, dtype=int):

    q_nn.load_weights("model"+str(weights))

    avg_reward_episodes = []

    for episode_count in range(0, 10):
        # Initial State
        state = env.reset()
        done=False
            
        episode_time = 0
        
        rewards = []
        while not(done):
            gt += 1

            # Reshape the state
            state = np.asarray(state)
            state = state.reshape(1,4)

            # Pick a action based on the state
            q_values = q_nn.predict_on_batch(state)

            action = np.argmax(q_values)

            # Implement action and observe the reward signal
            state_new, reward, done, _ = env.step(action)
            rewards.append(reward)

            state = state_new

            episode_time += 1
            if episode_time > 300:
                break

        avg_reward_episodes.append(sum(rewards))
    
    model_reward.append(np.mean(avg_reward_episodes))

np.save("sum_rewards_dqn_plot", model_reward)
plt.plot(model_reward)
plt.show()
