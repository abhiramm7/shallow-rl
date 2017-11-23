import numpy as np
import matplotlib.pyplot as plt

a = np.load("./DQN/sum_rewards_dqn_plot.npy")
b = np.load("./ADQN/sum_rewards_adqn_plot.npy")


plt.plot(a, label="DQN")
plt.plot(b, label="ADQN")
plt.xlabel("Episodes (100's)")
plt.ylabel("Avg.Reward")
plt.legend()
plt.savefig("Final")
