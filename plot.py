import matplotlib.pyplot as plt
import numpy as np


x = np.load("results\q_learning\sum_reward_Q-Learning Agent.npy")
print(x)
print(x.shape)

plt.plot(np.arange(500), x[0])
plt.savefig('sum_rewards.png')
plt.show()
