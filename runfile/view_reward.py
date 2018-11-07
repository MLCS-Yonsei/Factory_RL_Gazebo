import numpy as np
import matplotlib.pyplot as plt

d = np.load('reward_summary.npy')
plt.plot(d)
plt.show()
