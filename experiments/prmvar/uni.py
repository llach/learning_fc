import numpy as np
import matplotlib.pyplot as plt

means = []
for _ in range(10000):
    means.append(
        (np.random.uniform(0,1)+np.random.uniform(0,1))/2
    )
plt.hist(means, bins=20)
plt.show()