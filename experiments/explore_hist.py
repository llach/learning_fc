import pickle

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path

file_path = f"{model_path}/data/2023-08-04_11-00-23__glue.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

q = np.array(data["obs"]["q"])
force = np.array(data["obs"]["f"])

qdes = np.array(data["obs"]["qdes"])
act = np.array(data["obs"]["act"])
goal = np.array(data["goal"])


fig, axes = plt.subplots(ncols=3)

xs = np.arange(len(data["obs"]["q"]))

axes[0].plot(xs, q)
axes[0].plot(xs, qdes)
axes[0].set_ylim(-0.001, 0.049)

axes[1].plot(xs, force)
axes[1].plot(xs, goal)
axes[1].set_ylim(-0.05, 1.0)

axes[2].plot(xs, act)
axes[2].set_ylim(-1,1)
ax22 = axes[2].twinx()
ax22.plot()

plt.show()