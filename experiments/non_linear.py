import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(0,6,100)+0.00001

def plot_xs(ax, xs):
    ax.plot(xs, 1/(xs), c="blue")

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
plot_xs(axs[0], xs)
plot_xs(axs[1], np.log(xs))

for ax in axs: 
    ax.set_xlim(0,6)
    ax.set_ylim(0,6)

plt.show()