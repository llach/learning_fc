import os
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import plot_config

from learning_fc import  cm_uni

def read_grav(pklpath, fro, to):
    with open(f"{pathlib.Path(__file__).parent.absolute()}/{pklpath}", "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1") # Python3
        except TypeError:
            data = pickle.load(f) # Python2

    return {
        "deltaT": np.array(data['deltaT'][fro:to]),
        "f": data['f'][fro:to],
        "fDiff": data['fDiff'][fro:to],
        "fAdd": data['fAdd'][fro:to],
        "mgzn": data['mgzn'][fro:to],
        "cosGrav": data['cosGrav'][fro:to],
        "qDes": np.array(data['qDes'][fro:to]),
        "noiseT": data['noiseT'],
        "driftT": data['driftT'],
        "targetF": data['targetF']
    }
    

pkl_old = "data/plot_13.09.2021_13:01:57.pkl"
pkl_grav = "data/grav180_FC_28.02.2022_11:28:31.pkl"
pkl_nograv = "data/nograv180_FC_28.02.2022_11:30:38.pkl"

dnog = read_grav(pkl_nograv, 1350, 1970)
dg = read_grav(pkl_grav, 1350, 1970)

otdg = ((dg["qDes"][:, 0]-dg["qDes"][:, 1])/2) * 1000
otdnog = ((dnog["qDes"][:, 0]-dnog["qDes"][:, 1])/2) * 1000

otdg = otdg - otdg[0]
otdnog = otdnog - otdnog[0]

# dnog["deltaT"] += 5.8

dnog["deltaT"] -= dnog["deltaT"][0]
dg["deltaT"] -= dg["deltaT"][0]
dnog["mgzn"] += np.abs(dnog["mgzn"][0])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))

# O(t)
axes[0].set_ylabel(r"$\Delta O(t)$ [$mm$]")

axes[0].plot(dg["deltaT"], otdg, label=r"gravity comp. \textbf{on}", c=cm_uni["blue"],lw=1.2)
axes[0].plot(dnog["deltaT"], otdnog, label=r"gravity comp. \textbf{off}", c=cm_uni["lightred"],lw=1.2)

leg = axes[0].legend(loc="lower left")
leg.get_frame().set_linewidth(0.3)

# axes[0].tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
# )
axes[0].get_xaxis().set_ticklabels([])

# mgzn
axes[1].set_ylabel(r"$m\,\mathbf{g}\cdot\mathbf{n}$ [$N$]")
axes[1].plot(dnog["deltaT"], dnog["mgzn"], c=cm_uni["green"],lw=1.2)
axes[1].set_xlabel(r"Time [$s$]")

for ax in axes:
    ax.set_xlim(0,6)
    ylims = ax.get_ylim()
    # ax.set_ylim(ylims[0], 0)

# for ax in axes[:-1].flatten():
#     ax.set_xlim(15, 19)
#     ax.get_xaxis().set_visible(False) # hide xlabels for all but last rows

# fig.text(0.5, 0.009, 'time [$s$]', ha='center')
plt.savefig(f"{os.environ['HOME']}/repos/diss/images/ctrl/gravity.pdf")
plt.show()
