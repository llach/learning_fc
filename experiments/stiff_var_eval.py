import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

from learning_fc import model_path

widths = {
    "sponge": 45,
    "mug": 84.5,
    # "glue": 58.15
}
colors = ["orange", "blue"]

def read_f_q(obj_name):
    assert obj_name in widths, f"{obj_name} not in {list(widths.keys())}"
    exp_path = f"{model_path}/stiff_exp/"
    fTs, dpTs, dqdes = [], [], []

    for fi in os.listdir(exp_path):
        if not fi.endswith(".pkl") or obj_name not in fi: continue

        with open(f"{exp_path}/{fi}", "rb") as f:
            data = pickle.load(f)

        f = np.mean(data["force"], axis=1)
        fT = np.mean(f[-10:])
        fTs.append(fT)

        apertureT = np.sum(data["q"], axis=1)[-1]
        dpT = widths[obj_name]-1000*apertureT
        dpTs.append(dpT)

        dqdes.append(float(fi.split("__")[1].split(".pkl")[0]))
    fTs = np.array(fTs)
    dpTs = np.array(dpTs)
    dqdes = np.array(dqdes)

    idxs = np.argsort(dqdes)
    fTs = fTs[idxs][2:]
    dpTs = dpTs[idxs][2:]
    dqdes = dqdes[idxs][2:]

    return fTs, dpTs, dqdes
        
fig, axs = plt.subplots(ncols=2, figsize=(10,5))
for i, obj in enumerate(widths.keys()):
    c= colors[i]
    fTs, dpTs, dqdes = read_f_q(obj)

    axs[0].scatter(dqdes, fTs, label=obj, c=c)
    b, m = polyfit(dqdes, fTs, 1)
    axs[0].plot(dqdes, b + m*dqdes, linestyle='-', alpha=0.3, c=c)

    axs[1].scatter(dqdes, dpTs, label=obj, c=c)
    b, m = polyfit(dqdes, dpTs, 1)
    axs[1].plot(dqdes, b + m*dqdes, linestyle='-', alpha=0.3, c=c)

axs[0].set_ylabel("f(T)")
axs[1].set_ylabel("Penetration Depth [mm]")

for ax in axs: 
    ax.legend()
    ax.set_xlabel("\Delta q_des")

fig.tight_layout()
plt.savefig("/Users/llach/stiffness_experiment.png")
plt.show()