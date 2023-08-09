import os
import pickle
import learning_fc

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.envs import GripperTactileEnv

from matplotlib.legend_handler import HandlerTuple

ftheta = 0.0075
oname = "glue_mid"
files_dir = f"{model_path}/data/net/"

tg = []
wo = []
fmax = []
forces = []
actions = []
goals = []
qs = []
dts = []

for fi in os.listdir(files_dir):
    if oname not in fi: continue 

    with open(f"{files_dir}{fi}", "rb") as f:
        data = pickle.load(f)
    
    t = np.array(data["timestamps"])
    q = np.array(data["obs"]["q"])
    f = np.array(data["obs"]["f"])

    tg.append(np.argmax(np.all(f>ftheta, axis=1)))
    fmax.append(np.mean(np.max(f, axis=0)))
    wo.append(np.abs(np.sum(q[tg[-1]])))

    qs.append(q)
    forces.append(f)

    dts.append(np.array([
        [(t[i]-t[i-1]).total_seconds(), (t[i]-t[i-1]).total_seconds()]
        for i in np.arange(1,len(t))
    ]))

    goals.append(np.array(data["goal"]))
    actions.append(np.array(data["net_out"]))

f_rob = forces[-1]
q_rob = qs[-1]
n_steps = len(q_rob)
wo = np.median(wo)/2
act = actions[-1]
goal = goals[-1]

fig, axes = plt.subplots(ncols=3, figsize=(13,5))

xs = np.arange(n_steps)

axes[0].plot(xs, q_rob)
axes[0].set_ylim(-.001, 0.047)
axes[0].set_title("joint positions")

axes[1].plot(xs, f_rob)
axes[1].plot(xs, goal)
axes[1].set_ylim(-.05, 1.0)
axes[1].set_title("forces")

axes[2].plot(xs, act)
axes[2].set_ylim(-1.05, 0.1)
axes[2].set_title("network output")

for ax in axes.flatten():
    ax.axvline(tg[-1], c="grey", ls="dashed", lw=0.7)

fig.suptitle(oname)
fig.tight_layout()
plt.show()