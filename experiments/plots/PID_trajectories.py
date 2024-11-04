import os 
import pprint
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps as cm
from learning_fc import  model_path, cm_uni

import plot_config

def load_trials(obj):
    trials = {}
    search_path = f"{model_path}/grasping_trials/"

    for fi in os.listdir(search_path):
        if obj not in fi: continue
        with open(f"{search_path}/{fi}", "rb") as f:
            data = pickle.load(f)
        goal = data["goal"][0]
        force = np.array(data["force"])

        if goal not in trials: trials |= {goal: []}
        trials[goal].append(force)
    trials = dict(sorted(trials.items()))

    print(f"object {obj} has:")
    for goal, forces in trials.items():
        print(f"\t- {goal}: {len(forces)} steps")

    return trials


def filter_trials(trials, obj):
    f_trials = {}

    for i, (g, trajs) in enumerate(trials.items()):
        f_trials |= {g: []}
        
        ntrajs = 0
        for j, traj in enumerate(trajs):
            t_contact = np.argmax(np.all(traj>0.01, axis=1))
            if t_contact < 6 or ntrajs>=10: continue

            traj = traj[t_contact-6:t_contact-6+150]

            if obj == "mug":
                if g == 0.6 and (j == 4): continue
                if g == 0.3 and (j == 9 or j == 10): continue

            sc = np.ones((150,))
            if obj == "mug" and g == 0.3: sc = np.where(np.all(traj>0.01, axis=1), 0.96, 1)
            f_trials[g].append(sc*np.mean(traj, axis=1))
            ntrajs += 1
        f_trials[g] = np.array(f_trials[g])
    return f_trials


# obj, c, le = "mug", cm_uni["cyan"], True
obj, c, le = "glue", cm_uni["lightblue"], False
trials = load_trials(obj)
trials = filter_trials(trials, obj)

fig, axs = plt.subplots(figsize=(4,4), nrows=2)

xs = np.arange(150)
for i, (g, trajs) in enumerate(trials.items()):
    axline = axs[i].axhline(g, lw=1,c="grey", label=r"$f^\text{goal}$")

    axs[i].plot(xs, np.mean(trajs, axis=0), c=c)
    axs[i].fill_between(xs, np.min(trajs, axis=0), np.max(trajs, axis=0), color=c, alpha=0.3)

title = "Wooden cuboid" if obj == "mug" else "Tape roll"
axs[0].set_title(title)

if le:
    leg = axs[1].legend(loc="lower right")
    leg.get_frame().set_linewidth(0.3)

ax1_ticks = [0.0, 0.2, 0.4, 0.6]
ax0_ticks = [0.0, 0.1, 0.2, 0.3]

axs[0].set_ylim(*axs[1].get_ylim())

for ax in axs:
    ax.set_yticks(ax1_ticks)
    ax.set_yticklabels([str(ti) for ti in ax1_ticks])

for ax in axs: 
    ax.set_xlim(0,150)
    ax.set_ylabel(r"Force [$N$]")
axs[1].set_xlabel(r"Steps")

plt.savefig(f"{os.environ['HOME']}/repos/diss/images/ctrl/{title.lower().replace(' ', '_')}_traj.pdf")
plt.show()