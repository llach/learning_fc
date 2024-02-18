import os 
import pprint
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps as cm
from learning_fc import  model_path, cm_uni

# legend 
plt.rcParams['legend.fancybox']  = True
plt.rcParams['legend.edgecolor'] = "#656565"

# axes
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['lines.linewidth'] = 1.2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] = "medium"
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams['figure.constrained_layout.use'] = True


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage[T1]{fontenc}'


model = "fc"

def pad_forces(f1, f2):
    """ pad shorter force trajectory by repeating noise at the beginning of the sequence
    """
    lf1, lf2 = len(f1), len(f2)
    if lf1 == lf2: return f1, f2

    pad_len = np.abs(lf1-lf2)
    if lf1 < lf2:
        return np.concatenate([f1[:pad_len], f1]), f1
    else:
        return f1, np.concatenate([f2[:pad_len], f2])
    
def assure_len(traj, minlen=150):
    tlen = len(traj)
    if tlen < minlen:
        lendiff = minlen-tlen
        traj = np.concatenate([traj[:lendiff], traj])
    return traj

def plot_two_models(ax, base, pol, title, with_offset=False, legend=False):
    cmap = cm["tab10"]
    g_offset = 0

    for goal, f_base in base.items():
        f_pol = np.squeeze(assure_len(pol[goal]))
        f_base = np.squeeze(assure_len(f_base))

        if title == "Mug" and goal == 0.7:
            f_pol = f_pol[20:]
            f_base = f_base[20:]
        if title == "Mug" and goal == 0.5:
            f_pol = f_pol[10:]
            f_base = f_base[10:]
        if title == "Mug" and goal == 0.2:
            f_pol = f_pol[13:]
            f_base = f_base[13:]

        if title == "Plush Toy" and goal == 0.7:
            f_pol = f_pol[10:]
            f_base = f_base[10:]
        if title == "Plush Toy" and goal == 0.5:
            f_pol = f_pol[10:]
            f_base = f_base[10:]
        if title == "Plush Toy" and goal == 0.2:
            f_pol = f_pol[10:]
            f_base = f_base[10:]

        if with_offset:
            if goal == 0.2:
                f_base = f_base[8:]
            elif goal == 0.5:
                f_base = f_base[10:]
            elif goal == 0.7:
                f_pol = f_pol[7:]

        f_pol = f_pol[:150]
        f_base = f_base[:150]
        f_base, f_pol = pad_forces(f_base, f_pol)
        l_forces = len(f_base)
        
        goals = np.repeat(goal, l_forces)
        xs = g_offset+np.arange(l_forces)

        gl_p, = ax.plot(xs, goals, color=cmap(7), label="f_goal")
        bl_p, = ax.plot(xs, np.mean(f_base, axis=1), color=cm_uni["orange"], label="Baseline")
        po_p, = ax.plot(xs, np.mean(f_pol, axis=1), color=cm_uni["blue"], label="Policy")

        g_offset += l_forces

    ax.set_ylabel(r"Force [$N$]")
    ax.set_xlim([-5,405])

    if legend: 
        l = ax.legend([bl_p, po_p, gl_p], [
            f'Baseline', 
            r"$\pi^{IB}$",
            r"$f^{\text{goal}}$",
        ])
        l.get_frame().set_linewidth(0.5)
    if title is not None: ax.set_title(title)

    return [gl_p, bl_p, po_p]

def plot_all_trials(trials, title=None):
    goals = []
    forces = []
    for k, v in trials.items():
        f = np.vstack(v)
        goals.append(np.repeat(k, f.shape[0]))
        forces.append(np.mean(f, axis=1))

    goals = np.hstack(goals)
    print(len(goals))
    forces = np.hstack(forces)
    xs = np.arange(len(goals))

    plt.plot(xs, goals)
    plt.plot(xs, forces)
    if title is not None: plt.title(title)

    plt.tight_layout()
    plt.show()


def load_trials(model, obj="chicken"):
    trials = {}
    search_path = f"{model_path}/robot_eval_{obj}/{model}"
    for fi in os.listdir(search_path):
        if not fi.endswith("pkl"): continue
        with open(f"{search_path}/{fi}", "rb") as f:
            data = pickle.load(f)
        goal = data["goal"][0]
        force = np.array(data["force"])

        if goal not in trials: trials |= {goal: []}
        trials[goal] = force
    trials = dict(sorted(trials.items()))

    print(f"model {model} has:")
    for goal, forces in trials.items():
        print(f"\t- {goal}: {len(forces)} steps")

    return trials

fc_mug = load_trials("fc", obj="mug")
model_mug = load_trials("pol", obj="mug")

fc_chick = load_trials("fc", obj="chicken")
model_chick = load_trials("pol", obj="chicken")

fig, (ax1, ax2) = plt.subplots(nrows=2)

plot_two_models(ax1, fc_mug, model_mug, title="Mug", with_offset=True, legend=True)
artists = plot_two_models(ax2, fc_chick, model_chick, title="Plush Toy", with_offset=False)

ax1.set_xticklabels([])
ax2.set_xlabel("Steps")

plt.savefig(f"{model_path}/traj_compare.pdf")
plt.show()