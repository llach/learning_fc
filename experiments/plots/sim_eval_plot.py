import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path, cm_uni
from sim_eval_data import dist, width, kappas

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

def configure_boxes(box, ax):

    # Customizing box colors
    colors = [cm_uni["orange"], cm_uni["blue"], cm_uni["gold"], cm_uni["green"], cm_uni["magenta"], cm_uni["lightblue"]]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Customizing whiskers and caps
    for whisker, cap in zip(box['whiskers'], box['caps']):
        whisker.set(color='#212121', linewidth=1)
        cap.set(color='#212121', linewidth=1)

    # Customizing median lines
    for median in box['medians']:
        median.set(color='#212121', linewidth=2)

    # Customizing outliers
    for flier in box['fliers']:
        flier.set(marker='o', color='#757575', alpha=0.5)

    # Adding grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, lw=0.6)

    # Customizing tick labels
    ax.set_xticklabels([
        f'Baseline', 
        r"$\pi^\text{\small IB}$",
        r"$\pi^\text{\small NO-IB}$",
        r"$\pi^\text{\small NO-PEN}$",
        r"$\pi^\text{\small NO-RAND}$",
        r"$\pi^\text{\small NO-CURR}$",
    ], rotation=45)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


force_rewards = []
objd = []

for mname in "fc,pol,noib,nopen,nodr,nocurr".split(","):
    with open(f"{os.environ['HOME']}/fc_models/{mname}_eval.pkl", "rb") as f:
        data = pickle.load(f)
        force_rewards.append(np.array(data["res"]).reshape((-1)))
        objd.append(np.array(data["objd"]).reshape((-1))*1000)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6.0,4.0))

box  = ax1.boxplot(force_rewards, showfliers=False, widths=0.25, patch_artist=True)
configure_boxes(box, ax1)

ax1.set_title("Force Reward")
ax2.set_title("Object Displacement")

a1ylim = list(ax1.get_ylim())
a1ylim[1] = 140
ax1.set_ylim(a1ylim)

a2ylim = [-0.13, 5]
a2ylim[1] = 5
ax2.set_ylim(a2ylim)

ax1.set_ylabel(r"$r^\text{force}$")
ax2.set_ylabel(r"$\Delta o_y$ [mm]")

objd_bpl  = ax2.boxplot(objd, widths=0.25, showfliers=False, patch_artist=True)
configure_boxes(objd_bpl, ax2)

plt.savefig(f"{model_path}/box.pdf")
plt.show()