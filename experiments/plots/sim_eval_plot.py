import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sim_eval_data import dist, width, kappas

 # legend 
plt.rcParams['legend.fancybox']  = False
plt.rcParams['legend.edgecolor'] = "#6C6C6D"

# # axes
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
# plt.rcParams['lines.linewidth'] = 1.5

plt.rcParams['font.size'] = 23
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17

# plt.rcParams['font.family'] = 'serif'

# high quality figure
plt.rcParams["figure.dpi"] = 100
# plt.rcParams["savefig.format"] = "pdf"
# plt.rcParams['figure.constrained_layout.use'] = True

# use LaTeX
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage[T1]{fontenc}'

def configure_boxes(box, ax):

    # Customizing box colors
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#FF5722']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Customizing whiskers and caps
    for whisker, cap in zip(box['whiskers'], box['caps']):
        whisker.set(color='#757575', linewidth=1.5)
        cap.set(color='#757575', linewidth=1.5)

    # Customizing median lines
    for median in box['medians']:
        median.set(color='#212121', linewidth=2)

    # Customizing outliers
    for flier in box['fliers']:
        flier.set(marker='o', color='#757575', alpha=0.5)

    # Adding grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Customizing tick labels
    ax.set_xticklabels([
        f'Baseline', 
        r"$\pi^{IB}$",
        r"$\pi^{NO-IB}$",
        r"$\pi^{NO-RAND}$",
    ], fontsize=12)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


force_rewards = []
objd = []

for mname in "fc,pol,noib,nodr".split(","):
    with open(f"{os.environ['HOME']}/{mname}_eval.pkl", "rb") as f:
        data = pickle.load(f)
        force_rewards.append(np.array(data["res"]).reshape((-1)))
        objd.append(np.array(data["objd"]).reshape((-1)))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7.8, 5.5))

box  = ax1.boxplot(force_rewards, showfliers=False, widths=0.25, patch_artist=True)
configure_boxes(box, ax1)

# for patch, color in zip(rew_bpl["boxes"], ("pink", "lightblue", "#05B24A", "#c77dff")):
#     patch.set_facecolor(color)

# ax1.legend(
#     [
#         fc_bpl["boxes"][0], 
#         pol_bpl["boxes"][0],
#         noib_bpl["boxes"][0],
#         nodr_bpl["boxes"][0], 
#     ], 
#     [
#         f'Baseline', 
#         r"$\pi^{IB}$",
#         r"$\pi^{NO-IB}$",
#         r"$\pi^{NO-RAND}$",
#     ],
#     loc='upper center', 
#     bbox_to_anchor=(0.5, 1.13),
#     ncol=4,
# )

ax1.set_ylabel(r"$r$")

objd_bpl  = ax2.boxplot(objd, widths=0.25, showfliers=False, patch_artist=True)
configure_boxes(objd_bpl, ax2)

fig.tight_layout()
# plt.savefig(f"{os.environ['HOME']}/box.pdf")
plt.show()