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
plt.rcParams['lines.linewidth'] = 1.5

plt.rcParams['font.size'] = 23
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17

plt.rcParams['font.family'] = 'serif'

# high quality figure
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams['figure.constrained_layout.use'] = True

# use LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage[T1]{fontenc}'


with open(f"{os.environ['HOME']}/sim_eval.pkl", "rb") as f:
    data = pickle.load(f)

fc_res = data["fc_res"]
fc_objd = data["fc_objd"]
pol_res = data["pol_res"]
pol_objd = data["pol_objd"]
noib_res = data["noib_res"]
noib_objd = data["noib_objd"]
nodr_res = data["nodr_res"]
nodr_objd = data["nodr_objd"]

fig, ax = plt.subplots(figsize=(7.8, 5.5))

fc_bpl  = ax.boxplot(fc_res, widths=width, positions=np.arange(len(kappas))+1-1.5*dist-1.5*width, patch_artist=True)
pol_bpl = ax.boxplot(list(pol_res), widths=width, positions=np.arange(len(kappas))+1-.5*dist-.5*width, patch_artist=True)
noib_bpl = ax.boxplot(list(noib_res), widths=width, positions=np.arange(len(kappas))+1+.5*dist+.5*width, patch_artist=True)
nodr_bpl = ax.boxplot(list(nodr_res), widths=width, positions=np.arange(len(kappas))+1+1.5*dist+1.5*width, patch_artist=True)

for bpls, color in zip((fc_bpl, pol_bpl, noib_bpl, nodr_bpl), ("pink", "lightblue", "#05B24A", "#c77dff")):
    for patch in bpls['boxes']: patch.set_facecolor(color)

ax.legend(
    [
        fc_bpl["boxes"][0], 
        pol_bpl["boxes"][0],
        noib_bpl["boxes"][0],
        nodr_bpl["boxes"][0], 
    ], 
    [
        f'Baseline', 
        r"$\pi^{IB}$",
        r"$\pi^{NO-IB}$",
        r"$\pi^{NO-RAND}$",
    ],
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.13),
    ncol=4,
)
ax.set_xticks(np.arange(len(kappas))+1)
ax.set_xticklabels([str(round(k,2)) for k in kappas])

for lx in np.arange(0,10,1)+1.5: ax.axvline(lx, ls="--", lw=0.7, c="grey")
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$r$")
# ax.set_xlim([0+3*dist+2*width, len(kappas)+1])
ax.set_ylim(0,150)
ax.set_xlim(0.325, 11.65)
print(ax.get_xlim())

# fig.tight_layout()
# plt.savefig(f"{os.environ['HOME']}/box.pdf")
plt.show()