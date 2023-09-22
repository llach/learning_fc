import os
import time
import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path, get_cumr
from learning_fc.utils import find_latest_model_in_path
from learning_fc.models import ForcePI
from learning_fc.training.evaluation import make_eval_env_model

dist = 0.1
width = 0.15
ntrials = 200
kappas = np.arange(0,1.1,0.1)

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
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams['figure.constrained_layout.use'] = True

# use LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage[T1]{fontenc}'

trial = find_latest_model_in_path(model_path, filters=["ppo"])
env, model, _, _ = make_eval_env_model(f"{model_path}/2023-09-14_10-53-25__gripper_tactile__ppo__k-3__lr-0.0006_M2_inb", with_vis=0, checkpoint="best")

fc = ForcePI(env)

start = time.time()
fc_res = []
for kappa in kappas:
    crs = []
    for _ in range(ntrials):
        fc.reset()
        crs.append(get_cumr(env, fc))
    fc_res.append(crs)
print(f"fc took {time.time()-start}")
# pol_res, noib_res, nodr_res = fc_res, fc_res, fc_res

start = time.time()
pol_res = []
env.set_attr("with_bias", True)
for kappa in kappas:
    pol_res.append([
        get_cumr(env, model) for _ in range(ntrials)
    ])
env.set_attr("with_bias", False)
print(f"pol took {time.time()-start}")

start = time.time()
noib_res = []
model.set_parameters(f"{model_path}/2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb/weights/_best_model.zip")
for kappa in kappas:
    noib_res.append([
        get_cumr(env, model) for _ in range(ntrials)
    ])
print(f"noib took {time.time()-start}")

start = time.time()
nodr_res = []
model.set_parameters(f"{model_path}/2023-09-15_08-22-36__gripper_tactile__ppo__k-3__lr-0.0006_M2_nor/weights/_best_model.zip")
for kappa in kappas:
    nodr_res.append([
        get_cumr(env, model) for _ in range(ntrials)
    ])
print(f"nodr took {time.time()-start}")

fig, ax = plt.subplots(figsize=(7.8, 5.5))

fc_bpl  = ax.boxplot(fc_res,  widths=width, positions=np.arange(len(kappas))+1-1.5*dist-1.5*width, patch_artist=True)
pol_bpl = ax.boxplot(pol_res, widths=width,   positions=np.arange(len(kappas))+1-.5*dist-.5*width, patch_artist=True)
noib_bpl = ax.boxplot(noib_res, widths=width, positions=np.arange(len(kappas))+1+.5*dist+.5*width, patch_artist=True)
nodr_bpl = ax.boxplot(nodr_res, widths=width, positions=np.arange(len(kappas))+1+1.5*dist+1.5*width, patch_artist=True)

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
plt.savefig(f"{os.environ['HOME']}/box.pdf")
# plt.show()