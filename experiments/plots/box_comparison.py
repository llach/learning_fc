import time
import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path, get_cumr
from learning_fc.utils import find_latest_model_in_path
from learning_fc.models import ForcePI
from learning_fc.training.evaluation import make_eval_env_model
from learning_fc.plotting import set_rcParams, PLOTMODE

dist = 0.1
width = 0.15
ntrials = 50
# kappas = [0.5, 0.6]#, 0.7, 0.8]
kappas = np.arange(0,1.1,0.1)

trial = find_latest_model_in_path(model_path, filters=["ppo"])
env, model, _, _ = make_eval_env_model(f"{model_path}/2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb", with_vis=0, checkpoint="best")

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

start = time.time()
pol_res = []
for kappa in kappas:
    pol_res.append([
        get_cumr(env, model) for _ in range(ntrials)
    ])
print(f"pol took {time.time()-start}")

start = time.time()
noib_res = []
model.set_parameters(f"{model_path}/2023-09-14_10-53-25__gripper_tactile__ppo__k-3__lr-0.0006_M2_inb/weights/_best_model.zip")
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

# set_rcParams(mode=PLOTMODE.paper)

fig, ax = plt.subplots(figsize=[7, 5])

fc_bpl  = ax.boxplot(fc_res,  widths=width, positions=np.arange(len(kappas))+1-1.5*dist-1.5*width, patch_artist=True)
pol_bpl = ax.boxplot(pol_res, widths=width,   positions=np.arange(len(kappas))+1-.5*dist-.5*width, patch_artist=True)
noib_bpl = ax.boxplot(noib_res, widths=width, positions=np.arange(len(kappas))+1+.5*dist+.5*width, patch_artist=True)
nodr_bpl = ax.boxplot(nodr_res, widths=width, positions=np.arange(len(kappas))+1+1.5*dist+1.5*width, patch_artist=True)

for bpls, color in zip((fc_bpl, pol_bpl, noib_bpl, nodr_bpl), ("pink", "lightblue", "green", "cyan")):
    for patch in bpls['boxes']: patch.set_facecolor(color)

ax.legend(
    [
        fc_bpl["boxes"][0], 
        pol_bpl["boxes"][0],
        noib_bpl["boxes"][0],
        nodr_bpl["boxes"][0], 
    ], 
    [
        f'FC {int(np.mean(fc_res))} {int(np.std(fc_res))}', 
        f'PI {int(np.mean(pol_res))} {int(np.std(pol_res))}',
        f'PI-NOIB {int(np.mean(noib_res))} {int(np.std(noib_res))}',
        f'PI-NODR {int(np.mean(nodr_res))} {int(np.std(nodr_res))}',
    ],
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.11),
    ncol=4,
)
ax.set_xticks(np.arange(len(kappas))+1)
ax.set_xticklabels([str(round(k,2)) for k in kappas])

for lx in np.arange(0,10,1)+1.5: ax.axvline(lx, ls="--", lw=0.7, c="grey")
ax.set_xlabel("kappa")
ax.set_ylabel("reward")
# ax.set_xlim([0+3*dist+2*width, len(kappas)+1])
ax.set_ylim(0,150)

fig.tight_layout()
plt.show()