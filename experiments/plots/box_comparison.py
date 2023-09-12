import time
import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path, get_cumr
from learning_fc.utils import find_latest_model_in_path
from learning_fc.models import ForcePI
from learning_fc.training.evaluation import make_eval_env_model

dist = 0.2
width = 0.25
ntrials = 100
# kappas = [0.5, 0.6]#, 0.7, 0.8]
kappas = np.arange(0,1.1,0.1)

trial = find_latest_model_in_path(model_path, filters=["ppo"])
env, model, _, _ = make_eval_env_model(trial, with_vis=0, checkpoint="best")

fc = ForcePI(env)

start = time.time()
fc_res = []
for kappa in kappas:
    fc_res.append([
        get_cumr(env, fc) for _ in range(ntrials)
    ])
print(f"fc took {time.time()-start}")

start = time.time()
pol_res = []
for kappa in kappas:
    pol_res.append([
        get_cumr(env, model) for _ in range(ntrials)
    ])
print(f"pol took {time.time()-start}")

fig, ax = plt.subplots()

fc_bpl  = ax.boxplot(fc_res,  widths=width, positions=np.arange(len(kappas))+1-dist, patch_artist=True)
pol_bpl = ax.boxplot(pol_res, widths=width, positions=np.arange(len(kappas))+1+dist, patch_artist=True)

for bpls, color in zip((fc_bpl, pol_bpl), ("pink", "lightblue")):
    for patch in bpls['boxes']: patch.set_facecolor(color)

ax.legend([fc_bpl["boxes"][0], pol_bpl["boxes"][0]], [f'FC {int(np.mean(fc_res))}±{int(np.std(fc_res))}', f'π {int(np.mean(pol_res))}±{int(np.std(pol_res))}'])
ax.set_xticks(np.arange(len(kappas))+1)
ax.set_xticklabels([str(round(k,2)) for k in kappas])

for lx in np.arange(0,10,1)+1.5: ax.axvline(lx, ls="--", lw=0.7, c="grey")
ax.set_xlim([1-dist-width, len(kappas)+dist+width])
ax.set_ylim(100,200)

fig.tight_layout()
plt.show()