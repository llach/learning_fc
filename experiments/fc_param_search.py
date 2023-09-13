import time
import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path, get_cumr
from learning_fc.utils import find_latest_model_in_path
from learning_fc.models import ForcePI
from learning_fc.training.evaluation import make_eval_env_model

ntrials = 20
nparams = 10
Kis = np.linspace(0.2, 3.0, nparams)
Kps = np.linspace(0.05, 1.2, nparams)

trial = find_latest_model_in_path(model_path, filters=["ppo"])
env, model, _, _ = make_eval_env_model(trial, with_vis=0, checkpoint="best")

res = []
cumrs = []

start = time.time()
i = 0
for Ki in Kis:
    for Kp in Kps:
        fc = ForcePI(env, Kp=Kp, Ki=Ki)
        print(f"trial {i+1}: {fc}")
        
        # rollout fc ntrial times
        rews = [get_cumr(env, model) for _ in range(ntrials)]
        r_mean = np.mean(rews)
        r_std  = np.std(rews)

        res.append([[r_mean, r_std], [Kp, Ki]])
        cumrs.append(r_mean)
        i += 1

res = np.array(res)
cumrs = np.array(cumrs)

sis = np.argsort(cumrs)[::-1]

for re in res[sis]:
    m, std = re[0]
    Kp, Ki = re[1]
    print(f"{m:.2f}±{int(std)} | Kp={Kp:.2f} Ki={Ki:.2f}")

print(f"search took {int(time.time()-start)}s")
