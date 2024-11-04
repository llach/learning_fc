import os
import pickle
import learning_fc

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode
from learning_fc.envs import GripperTactileEnv

from learning_fc import plot_config, cm_uni


oname = "wood3"
files_dir = f"{model_path}/data/"

qs = []
for fi in os.listdir(files_dir):
    if oname not in fi: continue 

    with open(f"{files_dir}{fi}", "rb") as f:
        data = pickle.load(f)
    
    q = np.array(data["obs"]["q"])
    qs.append(q)

q_rob = qs[-1]
n_steps = len(q_rob)
wo = 0.018

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    wo_range=[wo, wo],
    model_path="assets/pal_force.xml",
    noise_f=0.002,
    f_scale=3.1,
    sample_biasprm=False,
    randomize_stiffness=False,
)

env.biasprm = env.BIASPRM_RANGE[1]
q_low, _ = get_q_f(env, n_steps)

env.biasprm = env.BIASPRM_RANGE[0]
q_high, _ = get_q_f(env, n_steps)

env.biasprm = env.BIASPRM
q_env, _ = get_q_f(env, n_steps)

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    wo_range=[wo, wo],
)

fig, ax = plt.subplots(figsize=(5.5/1.25,4/1.25))
xs = np.arange(n_steps)

rq0, = ax.plot(xs, q_rob[:,0], c="#9a031e", lw=1.2, label="real robot")

c=cm_uni["darkgreen"]
mq,  = ax.plot(q_env[:,0], c=c, label=r"$b_2 = -9$")
qvar = ax.fill_between(xs, np.min(q_low, axis=1), np.max(q_high, axis=1), color=c, alpha=0.3, lw=0, label=r"$b_2 \in [-13, -6]$")

ax.set_xlim(0,45)
ax.set_ylim(0,0.048)

ax.set_xlabel(r"Steps")
ax.set_ylabel(r"Joint Position \, $q$ [$m$]")

leg = ax.legend()
leg.get_frame().set_linewidth(0.3)

plt.savefig(f"{os.environ['HOME']}/repos/diss/images/rl_ctrl/act_var.pdf")
plt.show()