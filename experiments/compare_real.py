import os
import pickle
import learning_fc

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.envs import GripperTactileEnv

from matplotlib.legend_handler import HandlerTuple

def get_q_f(env, n_steps):
    q_env = []
    f_env = []
    env.reset()
    for _ in range(n_steps):
        q_env.append(env.q)
        f_env.append(env.force)
        env.step ([-1,-1])
    q_env = np.array(q_env)
    f_env = np.array(f_env)
    return q_env, f_env


ftheta = 0.0075
oname = "wood_mid"
grasp_type = "dq"
# grasp_type = "power"
files_dir = f"{model_path}/data/{grasp_type}/"

tg = []
wo = []
fmax = []
forces = []
qs = []
dts = []

for fi in os.listdir(files_dir):
    if oname not in fi: continue 

    with open(f"{files_dir}{fi}", "rb") as f:
        data = pickle.load(f)
    
    t = np.array(data["timestamps"])
    q = np.array(data["obs"]["q"])
    f = np.array(data["obs"]["f"])

    tg.append(np.argmax(np.all(f>ftheta, axis=1)))
    fmax.append(np.max(f, axis=0))
    wo.append(np.abs(np.sum(q[tg[-1]])))

    qs.append(q)
    forces.append(f)

    dts.append(np.array([
        [(t[i]-t[i-1]).total_seconds(), (t[i]-t[i-1]).total_seconds()]
        for i in np.arange(1,len(t))
    ]))

f_rob = forces[-1]
q_rob = qs[-1]
n_steps = len(q_rob)
wo = np.median(wo)/2

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    wo_range=[wo, wo],
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
    noise_f=0.002,
    f_scale=3.1,
)

env.biasprm = env.BIASPRM_RANGE[1]
q_low, _ = get_q_f(env, n_steps)

env.biasprm = env.BIASPRM_RANGE[0]
q_high, _ = get_q_f(env, n_steps)

env.biasprm = env.BIASPRM
q_env, f_env = get_q_f(env, n_steps)

env = GripperTactileEnv(
    control_mode=ControlMode.PositionDelta,
    oy_init=0,
    wo_range=[wo, wo],
)
env.biasprm = [0, -500, -50]
q_bef, f_bef = get_q_f(env, n_steps)

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(13,8))

xs = np.arange(n_steps)


rq1, rq2 = axes[0,0].plot(xs, q_rob)
eq1, eq2 = axes[0,0].plot(xs, q_bef)
axes[0,0].set_ylim(-0.001, 0.049)
axes[0,0].legend([(rq1, rq2), (eq1, eq2)], ['robot', "sim"],
               handler_map={tuple: HandlerTuple(ndivide=None)})

rf1, rf2 = axes[0,1].plot(xs, f_rob)
ef1, ef2 = axes[0,1].plot(xs, f_bef)
axes[0,1].set_ylim(-0.05, 1.2)
axes[0,1].axhline(np.max(f_rob), c="red", ls="dashed", lw=0.7)
axes[0,1].legend([(rf1, rf2), (ef1, ef2)], ['robot', "sim"],
               handler_map={tuple: HandlerTuple(ndivide=None)})

""" second row
"""

rq1, rq2 = axes[1,0].plot(xs, q_rob)
eq1, eq2 = axes[1,0].plot(xs, q_env)
axes[1,0].fill_between(xs, np.min(q_low, axis=1), np.max(q_high, axis=1), color="red", alpha=0.2)
axes[1,0].set_ylim(-0.001, 0.049)
axes[1,0].legend([(rq1, rq2), (eq1, eq2)], ['robot', "sim"],
               handler_map={tuple: HandlerTuple(ndivide=None)})

rf1, rf2 = axes[1,1].plot(xs, f_rob)
ef1, ef2 = axes[1,1].plot(xs, f_env)
axes[1,1].set_ylim(-0.05, 1.2)
axes[1,1].axhline(np.max(f_rob), c="red", ls="dashed", lw=0.7)
axes[1,1].legend([(rf1, rf2), (ef1, ef2)], ['robot', "sim"],
               handler_map={tuple: HandlerTuple(ndivide=None)})

for ax in axes.flatten():
    ax.axvline(tg[-1], c="grey", ls="dashed", lw=0.7)
fig.suptitle(oname)
fig.tight_layout()
plt.show()