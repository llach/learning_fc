import os
import pickle
import learning_fc

import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode
from learning_fc.envs import GripperTactileEnv
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE

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
env.biasprm = [0, -100, -10]
q_bef, _ = get_q_f(env, n_steps)

mode = PLOTMODE.paper
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()
xs = np.arange(n_steps)

eq1, _ = ax.plot(xs, q_bef, c=Colors.grey)

mq,  = ax.plot(q_env[:,0], c=Colors.act_var_mean)
qvar = ax.fill_between(xs, np.min(q_low, axis=1), np.max(q_high, axis=1), color=Colors.act_var_var, alpha=0.3, lw=0)

rq0, = ax.plot(xs, q_rob[:,0], c=Colors.tab10_0, lw=2)
rq1, = ax.plot(xs, q_rob[:,1], c=Colors.tab10_1, lw=2)

legend_items = [
    [(rq0, rq1), (eq1,), (mq,), (qvar,)],
    ["robot", "sim before", "sim after", "variation"]
]

setup_axis(
    ax, 
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$q_i$" if tex else "q_i", 
    xlim=[0, 50], 
    ylim=[0, 0.045],
    legend_items=legend_items,
    legend_loc="upper right",
    remove_first_ytick=True,
    yticks=np.linspace(0,45,10)*0.001,
    yticklabels=['0.0', '', '0.01', '', '0.02', '', '0.03', '', '0.04', ''],
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig("act_var")