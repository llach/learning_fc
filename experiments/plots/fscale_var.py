import os
import learning_fc
import numpy as np

import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE

nsteps = 150
env = GripperTactileEnv(
    oy_init=0,
    wo_range=[0.035, 0.035],
    noise_f=0,
    control_mode=ControlMode.PositionDelta,
    model_path="assets/pal_force.xml",
)

env.change_stiffness(1)
env.solimp = env.SOLIMP_SOFT
env.f_m = 1.5
q_low, f_low = get_q_f(env, nsteps)

env.change_stiffness(0)
env.solimp = env.SOLIMP_HARD
env.f_m = 3.1
_, f_high = get_q_f(env, nsteps)
print(f"fmax {np.max(f_high)} | dpmax {0.035-q_low[-1,0]:.4f}")

env.change_stiffness(0.5)
env.f_m = 2.3
env.solimp = np.mean([env.SOLIMP_SOFT, env.SOLIMP_HARD], axis=0)
_, f_mid = get_q_f(env, nsteps)


mode = PLOTMODE.paper
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

xs = np.arange(nsteps)

mq,  = ax.plot(f_mid[:,0], c=Colors.fscale_var)
qvar = ax.fill_between(xs, np.min(f_low, axis=1), np.max(f_high, axis=1), color=Colors.fscale_var, alpha=0.3, lw=0)

legend_items = [
    [(mq,), (qvar,)],
    [
        r"$\kappa = 0.5$" if tex else "interval center", 
        r"$\kappa \in \{0,1\}$" if tex else "interval borders"
    ]
]

setup_axis(
    ax, 
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$f$" if tex else "f", 
    xlim=[0, 30], 
    ylim=[-0.002, 1.0],
    legend_items=legend_items,
    legend_loc="lower right",
    remove_first_ytick=True,
    # yticks=np.linspace(0,7,8)*0.1,
    # yticklabels=['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'],
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig(f"{model_path}/kappa_var")