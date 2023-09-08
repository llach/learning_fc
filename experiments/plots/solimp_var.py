import learning_fc
import numpy as np

import matplotlib.pyplot as plt

from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import get_q_f
from learning_fc.enums import ControlMode
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE


nsteps = 50
env = GripperTactileEnv(
    oy_init=0,
    wo_range=[0.035, 0.035],
    noise_f=0,
    control_mode=ControlMode.PositionDelta,
    model_path="assets/pal_force.xml",
)
env.f_m = 1

env.solimp = env.SOLIMP_SOFT
env.solimp[0] = 0
_, f_low = get_q_f(env, nsteps)

env.solimp = env.SOLIMP_HARD
env.solimp[0] = 0
_, f_high = get_q_f(env, nsteps)

env.solimp = np.mean([env.SOLIMP_SOFT, env.SOLIMP_HARD], axis=0)
env.solimp[0] = 0
_, f_mid = get_q_f(env, nsteps)


mode = PLOTMODE.paper
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

xs = np.arange(nsteps)

mq,  = ax.plot(f_mid[:,0], c=Colors.solimp_var)
qvar = ax.fill_between(xs, np.min(f_low, axis=1), np.max(f_high, axis=1), color=Colors.solimp_var, alpha=0.3, lw=0)

legend_items = [
    [(mq,), (qvar,)],
    [
        r"$\rho = 0.0065$" if tex else "interval center", 
        r"$\rho \in \{0.003, 0.01\}$" if tex else "interval borders"
    ]
]

setup_axis(
    ax, 
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$f$" if tex else "f", 
    xlim=[0, 30], 
    ylim=[-0.002, 0.35],
    legend_items=legend_items,
    legend_loc="lower right",
    remove_first_ytick=True,
    # yticks=np.linspace(0,25,6)*0.01,
    # yticklabels=['', '0.05', '0.10', '0.15', '0.20', '0.25'],
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig("solimp_var")