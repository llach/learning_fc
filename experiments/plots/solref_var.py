import learning_fc
import numpy as np

import matplotlib.pyplot as plt

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
    model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
)
env.solimp = env.SOLIMP_SOFT

env.solref = env.SOLREF_SOFT
_, f_low = get_q_f(env, nsteps)

env.solref = env.SOLREF_HARD
_, f_high = get_q_f(env, nsteps)

env.solref = np.mean([env.SOLREF_SOFT, env.SOLREF_HARD], axis=0)
_, f_mid = get_q_f(env, nsteps)


mode = PLOTMODE.debug
tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
fig, ax = plt.subplots()

xs = np.arange(nsteps)

mq,  = ax.plot(f_mid[:,0], c=Colors.solref_var)
qvar = ax.fill_between(xs, np.min(f_low, axis=1), np.max(f_high, axis=1), color=Colors.solref_var, alpha=0.3, lw=0)

legend_items = [
    [(mq,), (qvar,)],
    ["interval center", "interval borders"]
]

setup_axis(
    ax, 
    xlabel=r"$t$" if tex else "t", 
    ylabel=r"$f_i$" if tex else "f", 
    xlim=[0, 80], 
    ylim=[-0.002, 0.25],
    legend_items=legend_items,
    legend_loc="center right",
    yticks=np.linspace(0,25,6)*0.01,
    yticklabels=['', '0.05', '0.10', '0.15', '0.20', '0.25'],
)

if mode == PLOTMODE.debug: 
    plt.show()
else:
    plt.savefig("solref_var")