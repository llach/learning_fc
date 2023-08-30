import learning_fc
import numpy as np

import matplotlib.pyplot as plt

from learning_fc.envs import GripperTactileEnv
from learning_fc.enums import ControlMode
from learning_fc.utils import get_q_f
from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE


def param_trial_plot(pname, vrange, solimp, sidx, stidx, mode):
    ntrials = 10
    nsteps  = 200
    values = np.linspace(*vrange, ntrials)

    env = GripperTactileEnv(
        oy_init=0,
        wo_range=[0.03, 0.03],
        noise_f=0,
        control_mode=ControlMode.PositionDelta,
        model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
    )

    forces = np.zeros((ntrials, nsteps))
    for i in range(ntrials):
        solimp[sidx] = values[i]
        env.set_solver_parameters(solimp=solimp)

        _, f = get_q_f(env, nsteps)
        forces[i] = f[:,0]
    env.close()


    tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
    fig, ax = plt.subplots()

    lines = []
    for i, f in enumerate(forces):
        c = Colors.grey if i<stidx else Colors.purple
        lw = 1.4 if i<stidx else 1.8
        if i == stidx: print(f"highest stable {values[i-1]} | lowest instable {values[i]}")

        # peaks are scaled down to make things clearer
        if i == 9: f = np.where(f>0.25, 0.6*f, f)

        line, = ax.plot(f, lw=lw, color=c)
        lines.append(line)

    legend_items =[
        [(lines[0],), (lines[-1],)],
        ["stable", "instable"]
    ]
    if stidx == 10: legend_items=[]

    setup_axis(
        ax, 
        xlabel=r"$t$" if tex else "t", 
        ylabel=r"$f^\text{left}$" if tex else "force", 
        xlim=[0, 200], 
        ylim=[0, 0.25],
        legend_items=legend_items,
        legend_loc="lower right",
        remove_first_ytick=True,
    )

    if mode == PLOTMODE.debug: 
        plt.show()
    else:
        plt.savefig(f"{pname}_tune")

if __name__ == "__main__":
    mode =  PLOTMODE.debug

    # param_trial_plot("dmin", [0.9, 0.0], [0.95, 0.99, 0.001, 0.5, 2], sidx=0, stidx=10, mode=mode)
    # param_trial_plot("width", [0.0005, 0.07], [0.0, 0.99, 0.001, 0.5, 1], sidx=2, stidx=9, mode=mode)
    param_trial_plot("midpoint", [0.0, 0.6], [0.0, 0.99, 0.06, 0.5, 1], sidx=3, stidx=9, mode=mode)
    # param_trial_plot("power", [1, 10], [0.0, 0.99, 0.025, 0.55, 2], sidx=4, stidx=10, mode=mode)