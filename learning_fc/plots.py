import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results

from learning_fc.plotting import Colors, set_rcParams, setup_axis, PLOTMODE, FIGTYPE, EPS_SEP_LINE_KW, SEND_LINE_KW

def f_act_plot(r, mode=PLOTMODE.debug, prefix=""):
    tex = set_rcParams(mode=mode, ftype=FIGTYPE.multirow, nrows=2)
    fig, axes = plt.subplots(nrows=2)

    ntrials = r.ntrials
    nsteps  = r.nsteps
    total_steps = ntrials*nsteps
    xs = np.arange(total_steps).reshape((ntrials, nsteps))

    forces = r.force.reshape((ntrials, nsteps, -1))
    for x, y in zip(xs, forces): # avoids discontinuity on episode end
        lline, = axes[0].plot(x, y[:,0], c=Colors.tab10_0)
        rline, = axes[0].plot(x, y[:,1], c=Colors.tab10_1)

    goals = r.goals.reshape((ntrials, nsteps))
    for x, y in zip(xs, goals):
        gline, = axes[0].plot(x, y, c=Colors.fgoal, lw=1.0)


    actions = r.actions.reshape((ntrials, nsteps, -1))
    for x, y in zip(xs, actions):
        axes[1].plot(x, y[:,0], c=Colors.tab10_0)
        axes[1].plot(x, y[:,1], c=Colors.tab10_1)

    reset_lines = ((np.arange(ntrials)+1)*nsteps)[:-1]
    for ax in axes.flatten(): 
        for rl in reset_lines: epline = ax.axvline(rl, **EPS_SEP_LINE_KW)

    setup_axis(
        axes[0], 
        xlabel=None,
        ylabel=r"$f_i$" if tex else "force", 
        xlim=[0, total_steps], 
        ylim=[-0.015, 0.7],
        remove_xticks=True,
        legend_items=[[(gline,), (epline,)], [r"$f^\text{goal}$" if tex else "fgoal", r"$T$" if tex else "T"]],
        legend_loc="upper left"
    )

    setup_axis(
        axes[1], 
        xlabel=r"$t$" if tex else "t", 
        ylabel=r"$a_i$" if tex else "action", 
        xlim=[0, total_steps], 
        ylim=[-1.02, 1.0],
        yticks=np.arange(-1,1.1,0.5),
        legend_items=[[(lline,), (rline,)], ["left", "right"]],
        legend_loc="upper left"
    )

    if mode == PLOTMODE.debug: 
        plt.show()
    else:
        plt.savefig(f"{prefix}mr_test")


def clean_lc(trialdir, params, mode=PLOTMODE.camera_ready, prefix=""):
    total_steps = params["train"]["timesteps"]
    max_send = np.max([s["dur"] for s in params["make_model"]["schedules"]]) if  len(params["make_model"]["schedules"])>0 else None

    df = load_results(trialdir)
    x = np.cumsum(df.l.values)
    y = df.r.values
    y_mean = df.r.rolling(150).mean()
    y_std = df.r.rolling(150).std()

    tex = set_rcParams(mode=mode, ftype=FIGTYPE.single)
    fig, ax = plt.subplots()

    mline, = ax.plot(x, y_mean, c=Colors.mean_r, lw=2)
    fill   = ax.fill_between(x, y_mean+y_std, y_mean-y_std, color=Colors.sigma_r, alpha=0.45, lw=0)

    legend_items=[
            [(mline,), (fill,)],
            [
                r"mean $\int r(t)$" if tex else "mean r", 
                r"$\pm$ std. dev." if tex else "± sigma",
            ]
        ]
    
    # TODO set ymin/max dynamically
    if max_send:
        sline = ax.axvline(max_send, ymin=0.58, ymax=0.89, **SEND_LINE_KW)
        legend_items[0].append((sline,))
        legend_items[1].append(r"$\max_i(s_i^\text{end})$" if tex else "max(s_i^end)",)

    if total_steps == 2e6:
        xticks=[0, 0.5e6, 1e6, 1.5e6, 2e6]
        xticklabels=["", "0.5M", "1M", "1.5M", "2M"]
    else:
        xticks=[0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6, 3.5e6, 4e6, 4.5e6]
        xticklabels=["", "", "1M", "", "2M", "", "3M", "", "4M", ""]
    
    setup_axis(
        ax, 
        xlabel="Training Steps", 
        ylabel="Episode Rewards", 
        xlim=[0,total_steps],
        ylim=[0,225],
        remove_first_ytick=False,
        xticks=xticks,
        xticklabels=xticklabels,
        legend_items=legend_items,
        legend_loc="center right",
    )

    if mode == PLOTMODE.debug: 
        plt.show()
    else:
        plt.savefig(f"{prefix}curve")
