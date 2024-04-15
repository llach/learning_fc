import os
import json
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results
from learning_fc import plot_config, model_path, cm_itten, cm_uni
from matplotlib.legend_handler import HandlerTuple
from matplotlib import colormaps as cm
cmap = cm["tab10"]


if __name__ == "__main__":
    trial = f"{model_path}/2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb"

    with open(f"{trial}/parameters.json", "r") as f:
        params = json.load(f)
    
    total_steps = params["train"]["timesteps"]
    max_send = np.max([s["dur"] for s in params["make_model"]["schedules"]]) if  len(params["make_model"]["schedules"])>0 else None

    df = load_results(trial)
    x = np.cumsum(df.l.values)
    y = df.r.values
    y_mean = df.r.rolling(150).mean()
    y_std = df.r.rolling(150).std()

    fig, ax = plt.subplots(figsize=(8/1.5,5.5/1.5))

    mline, = ax.plot(x, y_mean, lw=1, c=cm_itten["blue"])
    fill   = ax.fill_between(x, y_mean+y_std, y_mean-y_std, alpha=0.45, lw=0, color=cm_itten["blue"])

    legend_items=[
            [(mline,), (fill,)],
            [
                r"$\sum_t r(t)$", 
                r"$\pm\,\sigma$",
            ]
        ]
    
    if max_send:
        sline = ax.axvline(max_send, ymin=0.65, ymax=0.98, c=cmap(7), ls="dashed")
        legend_items[0].append((sline,))
        legend_items[1].append(r"$\max(s_\text{end})$",)

    ax.set_xlim(0,3.5e6)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], 150)

    xticks=[0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6, 3.5e6]
    xticklabels=["", "", "1M", "", "2M", "", "3M", ""]

    ax.set_xticks(xticks) 
    ax.set_xticklabels(xticklabels)

    legend = ax.legend(
        *legend_items,
        loc="lower right",
        handler_map={tuple: HandlerTuple(ndivide=None)}
    )
    legend.get_frame().set_linewidth(0.3)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Episode Reward")
    
    plt.savefig(f"{os.environ['HOME']}/repos/diss/images/rl_ctrl/curve.pdf")
    plt.show()

    # setup_axis(
    #     ax, 
    #     xlabel="Training Steps", 
    #     ylabel="Episode Rewards", 
    #     xlim=[0,total_steps],
    #     ylim=[0,200],
    #     remove_first_ytick=False,
    #     xticks=xticks,
    #     xticklabels=xticklabels,
    #     legend_items=legend_items,
    #     legend_loc="center right",
    # )

    # if mode == PLOTMODE.debug: 
    #     plt.show()
    # else:
    #     