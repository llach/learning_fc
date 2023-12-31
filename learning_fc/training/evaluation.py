import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple
from stable_baselines3.common.results_plotter import load_results, window_func, X_TIMESTEPS, EPISODES_WINDOW, ts2xy

from learning_fc.envs import GripperTactileEnv
from learning_fc.utils import CsvReader, safe_unwrap
from learning_fc.models import ForcePI, PosModel
from learning_fc.training import make_env, make_model
from learning_fc.plots import clean_lc, f_act_plot, PLOTMODE, Colors

TACTILE_ENV_MEMBERS = [
    "force_deltas", 
    "r_force",
    "r_obj_pos",
    "r_obj_prox",
    "r_act",
    "obj_v",
    "obj_pos",
    "f_scale",
]

POSITION_ENV_MEMBERS = [
    "r_pos",
    "r_act"
]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def safe_last_cumr(res):
    try: return np.array(res["cumr"])[:,-1]
    except: return np.array([cr[-1] for cr in res["cumr"]])

def rollout_model(env, model, vis, n_rollouts, reset_cb=None, before_step_cb=None, after_step_cb=None):
    results = dict(
        q=[[] for _ in range(n_rollouts)],
        qdes=[[] for _ in range(n_rollouts)],
        qdot=[[] for _ in range(n_rollouts)],
        qacc=[[] for _ in range(n_rollouts)],
        force=[[] for _ in range(n_rollouts)],
        actions=[[] for _ in range(n_rollouts)],
        in_contact=[[] for _ in range(n_rollouts)],
        r=[[] for _ in range(n_rollouts)],
        cumr=[[] for _ in range(n_rollouts)],
        eps_rew=[],
        goals=[],
    )
    for i in range(n_rollouts):
        obs, _ = env.reset()

        if reset_cb: results = reset_cb(env, model, i, results) or results
        if vis: vis.reset()

        done = False
        results["goals"].append(env.get_goal())
        nsteps = 0
        while not done:
            nsteps +=1
            if before_step_cb: results = before_step_cb(env, model, i, results) or results

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action.copy())
            done = terminated or truncated

            results["q"][i].append(env.q)
            results["qdes"][i].append(env.qdes)
            results["qdot"][i].append(env.qdot)
            results["qacc"][i].append(env.qacc)
            results["force"][i].append(env.force)
            results["in_contact"][i].append(env.in_contact)
            results["r"][i].append(reward)
            results["cumr"][i].append(sum(results["r"][i]))
            results["actions"][i].append(action)

            if vis: vis.update_plot(action=action, reward=reward)
            if after_step_cb: results = after_step_cb(env, model, i, results) or results
        results["eps_rew"].append(results["cumr"][i][-1])

    for k, v in results.items():
        if k in ["goals", "eps_rew"]: continue # goals are a special case, see below the loop
        if isinstance(v[0], list) and isinstance(v[0][0], np.ndarray):
            results[k] = np.concatenate(v)
        else:
            results[k]  = np.concatenate(v).reshape((-1,))

    results["eps_rew"] = np.array(results["eps_rew"]).reshape((-1,))
    results["goals"] = np.repeat(np.array(results["goals"]).reshape((-1,)), nsteps)
    results["nsteps"] = nsteps
    results["ntrials"] = n_rollouts

    return AttrDict(results)


def deterministic_eval(env, model, vis, goals, reset_cb=None, before_step_cb=None, after_step_cb=None):
    n_rollouts = len(goals)
    def set_env_goal(env, model, i, results, **kw):
        env.set_goal(goals[i])

        if reset_cb: results = reset_cb(env, model, i, results, **kw) or results
        return results

    return rollout_model(
        env, model, vis, n_rollouts, 
        reset_cb=set_env_goal, 
        before_step_cb=before_step_cb, 
        after_step_cb=after_step_cb
    )

def force_reset_cb(env, model, i, results, **kw): 
    if isinstance(model, ForcePI): model.reset()

    for key in TACTILE_ENV_MEMBERS:
        if key not in results: results |= {key: []}
        results[key].append([])

    return results

def force_after_step_cb(env, model, i, results, goal=None, **kw):
    for key in TACTILE_ENV_MEMBERS:  results[key][i].append(getattr(env, key))
    return results

def pos_reset_cb(env, model, i, results, **kw): 
    for key in POSITION_ENV_MEMBERS:
        if key not in results: results |= {key: []}
        results[key].append([])

    return results

def pos_after_step_cb(env, model, i, results, goal=None, **kw):
    for key in POSITION_ENV_MEMBERS:  results[key][i].append(getattr(env, key))
    return results

def _get_checkpoint(chkp, trialdir):
    if chkp == "best":
        return "_best_model"
    elif chkp == "latest":
        fn, nstep = [], []
        for fi in os.listdir(f"{trialdir}/weights"):
            if fi.endswith(".zip") and "best" not in fi and "model" in fi:
                fn.append(fi)
                nstep.append(int(fi[:-4].replace("model", "")))
        return fn[np.argmax(nstep)]
    else: return chkp

def make_eval_env_model(trialdir, with_vis=False, env_override={}, checkpoint="best"):
    # load parameters
    with open(f"{trialdir}/parameters.json", "r") as f:
        params = json.load(f)

    # modify env creation parameters for eval and create
    params["make_env"] = {**params["make_env"], **dict(training=False, nenv=1, with_vis=with_vis)}
    params["make_env"]["env_kw"] = {**params["make_env"].pop("init_params"), **params["make_env"]["env_kw"]}
    params["make_env"]["env_kw"] = {**params["make_env"]["env_kw"], **env_override}
    env, vis, _ = make_env(**params["make_env"])

    # set the final values of scheduled parameters
    if "schedules" in params["train"]:
        for sc in params["train"]["schedules"]:
            if sc["var_name"] in env_override.keys(): continue # override prevents loading final parameter value
            env.set_attr(sc["var_name"], sc["final_value"])

    # same for the model
    checkpoint = _get_checkpoint(checkpoint, trialdir)
    params["make_model"] = {**params["make_model"] , **dict(training=False, weights=f"{trialdir}/weights/{checkpoint}")}
    params["make_model"]["model_kw"] = {**params["make_model"]["model_kw"], **params["make_model"].pop("init_params") , **params["make_model"].pop("mkw")}
    params["make_model"]["logdir"] = trialdir # in case folder was renamed 
    model, _, _ = make_model(env, **params["make_model"])

    return env, model, vis, params

def agent_oracle_comparison(env, agent, oracle, vis, goals, reset_cb=None, after_step_cb=None, plot=True, plot_title="", trialdir=None):

    print("baseline evaluation")
    oracle_results = deterministic_eval(env, oracle, vis, goals, reset_cb=reset_cb, after_step_cb=after_step_cb)

    print("policy evaluation")
    agent_results = deterministic_eval(env, agent, vis, goals, reset_cb=reset_cb, after_step_cb=after_step_cb)

    cumr_a = agent_results.eps_rew
    cumr_o = oracle_results.eps_rew

    mean_a = np.mean(cumr_a)
    mean_o = np.mean(cumr_o)

    std_a = np.std(cumr_a)
    std_o = np.std(cumr_o)

    print(f"RL   {mean_a:.0f}±{std_a:.1f}")
    print(f"BASE {mean_o:.0f}±{std_o:.1f}")

    if plot:
        plt.figure(figsize=(6.5,7), layout="constrained")

        plt.scatter(goals, cumr_a, label=f"pol  {np.mean(cumr_a):.0f}+-{np.std(cumr_a):.1f}")
        plt.scatter(goals, cumr_o, label=f"base {np.mean(cumr_o):.0f}+-{np.std(cumr_o):.1f}")

        plt.axhline(np.mean(cumr_a), c=Colors.tab10_0)
        plt.axhline(np.mean(cumr_o), c=Colors.tab10_1)

        plt.fill_between(goals, mean_a-std_a, mean_a+std_a, color=Colors.tab10_0, alpha=0.2)
        plt.fill_between(goals, mean_o-std_o, mean_o+std_o, color=Colors.tab10_1, alpha=0.2)

        plt.title(plot_title)
        plt.xlabel("target")
        plt.ylim(0,200)
        plt.ylabel("cumulative episode reward") 

        plt.legend()

    return agent_results, oracle_results


def plot_rollouts(env, r, plot_title):
    n_trials = r.ntrials
    n_steps  = r.nsteps
    x = np.arange(r.q.shape[0])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14.5, 8.8), layout="constrained")

    axes[0,0].set_title("joint position")
    q1,  = axes[0,0].plot(x, r.q[:,0], lw=1, label="qdes")
    q2,  = axes[0,0].plot(x, r.q[:,1], lw=1)
    qd1, = axes[0,0].plot(x, r.qdes[:,0], lw=1)
    qd2, = axes[0,0].plot(x, r.qdes[:,1], lw=1)
    axes[0,0].legend([(q1, q2), (qd1, qd2)], ['q', 'qdes'],
               handler_map={tuple: HandlerTuple(ndivide=None)})

    axes[0,1].set_title("forces")
    axes[0,1].plot(x, r.force[:,0], lw=1)
    axes[0,1].plot(x, r.force[:,1], lw=1)
    axes[0,1].plot(r.goals, c="grey", label="fgoal")
    axes[0,1].legend()

    axes[1,0].set_title("joint velocity")
    axes[1,0].plot(x, r.qdot[:,0], lw=1)
    axes[1,0].plot(x, r.qdot[:,1], lw=1)
    axes[1,0].set_ylim(-1.1*env.vmax, 1.1*env.vmax)
    axes[1,0].axhline(-env.vmax, lw=0.7, ls="dashed", c="grey")
    axes[1,0].axhline(env.vmax, lw=0.7, ls="dashed", c="grey")

    axes[1,1].set_title("actions")
    axes[1,1].plot(x, r.actions, lw=1)
    axes[1,1].set_ylim(-1.5, 1.05)
    axes[1,1].axhline(-1, lw=0.7, ls="dashed", c="grey")
    axes[1,1].axhline(1, lw=0.7, ls="dashed", c="grey")
    axes[1,1].axhline(0, lw=0.7, ls="dashed", c="grey")

    axes[2,0].set_title("partial rewards")
    axes[2,0].plot(x, r.r_force, lw=1, label="r_force", c="cyan")
    axes[2,0].plot(x, r.r_obj_pos, lw=1, label="r_obj_pos", c="orange")
    axes[2,0].plot(x, r.r_obj_prox, lw=1, label="r_obj_prox", c="red")
    axes[2,0].plot(x, r.r_act, lw=1, label="r_act", c="green")
    axes[2,0].legend()

    axes[2,1].set_title("cumulative episode reward")
    axes[2,1].plot(x, r.cumr, c="red")

    ty = 0.9*np.max(r.eps_rew)
    for tx, epsr in zip(np.arange(n_trials)*n_steps+(0.29*n_steps), r.eps_rew): axes[2,1].text(tx, ty, int(epsr))

    # axes[3,0].set_title("object velocity")

    # timesteps for horizontal episode reset lines
    # we don't need one at 0 nor at the end (hence the +1 and [:-1])
    reset_lines = ((np.arange(n_trials)+1)*n_steps)[:-1]
    for ax in axes.flatten():
        for rl in reset_lines:
            ax.axvline(rl, lw=.7, ls="dashed", c="grey")
        ax.set_xlim(0, int(len(x)*1.05))
    
    fig.suptitle(plot_title)


def plot_curves(
    xy_list, x_axis, title, figsize
):
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for _, (x, y) in enumerate(xy_list):
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    
def plot_results(
    dirs, num_timesteps, x_axis, task_name, figsize=(14.5, 8.8)
):
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, figsize)

    sched_file = f"{dirs[0]}/scheduled_params.csv"
    if os.path.isfile(sched_file):
        csvr = CsvReader(sched_file)

        ax2 = plt.twinx()
        next(ax2._get_lines.prop_cycler)['color']

        tx = csvr.data["timesteps"]
        for k, v in csvr.data.items():
            if k in ["timesteps", "walltime"]: continue

            v = np.array(v)
            if len(v.shape)>=2 and v.shape[1]==2: v = np.abs(v[:,1]-v[:,0])
            v /= v[-1]

            ax2.plot(tx, v, label=k)
        ax2.legend(loc="lower right")
    plt.tight_layout()


def tactile_eval(trialdir, trial_name=None, plot_title=None, with_vis=False, training=True, nrollouts=5, checkpoint="best"):
    env, model, vis, params = make_eval_env_model(trialdir, with_vis=with_vis, checkpoint=checkpoint)
    if trial_name is None: trial_name = params["train"]["trial_name"]
    if plot_title is None: plot_title = params["train"]["plot_title"]

    # recover relevant parameters
    prefix = "__".join(trial_name.split("__")[2:])+"__" # cut off first two name components (date and env name)
    timesteps = int(params["train"]["timesteps"])
    trial_name = params["train"]["trial_name"]
    plot_title = f'{plot_title or "Force Control"} |{checkpoint.upper()}|\n{trial_name}'
    os.makedirs(f"{trialdir}/", exist_ok=True)

    # learning curve
    plot_results([trialdir], timesteps, X_TIMESTEPS, task_name=plot_title.replace("\n", " - Learning Curve\n"), figsize=(11.5, 6.8))
    plt.savefig(f"{trialdir}/{prefix}learning_curve.png")

    fc = ForcePI(env, verbose=with_vis)

    # comparison plot
    agent_oracle_comparison(
        env, model, fc, vis, 
        goals=np.round(np.linspace(*env.fgoal_range_max, num=20), 4),
        reset_cb=force_reset_cb, after_step_cb=force_after_step_cb,
        plot=True, trialdir=trialdir, 
        plot_title=plot_title.replace("\n", " - Baseline Comparison\n"))
    plt.savefig(f"{trialdir}/{prefix}baseline_comparison.png")
    
    # plot a few rollouts in more detail
    a_res, o_res = agent_oracle_comparison(
        env, model, fc, vis, plot=False,
        goals=np.round(np.linspace(*env.fgoal_range_max, num=nrollouts), 4),
        reset_cb=force_reset_cb, after_step_cb=force_after_step_cb
    )

    plot_rollouts(env, a_res, plot_title=plot_title.replace("\n", " - POLICY\n"))
    plt.savefig(f"{trialdir}/{prefix}rollouts_policy.png")

    plot_rollouts(env, o_res, plot_title=plot_title.replace("\n", " - BASELINE\n"))
    plt.savefig(f"{trialdir}/{prefix}rollouts_baseline.png")

    print("stiffness eval policy")
    stiffness_var_plot(env, model, vis, 5, 6, plot_title.replace("\n", " - POLICY\n"))
    plt.savefig(f"{trialdir}/{prefix}stiff_var_policy.png")

    print("stiffness eval baseline")
    stiffness_var_plot(env, fc, vis, 5, 6, plot_title.replace("\n", " - BASELINE\n"))
    plt.savefig(f"{trialdir}/{prefix}stiff_var_baseline.png")

    cumr_a = np.mean(a_res.eps_rew)
    cumr_o = np.mean(o_res.eps_rew)

    ### clean plots
    clean_lc(trialdir, params, mode=PLOTMODE.paper, prefix=f"{trialdir}/{prefix}")

    goals = np.linspace(*env.fgoal_range, 4)
    res = deterministic_eval(env, model, None, goals, reset_cb=force_reset_cb, after_step_cb=force_after_step_cb)

    f_act_plot(res, mode=PLOTMODE.paper, prefix=f"{trialdir}/{prefix}")

    plt.rcParams.update(plt.rcParamsDefault)

    print("tactile eval done!")
    
    return cumr_a, cumr_o

def stiffness_var_plot(env, model, vis, n_goals, n_trials, plot_title):
    rand_stiff = env.randomize_stiffness
    env.set_attr("randomize_stiffness", False)

    reses = []
    fmaxes = []
    kappas = np.linspace(0, 1, n_goals)
    for kappa in kappas:
        env.change_stiffness(kappa)
        fmaxes.append(env.fmax)
        reses.append(
            deterministic_eval(
                env, 
                model, 
                vis, 
                np.linspace(*env.fgoal_range, n_trials),
                reset_cb=force_reset_cb,
                after_step_cb=force_after_step_cb
            )
        )
    env.set_attr("randomize_stiffness", rand_stiff)

    fig, axes = plt.subplots(nrows=n_goals, ncols=2, figsize=(10, 0.5+n_goals*2.5)) 

    for i, (r, fs, fmax) in enumerate(zip(reses, kappas, fmaxes)):
        n_steps  = r.nsteps
        x = np.arange(r.nsteps*r.ntrials)

        axes[i,0].plot(x, r.force)
        axes[i,0].plot(x, r.goals)
        axes[i,0].set_ylabel(f"f_alpha {fs:.2f} | f_max {fmax:.2f}")
        axes[i,0].set_ylim(-0.02, fmax*1.13)
        axes[i,0].axhline(fmax, lw=.7, ls="dashed", c="red")

        ty = 1.03*fmax
        for tx, epsr in zip(np.arange(n_trials)*n_steps+(0.29*n_steps), r.eps_rew): 
            axes[i,0].text(tx, ty, int(epsr))

        axes[i,1].plot(x, r.actions)
        axes[i,1].set_ylim(-1.05, 0.2)

        axes[0,0].set_title("forces")
        axes[0,1].set_title("actions")

    reset_lines = ((np.arange(n_trials)+1)*n_steps)[:-1]
    for ax in axes.flatten():
        for rl in reset_lines:
            ax.axvline(rl, lw=.7, ls="dashed", c="grey")

    fig.suptitle(plot_title)
    fig.tight_layout()


def plot_pos_rollouts(env, r, plot_title):
    n_trials = len(r.q)
    n_steps  = len(r.q[0])
    x = np.arange(r.q.shape[0])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14.5, 8.8))

    axes[0,0].set_title("joint position")
    q1,  = axes[0,0].plot(x, r.q[:,0], lw=1, label="qdes")
    q2,  = axes[0,0].plot(x, r.q[:,1], lw=1)
    qd1, = axes[0,0].plot(x, r.qdes[:,0], lw=1)
    qd2, = axes[0,0].plot(x, r.qdes[:,1], lw=1)
    qg,  = axes[0,0].plot(r.goals, c="grey", label="qgoal")
    axes[0,0].legend([(q1, q2), (qd1, qd2), (qg,)], ['q', 'qdes', "qgoal"],
               handler_map={tuple: HandlerTuple(ndivide=None)})

    axes[0,1].set_title("forces")
    axes[0,1].plot(x, r.force[:,0], lw=1)
    axes[0,1].plot(x, r.force[:,1], lw=1)

    axes[1,0].set_title("joint velocity")
    axes[1,0].plot(x, r.qdot[:,0], lw=1)
    axes[1,0].plot(x, r.qdot[:,1], lw=1)
    axes[1,0].set_ylim(-1.1*env.vmax, 1.1*env.vmax)
    axes[1,0].axhline(-env.vmax, lw=0.7, ls="dashed", c="grey")
    axes[1,0].axhline(env.vmax, lw=0.7, ls="dashed", c="grey")

    axes[1,1].set_title("joint acceleration")
    axes[1,1].plot(x, r.qacc[:,0], lw=1)
    axes[1,1].plot(x, r.qacc[:,1], lw=1)
    axes[1,1].set_ylim(-1.1*env.amax, 1.1*env.amax)
    axes[1,1].axhline(-env.amax, lw=0.7, ls="dashed", c="grey")
    axes[1,1].axhline(env.amax, lw=0.7, ls="dashed", c="grey")

    axes[2,0].set_title("partial rewards")
    axes[2,0].plot(x, r.r_pos,  lw=1, label="r_pos",  c="cyan")
    axes[2,0].plot(x, r.r_act, lw=1, label="r_act", c="orange")
    axes[2,0].legend()

    axes[2,1].set_title("cumulative episode reward")
    axes[2,1].plot(x, r.cumr, c="red")

    # axes[3,0].set_title("object velocity")

    # timesteps for horizontal episode reset lines
    # we don't need one at 0 nor at the end (hence the +1 and [:-1])
    reset_lines = ((np.arange(n_trials)+1)*n_steps)[:-1]
    for ax in axes.flatten():
        for rl in reset_lines:
            ax.axvline(rl, lw=.7, ls="dashed", c="grey")
        ax.set_xlim(0, int(len(x)*1.05))
    
    fig.suptitle(plot_title)
    plt.tight_layout()


def pos_eval(trialdir, trial_name=None, plot_title=None, with_vis=False, training=True, nrollouts=5, checkpoint="best"):
    env, model, vis, params = make_eval_env_model(trialdir, with_vis=with_vis, checkpoint=checkpoint)
    if trial_name is None: trial_name = params["train"]["trial_name"]
    if plot_title is None: plot_title = params["train"]["plot_title"]

    # recover relevant parameters
    prefix = "__".join(trial_name.split("__")[2:])+"__" # cut off first two name components (date and env name)
    timesteps = int(params["train"]["timesteps"])
    trial_name = params["train"]["trial_name"]
    plot_title = f'{plot_title or "Position Control"}\n{trial_name}'

    # learning curve
    plot_results([trialdir], timesteps, X_TIMESTEPS, task_name=plot_title.replace("\n", " - Learning Curve\n"), figsize=(8,4))
    plt.savefig(f"{trialdir}/{prefix}learning_curve.png")

    # baseline model
    pc = PosModel(env)

    # comparison plot
    agent_oracle_comparison(
        env, model, pc, vis, 
        goals=np.round(np.linspace(*env.qgoal_range, num=20), 4),
        reset_cb=pos_reset_cb, after_step_cb=pos_after_step_cb,
        plot=True, trialdir=trialdir, 
        plot_title=plot_title.replace("\n", " - Baseline Comparison\n"))
    plt.savefig(f"{trialdir}/{prefix}baseline_comparison.png")

     # plot a few rollouts in more detail
    a_res, o_res = agent_oracle_comparison(
        env, model, pc, vis, plot=False,
        goals=np.round(np.linspace(*env.qgoal_range, num=nrollouts), 4),
        reset_cb=pos_reset_cb, after_step_cb=pos_after_step_cb,
    )

    plot_pos_rollouts(env, a_res, plot_title=plot_title.replace("\n", " - POLICY\n"))
    plt.savefig(f"{trialdir}/{prefix}rollouts_policy.png")

    plot_pos_rollouts(env, o_res, plot_title=plot_title.replace("\n", " - BASELINE\n"))
    plt.savefig(f"{trialdir}/{prefix}rollouts_baseline.png")

    cumr_a = np.mean(safe_last_cumr(a_res))
    cumr_o = np.mean(safe_last_cumr(o_res))

    print("pos eval done!")

    return cumr_a, cumr_o