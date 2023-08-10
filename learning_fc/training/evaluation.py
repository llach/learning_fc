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

TACTILE_ENV_MEMBERS = [
    "force_deltas", 
    "r_force",
    "r_obj_pos",
    "r_obj_prox",
    "r_act",
    "obj_v",
    "obj_pos"
]

POSITION_ENV_MEMBERS = [
    "r_pos",
    "r_act"
]

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
        while not done:
            if before_step_cb: results = before_step_cb(env, model, i, results) or results

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            results["q"][i].append(env.q)
            results["qdes"][i].append(env.qdes)
            results["qdot"][i].append(env.qdot)
            results["qacc"][i].append(env.qacc)
            results["force"][i].append(env.force)
            results["in_contact"][i].append(env.in_contact)
            results["r"][i].append(reward)
            results["cumr"][i].append(sum(results["r"][i]))

            if vis: vis.update_plot(action=action, reward=reward)
            if after_step_cb: results = after_step_cb(env, model, i, results) or results
        results["eps_rew"].append(results["cumr"][i][-1])
    return results


def deterministic_eval(env, model, vis, goals, reset_cb=None, before_step_cb=None, after_step_cb=None):
    n_rollouts = len(goals)
    def set_env_goal(env, model, i, results, **kw):
        env.set_goal(goals[i])
        if isinstance(safe_unwrap(env), GripperTactileEnv):
            if env.fgoal > env.f_scale*env.fmax: # if the maximum force is smaller than the goal, raise it
                env.set_attr("f_scale", (env.fgoal*1.10)/env.fmax)

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

    cumr_a = np.array([cr[-1] for cr in agent_results["cumr"]])
    cumr_o = np.array([cr[-1] for cr in oracle_results["cumr"]])

    print(f"RL   {np.mean(cumr_a):.0f}±{np.std(cumr_a):.1f}")
    print(f"BASE {np.mean(cumr_o):.0f}±{np.std(cumr_o):.1f}")

    if plot:
        plt.clf()
        plt.figure(figsize=(6.5,7))

        plt.scatter(goals, cumr_a, label=f"π    {np.mean(cumr_a):.0f}±{np.std(cumr_a):.1f}")
        plt.scatter(goals, cumr_o, label=f"base {np.mean(cumr_o):.0f}±{np.std(cumr_o):.1f}")

        plt.title(plot_title)
        plt.xlabel("target")
        plt.ylabel("cumulative episode reward") 

        plt.legend()
        plt.tight_layout()

    return agent_results, oracle_results


def plot_rollouts(env, res, plot_title):
    n_trials = len(res["q"])
    n_steps  = len(res["q"][0])

    q = np.concatenate(res["q"])
    qdes = np.concatenate(res["qdes"])
    qdot = np.concatenate(res["qdot"])
    qacc = np.concatenate(res["qacc"])
    force = np.concatenate(res["force"])

    r_force = np.concatenate(res["r_force"]).reshape((-1,))
    r_obj_pos = np.concatenate(res["r_obj_pos"]).reshape((-1,))
    r_obj_prox = np.concatenate(res["r_obj_prox"]).reshape((-1,))
    r_act = np.concatenate(res["r_act"]).reshape((-1,))
    cumr = np.concatenate(res["cumr"]).reshape((-1,))
    goals = np.repeat(np.array(res["goals"]).reshape((-1,)), n_steps)
    eps_rew = np.array(res["eps_rew"]).reshape((-1,))

    x = np.arange(q.shape[0])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14.5, 8.8))

    axes[0,0].set_title("joint position")
    q1,  = axes[0,0].plot(x, q[:,0], lw=1, label="qdes")
    q2,  = axes[0,0].plot(x, q[:,1], lw=1)
    qd1, = axes[0,0].plot(x, qdes[:,0], lw=1)
    qd2, = axes[0,0].plot(x, qdes[:,1], lw=1)
    axes[0,0].legend([(q1, q2), (qd1, qd2)], ['q', 'qdes'],
               handler_map={tuple: HandlerTuple(ndivide=None)})

    axes[0,1].set_title("forces")
    axes[0,1].plot(x, force[:,0], lw=1)
    axes[0,1].plot(x, force[:,1], lw=1)
    axes[0,1].plot(goals, c="grey", label="fgoal")
    axes[0,1].legend()

    axes[1,0].set_title("joint velocity")
    axes[1,0].plot(x, qdot[:,0], lw=1)
    axes[1,0].plot(x, qdot[:,1], lw=1)
    axes[1,0].set_ylim(-1.1*env.vmax, 1.1*env.vmax)
    axes[1,0].axhline(-env.vmax, lw=0.7, ls="dashed", c="grey")
    axes[1,0].axhline(env.vmax, lw=0.7, ls="dashed", c="grey")

    axes[1,1].set_title("joint acceleration")
    axes[1,1].plot(x, qacc[:,0], lw=1)
    axes[1,1].plot(x, qacc[:,1], lw=1)
    axes[1,1].set_ylim(-1.1*env.amax, 1.1*env.amax)
    axes[1,1].axhline(-env.amax, lw=0.7, ls="dashed", c="grey")
    axes[1,1].axhline(env.amax, lw=0.7, ls="dashed", c="grey")

    axes[2,0].set_title("partial rewards")
    axes[2,0].plot(x, r_force, lw=1, label="r_force", c="cyan")
    axes[2,0].plot(x, r_obj_pos, lw=1, label="r_obj_pos", c="orange")
    axes[2,0].plot(x, r_obj_prox, lw=1, label="r_obj_prox", c="red")
    axes[2,0].plot(x, r_act, lw=1, label="r_act", c="green")
    axes[2,0].legend()

    axes[2,1].set_title("cumulative episode reward")
    axes[2,1].plot(x, cumr, c="red")

    ty = 0.9*np.max(eps_rew)
    for tx, epsr in zip(np.arange(n_trials)*n_steps+(0.29*n_steps), eps_rew): axes[2,1].text(tx, ty, int(epsr))

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
    prefix = f"{checkpoint}/"+"__".join(trial_name.split("__")[2:])+"__" # cut off first two name components (date and env name)
    timesteps = int(params["train"]["timesteps"])
    trial_name = params["train"]["trial_name"]
    plot_title = f'{plot_title or "Force Control"} |{checkpoint.upper()}|\n{trial_name}'
    os.makedirs(f"{trialdir}/{checkpoint}/", exist_ok=True)


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

    cumr_a = np.mean(safe_last_cumr(a_res))
    cumr_o = np.mean(safe_last_cumr(o_res))

    print("tactile eval done!")

    return cumr_a, cumr_o


def plot_pos_rollouts(env, res, plot_title):
    n_trials = len(res["q"])
    n_steps  = len(res["q"][0])

    q = np.concatenate(res["q"])
    qdes = np.concatenate(res["qdes"])
    qdot = np.concatenate(res["qdot"])
    qacc = np.concatenate(res["qacc"])
    force = np.concatenate(res["force"])

    r_pos  = np.concatenate(res[ "r_pos"]).reshape((-1,))
    r_act = np.concatenate(res["r_act"]).reshape((-1,))
    
    cumr = np.concatenate(res["cumr"]).reshape((-1,))
    goals = np.repeat(np.array(res["goals"]).reshape((-1,)), n_steps)

    x = np.arange(q.shape[0])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14.5, 8.8))

    axes[0,0].set_title("joint position")
    q1,  = axes[0,0].plot(x, q[:,0], lw=1, label="qdes")
    q2,  = axes[0,0].plot(x, q[:,1], lw=1)
    qd1, = axes[0,0].plot(x, qdes[:,0], lw=1)
    qd2, = axes[0,0].plot(x, qdes[:,1], lw=1)
    qg,  = axes[0,0].plot(goals, c="grey", label="qgoal")
    axes[0,0].legend([(q1, q2), (qd1, qd2), (qg,)], ['q', 'qdes', "qgoal"],
               handler_map={tuple: HandlerTuple(ndivide=None)})

    axes[0,1].set_title("forces")
    axes[0,1].plot(x, force[:,0], lw=1)
    axes[0,1].plot(x, force[:,1], lw=1)

    axes[1,0].set_title("joint velocity")
    axes[1,0].plot(x, qdot[:,0], lw=1)
    axes[1,0].plot(x, qdot[:,1], lw=1)
    axes[1,0].set_ylim(-1.1*env.vmax, 1.1*env.vmax)
    axes[1,0].axhline(-env.vmax, lw=0.7, ls="dashed", c="grey")
    axes[1,0].axhline(env.vmax, lw=0.7, ls="dashed", c="grey")

    axes[1,1].set_title("joint acceleration")
    axes[1,1].plot(x, qacc[:,0], lw=1)
    axes[1,1].plot(x, qacc[:,1], lw=1)
    axes[1,1].set_ylim(-1.1*env.amax, 1.1*env.amax)
    axes[1,1].axhline(-env.amax, lw=0.7, ls="dashed", c="grey")
    axes[1,1].axhline(env.amax, lw=0.7, ls="dashed", c="grey")

    axes[2,0].set_title("partial rewards")
    axes[2,0].plot(x, r_pos,  lw=1, label="r_pos",  c="cyan")
    axes[2,0].plot(x, r_act, lw=1, label="r_act", c="orange")
    axes[2,0].legend()

    axes[2,1].set_title("cumulative episode reward")
    axes[2,1].plot(x, cumr, c="red")

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