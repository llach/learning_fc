import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple
from stable_baselines3.common.results_plotter import plot_results, X_TIMESTEPS

from learning_fc.models import ForcePI
from learning_fc.training import make_env, make_model

TACTILE_ENV_MEMBERS = [
    "force_deltas", 
    "r_force",
    "r_obj_pos",
    "objv",
    "objw",
    "oy_t"
]

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
        goals=[],
    )
    for i in range(n_rollouts):
        obs, _ = env.reset()

        if vis: vis.reset()
        if reset_cb: results = reset_cb(env, model, i, results) or results

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
    return results


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

def make_eval_env_model(trialdir, with_vis=False):
    # load parameters
    with open(f"{trialdir}/parameters.json", "r") as f:
        params = json.load(f)

    # modify env creation parameters for eval and create
    params["make_env"] |= dict(training=False, nenv=1, with_vis=with_vis)
    params["make_env"]["env_kw"] |= params["make_env"].pop("init_params")
    env, vis, _ = make_env(**params["make_env"])

    # same for the model
    params["make_model"] |= dict(training=False, weights="_best_model")
    params["make_model"]["model_kw"] |= params["make_model"].pop("init_params") | params["make_model"].pop("mkw")
    model, _, _ = make_model(env, **params["make_model"])

    return env, model, vis, params

def agent_oracle_comparison(env, agent, oracle, vis, goals, reset_cb=None, after_step_cb=None, plot=True, plot_title="", trialdir=None):

    print("baseline evaluation")
    oracle_results = deterministic_eval(env, oracle, vis, goals, reset_cb=reset_cb, after_step_cb=after_step_cb)

    print("policy evaluation")
    agent_results = deterministic_eval(env, agent, vis, goals, reset_cb=reset_cb, after_step_cb=after_step_cb)

    cumr_a = np.array(agent_results["cumr"])[:,-1]
    cumr_o = np.array(oracle_results["cumr"])[:,-1]

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
    x        = np.arange(n_steps*n_trials)

    q = np.array(res["q"]).reshape((-1,2))
    qdes = np.array(res["qdes"]).reshape((-1,2))
    qdot = np.array(res["qdot"]).reshape((-1,2))
    qacc = np.array(res["qacc"]).reshape((-1,2))
    force = np.array(res["force"]).reshape((-1,2))

    r_force = np.array(res["r_force"]).reshape((-1,))
    r_obj_pos = np.array(res["r_obj_pos"]).reshape((-1,))
    cumr = np.array(res["cumr"]).reshape((-1,))
    goals = np.repeat(np.array(res["goals"]).reshape((-1,)), n_steps)

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

def tactile_eval(trialdir, trial_name=None, plot_title=None, with_vis=False):
    env, model, vis, params = make_eval_env_model(trialdir, with_vis=with_vis)

    # recover relevant parameters
    prefix = "__".join(trial_name.split("__")[2:])+"__" # cut off first two name components (date and env name)
    timesteps = int(params["train"]["timesteps"])
    trial_name = params["train"]["trial_name"]
    plot_title = f'{plot_title or "Force Control"}\n{trial_name}'

    # learning curve
    plot_results([trialdir], timesteps, X_TIMESTEPS, task_name=plot_title.replace("\n", " - Learning Curve\n"), figsize=(8,4))
    plt.savefig(f"{trialdir}/{prefix}learning_curve.png")

    fc = ForcePI(env, verbose=with_vis)

    # comparison plot
    agent_oracle_comparison(
        env, model, fc, vis, 
        goals=np.round(np.linspace(*env.fgoal_range, num=20), 4),
        reset_cb=force_reset_cb, after_step_cb=force_after_step_cb,
        plot=True, trialdir=trialdir, 
        plot_title=plot_title.replace("\n", " - Baseline Comparison\n"))
    plt.savefig(f"{trialdir}/{prefix}baseline_comparison.png")
    
    # plot a few rollouts in more detail
    a_res, o_res = agent_oracle_comparison(
        env, model, fc, vis, plot=False,
        goals=np.round(np.linspace(*env.fgoal_range, num=5), 4),
        reset_cb=force_reset_cb, after_step_cb=force_after_step_cb
    )

    plot_rollouts(env, a_res, plot_title=plot_title.replace("\n", " - POLICY\n"))
    plt.savefig(f"{trialdir}/{prefix}rollouts_policy.png")

    plot_rollouts(env, o_res, plot_title=plot_title.replace("\n", " - BASELINE\n"))
    plt.savefig(f"{trialdir}/{prefix}rollouts_baseline.png")

    cumr_a = np.mean(np.array(a_res["cumr"])[:,-1])
    cumr_o = np.mean(np.array(o_res["cumr"])[:,-1])

    print("tactile eval done!")

    return cumr_a, cumr_o