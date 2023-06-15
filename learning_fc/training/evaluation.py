import json
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import plot_results, X_TIMESTEPS

from learning_fc.models import ForcePI
from learning_fc.training import make_env, make_model

def rollout_model(env, model, vis, n_rollouts, reset_cb=None, before_step_cb=None, after_step_cb=None):
    results = dict(
        q=[[] for _ in range(n_rollouts)],
        qdot=[[] for _ in range(n_rollouts)],
        qacc=[[] for _ in range(n_rollouts)],
        force=[[] for _ in range(n_rollouts)],
        in_contact=[[] for _ in range(n_rollouts)],
        r=[[] for _ in range(n_rollouts)],
        cumr=[[] for _ in range(n_rollouts)],
    )
    for i in range(n_rollouts):
        obs, _ = env.reset()

        if vis: vis.reset()
        if reset_cb: results = reset_cb(env, model, i, results) or results

        done = False
        while not done:
            if before_step_cb: results = before_step_cb(env, model, i, results) or results

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            results["q"][i].append(env.q)
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
        plt.savefig(f"{trialdir}/baseline_comparison.png")

    return agent_results, oracle_results


def tactile_eval(trialdir, with_vis=False):
    env, model, vis, params = make_eval_env_model(trialdir, with_vis=with_vis)

    # recover relevant parameters
    timesteps = int(params["train"]["timesteps"])
    trial_name = params["train"]["trial_name"]
    plot_title = f'{params["train"]["plot_title"] or "Force Control"}\n{trial_name}'

    # learning curve
    plot_results([trialdir], timesteps, X_TIMESTEPS, task_name=plot_title.replace("\n", " - Learning Curve\n"), figsize=(8,4))
    plt.savefig(f"{trialdir}/learning_curve.png")
    plt.clf()

    def force_reset_cb(env, model, i, results, **kw): 
        if isinstance(model, ForcePI): model.reset()

        if "deltaf" not in results: results |= dict(deltaf=[])
        results["deltaf"].append([])

        return results

    def force_after_step_cb(env, model, i, results, goal=None, **kw):
        results["deltaf"][i].append(env.force_deltas)
        return results

    fc = ForcePI(env, Kp=1.5, Ki=3.1, k=160, verbose=with_vis)
    ftargets = np.round(np.linspace(*env.fgoal_range, num=20), 4)

    agent_res, oracle_res = agent_oracle_comparison(
        env, model, fc, vis, ftargets, 
        reset_cb=force_reset_cb, after_step_cb=force_after_step_cb,
        plot=True, trialdir=trialdir, 
        plot_title=plot_title.replace("\n", " - Baseline Comparison\n"))
