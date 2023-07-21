import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path, safe_unwrap
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI, PosModel, StaticModel
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb, plot_rollouts
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 5
with_vis = 0
# trial = f"{model_path}/2023-07-19_14-28-19__centered__minimal_reward__nenv-6__k-1"
trial = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, _ = make_eval_env_model(trial, with_vis=with_vis, checkpoint="best")

def as_cb(env, model, i, results, goal=None, **kw):
    return force_after_step_cb(env, model, i, results, goal=None, **kw)

# model = ForcePI(env)
# model = StaticModel(-1)
# res = deterministic_eval(env, model, vis, np.linspace(*env.fgoal_range, N_GOALS), reset_cb=force_reset_cb, after_step_cb=as_cb)
# print(np.array(res["cumr"])[:,-1])


for i in range(N_GOALS):
    obs, _ = env.reset()
    if isinstance(model, ForcePI): model.reset()

    if vis: vis.reset()

    cumrew = 0
    for j in range(200):
        action, _ = model.predict(obs)
        # action    = np.array([-1,-1])

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        cumrew += r
    print(cumrew)
env.close()


# plot_rollouts(env, res, trial)
# plt.show()