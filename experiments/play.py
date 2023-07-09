import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb, plot_rollouts
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 5
with_vis = 1
trial = f"{model_path}/2023-07-06_17-17-37__gripper_tactile__ppo__pos_delta__obs_q-qdot-f-df-inC-hadC__nenv-6__k-3"
# trial = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, _ = make_eval_env_model(trial, with_vis=with_vis)
# model = ForcePI(env)

res = deterministic_eval(env, model, vis, np.linspace(*env.fgoal_range, N_GOALS), reset_cb=force_reset_cb, after_step_cb=force_after_step_cb)
print(np.array(res["cumr"])[:,-1])

plot_rollouts(env, res, trial)
plt.show()