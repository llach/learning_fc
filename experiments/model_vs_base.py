import numpy as np
import matplotlib.pyplot as plt 

from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path, safe_unwrap
from learning_fc.models import ForcePI
from learning_fc.training.evaluation import make_eval_env_model, agent_oracle_comparison, force_reset_cb, force_after_step_cb

nrollouts=10
with_vis=0
training=False

trialdir = find_latest_model_in_path(model_path, filters=["ppo"])

env, model, vis, params = make_eval_env_model(trialdir, with_vis=with_vis)
trial_name = params["train"]["trial_name"]

uenv = safe_unwrap(env)
uenv.ro_scale = 800
uenv.rv_scale = 0

# recover relevant parameters
prefix = "__".join(trial_name.split("__")[2:])+"__" # cut off first two name components (date and env name)
timesteps = int(params["train"]["timesteps"])
trial_name = params["train"]["trial_name"]

# comparison plot
fc = ForcePI(env, verbose=with_vis)
agent_oracle_comparison(
    env, model, fc, vis, 
    goals=np.round(np.linspace(*env.fgoal_range, num=20), 4),
    reset_cb=force_reset_cb, after_step_cb=force_after_step_cb,
    plot=True, trialdir=trialdir, 
    plot_title="Baseline Comparison")
# plt.show()