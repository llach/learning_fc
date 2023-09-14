import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import stiffness_var_plot
from learning_fc.utils import find_latest_model_in_path
from learning_fc.models import ForcePI

N_GOALS  = 2
N_TRIALS = 6

# trial = f"{model_path}/2023-08-31_11-35-02__gripper_tactile__ppo__k-3__lr-0.0006"
trial = find_latest_model_in_path(model_path, filters=["ppo"])
print(trial)

env, model, vis, params = make_eval_env_model(trial, with_vis=0, checkpoint="best")
# model = ForcePI(env)

plot_title = f"{params['train']['plot_title']}\n{params['train']['trial_name']}"
stiffness_var_plot(env, model, vis, N_GOALS, N_TRIALS, plot_title)

plt.show()
