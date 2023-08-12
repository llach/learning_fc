import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.training import make_eval_env_model
from learning_fc.training.evaluation import stiffness_var_plot
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 3
N_TRIALS = 6

# trial = f"{model_path}/2023-08-11_17-38-07__gripper_tactile__ppo__pos_delta__obs_q-qdes-f-df-hadC-act__nenv-10__k-3"
trial = find_latest_model_in_path(model_path, filters=["ppo"])
print(trial)

env, model, vis, params = make_eval_env_model(trial, with_vis=0, checkpoint="best",
                                         env_override=dict(sample_fscale=False))

plot_title = f"{params['train']['plot_title']}\n{params['train']['trial_name']}"
stiffness_var_plot(env, model, vis, N_GOALS, N_TRIALS, plot_title)

plt.show()
