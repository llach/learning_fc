import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model, make_env
from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb, plot_rollouts

N_GOALS  = 2
with_vis = 1
trial = "2023-07-02_20-45-11__gripper_tactile__ppo__pos_delta__obs_q-f-df__nenv-1__k-3"

env, vis, _ = make_env(
    env_name="gripper_tactile", 
    training=False, 
    with_vis=with_vis, 
    max_steps=250,
    env_kw=dict(control_mode=ControlMode.PositionDelta, obs_config=ObsConfig.Q_F_DF, ro_scale=0, oy_init=0, wo_range=[0.02, 0.03])
)

_, model, _, _ = make_eval_env_model(f"{model_path}/{trial}", with_vis=with_vis)
# model = ForcePI(env)

res = deterministic_eval(env, model, vis, np.linspace(*env.fgoal_range, N_GOALS), reset_cb=force_reset_cb, after_step_cb=force_after_step_cb)
print(np.array(res["cumr"])[:,-1])

plot_rollouts(env, res, trial)
plt.show()