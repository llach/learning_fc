import numpy as np
import mujoco as mj

from learning_fc.training.evaluation import deterministic_eval, force_reset_cb, force_after_step_cb, plot_rollouts
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI, StaticModel
from learning_fc.training import make_env

import matplotlib.pyplot as plt


N_GOALS  = 5
with_vis = 0
env, vis, _ = make_env(
    env_name="gripper_pos", 
    training=False, 
    with_vis=with_vis, 
    max_steps=250,
    env_kw=dict(control_mode=ControlMode.PositionDelta, obs_config=ObsConfig.Q_DQ)
)
model = ForcePI(env, verbose=0)
# model = StaticModel(0.0)

def after_cb(env, *args, **kwargs): 
    # print(env.force, env.qdes)
    return force_after_step_cb(env, *args, **kwargs)

goals = np.linspace(*env.fgoal_range, N_GOALS)
# goals = 5*[0.6]
print(f"goals={goals}")
res = deterministic_eval(env, model, vis, goals, reset_cb=force_reset_cb, after_step_cb=after_cb)

cumrs = np.array([cr[-1] for cr in res["cumr"]])
# print(cumrs)
# exit(0)
r_obj_pos = np.array(res["r_obj_pos"][-1])
# oy_t = np.array(res["oy_t"][-1])
objv = np.array(res["obj_v"][-1])[:,1]
x = range(len(r_obj_pos))

# plt.plot(r_obj_pos, label="r_obj_pos", color="orange")

# ax2 = plt.twinx()
# # ax2.plot(oy_t, label="oy_t")
# ax2.plot(objv, label="objv")

# plt.legend()
# plt.tight_layout()
# plt.show()

plot_rollouts(env, res, f"Baseline Rollouts")
plt.show()