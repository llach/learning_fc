import numpy as np
np.set_printoptions(suppress=True, precision=5)

from learning_fc.envs import ControlMode
from learning_fc.models import PosModel, ForcePI
from learning_fc.training import make_env

with_vis = 1
steps  = 200
trials = 5

env, vis, _ = make_env(
    env_name="gripper_tactile", 
    training=False, 
    with_vis=with_vis, 
    env_kw=dict(control_mode=ControlMode.PositionDelta)
)
model = ForcePI(env)

for i in range(trials):
    obs, _ = env.reset()
    if vis: vis.reset()

    for j in range(steps):
        # action = [1,1]
        action, _ = model.predict(obs)

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)
env.close()