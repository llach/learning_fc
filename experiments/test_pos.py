import numpy as np
np.set_printoptions(suppress=True, precision=5)

from learning_fc import safe_rescale
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import PosModel
from learning_fc.training import make_env

with_vis = 1
steps  = 100
trials = 5

env, vis, _ = make_env(
    env_name="gripper_pos", 
    training=False, 
    with_vis=with_vis, 
    env_kw=dict(
        dq_max=0.002,
        obs_config=ObsConfig.Q_DQ, 
        control_mode=ControlMode.PositionDelta, 
    )
)
model = PosModel(env)

for i in range(trials):
    obs, _ = env.reset()
    if vis: vis.reset()

    cumrew = 0
    for j in range(steps):
        action = np.array([-1,-1])
        action, _ = model.predict(obs)

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        print(env.q, env.qdot, env.qdes)

        cumrew += r
env.close()

print(cumrew)