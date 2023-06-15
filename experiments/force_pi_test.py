import numpy as np
np.set_printoptions(suppress=True, precision=6)

from learning_fc import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.models import ForcePI
from learning_fc.live_vis import PIVis

with_vis = 1
steps  = 250
trials = 5


env = GripperTactileEnv(
    obj_pos_range=[-0.008,0.008],
    fgoal_range=[0.2, 0.6],
    **{"render_mode": "human"} if with_vis else {}
)
fc = ForcePI(env.dt, env.fmax, env.fgoal, env.ftheta, Kp=1.5, Ki=3.1, k=160)
vis = PIVis(env) if with_vis else None

for i in range(trials):
    obs, _ = env.reset()
    # env.set_goal(0.6)  

    fc.reset(env.fgoal)
    if vis: vis.reset()

    cumr = 0
    for j in range(steps):
        # action = fc.get_q(env.q, env.forces)
        action, _ = fc.predict(obs)

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        cumr += r
    print(f"episode reward: {cumr}")
env.close()