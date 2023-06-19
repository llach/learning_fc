import numpy as np
np.set_printoptions(suppress=True, precision=5)

from learning_fc import safe_rescale
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import PosModel, ForcePI
from learning_fc.training import make_env

with_vis = 1
steps  = 200
trials = 2

env, vis, _ = make_env(
    # env_name="gripper_pos", 
    env_name="gripper_tactile", 
    training=False, 
    with_vis=with_vis, 
    env_kw=dict(control_mode=ControlMode.Position, obs_config=ObsConfig.Q_DQ)#, obj_pos_range=[-0.03, -0.03])
)
# model = PosModel(control_mode=env.control_mode)
model = ForcePI(env)


n_a = 10
freqs = np.fft.fftfreq(n_a, env.dt)
for i in range(trials):
    obs, _ = env.reset()
    model.reset()
    if vis: vis.reset()

    last_a = np.zeros((n_a,2))
    for j in range(steps):
        # fixed position action, requires ControlMode.Position
        # action = safe_rescale([0.014, 0.014], [0.0, 0.045])

        # trying (oscillating) min/max actions
        # action = np.array([-1,-1])
        # if j%2==1: action *= -1

        # model action
        action, _ = model.predict(obs)

        # FFT
        last_a[j%n_a,:] = action

        print(j, freqs[:6])
        print(j, np.abs(np.fft.fft(last_a[:,0]))[:6])
        print()

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)
env.close()