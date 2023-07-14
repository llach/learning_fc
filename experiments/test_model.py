import numpy as np
np.set_printoptions(suppress=True, precision=5)

from learning_fc import safe_rescale
from learning_fc.enums import ControlMode, ObsConfig
from learning_fc.models import ForcePI
from learning_fc.training import make_env

with_vis = 0
steps  = 250
trials = 1

env, vis, _ = make_env(
    env_name="gripper_tactile", 
    training=False, 
    with_vis=with_vis, 
    env_kw=dict(
        control_mode=ControlMode.Position, 
        obs_config=ObsConfig.Q_DQ, 
        qinit_range=[0.045, 0.045], 
        oy_init=-0.015,
        wo_range=[0.025, 0.025],
        co_scale=0,
        rf_scale=0,
        ro_scale=50,
        rp_scale=1,
    )
)
model = ForcePI(env)

n_a = 10
freqs = np.fft.fftfreq(n_a, env.dt)
for i in range(trials):
    obs, _ = env.reset()
    model.reset()
    if vis: vis.reset()

    last_a = np.zeros((n_a,2))
    cumrew = 0

    obvs = []
    for j in range(steps):
        # fixed position action, requires ControlMode.Position
        # action = safe_rescale([0.01, 0.01], [0.0, 0.045])

        # trying (oscillating) min/max actions
        action = np.array([-1,-1])
        # if j%2==1: action *= -1

        # model action
        # action, _ = model.predict(obs)
        # print(model.error_integral)

        # FFT
        # last_a[j%n_a,:] = action

        # print(j, freqs[:6])
        # print(j, np.abs(np.fft.fft(last_a[:,0]))[:6])
        # print()

        obs, r, _, _, _ = env.step(action)
        if vis: vis.update_plot(action=action, reward=r)

        obvs.append(np.abs(env.obj_v[1]))

        cumrew += r
env.close()

print(cumrew, np.max(obvs))