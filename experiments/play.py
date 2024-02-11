import time
import numpy as np
import matplotlib.pyplot as plt

from learning_fc import model_path
from learning_fc.enums import ControlMode
from learning_fc.models import ForcePI
from learning_fc.training import make_eval_env_model
from learning_fc.utils import find_latest_model_in_path

N_GOALS  = 10
with_vis = 0
# trial = f"{model_path}/30_base"
# trial = find_latest_model_in_path(model_path, filters=["ppo"])
# trial = f"{model_path}/2023-09-13_18-47-36__gripper_tactile__ppo__k-3__lr-0.0006"
# trial = f"{model_path}/2023-09-14_10-53-25__gripper_tactile__ppo__k-3__lr-0.0006_M2_inb"
trial = f"{model_path}/2024-02-11_16-40-14__gripper_tactile__ppo__k-3__lr-0.0006"


env, model, vis, _ = make_eval_env_model(
    trial, 
    with_vis=with_vis, 
    checkpoint="best", 
    env_override = dict(
        control_mode=ControlMode.PositionDelta,
        oy_range=[0.04,0.04],
        # wo_range=2*[0.035],
        # model_path=learning_fc.__path__[0]+"/assets/pal_force.xml",
        # f_scale=3.0,
        # sample_fscale=True,
        # fgoal_range=[0.05, 0.6],
        # sample_biasprm=True
        # noise_f=0.002,
        # ftheta=0.008,
    )
)

env.set_attr("noise_f", 0.002)
env.set_attr("fth", 0.02)
env.set_attr("with_bias", True)

print(env.ro_scale, env.ov_max)

# model = ForcePI(env)
kappas = np.linspace(0, 1, N_GOALS)
# kappas = [0.4]
cumrews = np.zeros((N_GOALS,))
for i, kappa in enumerate(kappas):
    obs, _ = env.reset()

    # env.change_stiffness(0)
    # env.change_dfdq(kappa)
    env.set_goal(np.random.uniform(*env.fgoal_range))
    # env.set_goal(0.3)

    if isinstance(model, ForcePI): model.reset()
    if vis: vis.reset()

    oy = env.obj_pos[1]
    oys = []
    hcs = []
    
    for j in range(300):
        ain, _ = model.predict(obs, deterministic=True)
        # if env.had_contact[0] and env.had_contact[1]: ain = [-0.1,-0.1]
        # else: ain = [-1,-1]
        
        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=ain, reward=r)

        oys.append(oy-env.obj_pos[1])
        hcs.append(env.had_contact)
        cumrews[i] += r

    plt.plot(list(range(300)), oys)
    ax2 = plt.twinx()
    ax2.plot(list(range(300)), hcs)
    plt.show()
    print(cumrews[i])

    time.sleep(0.01)
print(f"{np.mean(cumrews):.0f}±{np.std(cumrews):.1f}")
env.close()

# plot_rollouts(env, res, trial)
# plt.show()