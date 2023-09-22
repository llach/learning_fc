import time
import numpy as  np

from learning_fc import model_path
from learning_fc.training import make_eval_env_model
from learning_fc.live_vis import VideoVis
from learning_fc.envs.gripper_env import VIDEO_CAMERA_CONFIG

# trial, indb = f"{model_path}/2023-09-14_10-53-25__gripper_tactile__ppo__k-3__lr-0.0006_M2_inb", True
trial, indb = f"{model_path}/2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb", False
# trial, indb = f"{model_path}/2023-09-15_08-22-36__gripper_tactile__ppo__k-3__lr-0.0006_M2_nor", False

env, model, _, _ = make_eval_env_model(trial, with_vis=0, checkpoint="best", env_override=dict(
    render_mode="human",
    default_camera_config=VIDEO_CAMERA_CONFIG,
    wo_range=[0.025, 0.027],
    oy_range=[-0.012, -0.012]
))
env.set_attr("with_bias", indb)
vis = VideoVis(env)

obs, _ = env.reset()
time.sleep(30)

for _ in range(5):
    vis.reset(hard=True)

    # IB kappa low
    # obs, _ = env.reset()
    # env.change_stiffness(0.9)

    # NOIB kappa low
    obs, _ = env.reset()
    env.change_stiffness(0.4)

    # NO-RAND good
    # env.biasprm = [0, -100, -9]
    # obs, _ = env.reset()
    # env.change_stiffness(0.5)

    # NO-RAND bad 
    # env.biasprm = [0, -100, -13]
    # obs, _ = env.reset()
    # env.change_stiffness(0.9)

    goal = np.random.uniform(0.15, min(0.8, env.fgoal_range[1]))

    for j in range(350):
        if j < 100:
            vis.win.setBackground("w")
            ain = np.array([0,0])
            dain = ain
            env.set_goal(0.0001)
        elif 100 <= j <= 250:
            env.set_goal(goal)
            ain, _ = model.predict(obs, deterministic=True)
            dain = ain
        else:
            env.set_goal(0.0001)
            ain = np.array([1,1])               
            dain = np.array([0,0])
        if 97 < j <= 100: env.set_goal(goal)

        obs, r, _, _, _ = env.step(ain)
        if vis: vis.update_plot(action=dain)
        time.sleep(env.dt/2)
        # print(env.mujoco_renderer.viewer.cam)
env.close()