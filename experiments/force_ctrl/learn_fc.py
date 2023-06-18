from learning_fc.envs import ControlMode
from learning_fc.training import train

name = "obs_q_f_df-abs_ctrl"
plot_title = "OBS={q, force, deltaf} & Absolute Position as Output"

for alg in ["ppo", "td3"]:
    train(
        env_name="gripper_tactile", 
        model_name=alg,
        name=name,
        plot_title=plot_title,
        env_kw=dict(control_mode=ControlMode.Position),
        # train_kw=dict(timesteps=2e3) # quick training for debugging
    )