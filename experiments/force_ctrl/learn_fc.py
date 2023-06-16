from learning_fc.training import train

name = "obs_q_f_df"
plot_title = "OBS={q, force, deltaf}"

for alg in ["ppo", "td3"]:
    train(
        env_name="gripper_tactile", 
        model_name=alg,
        name=name,
        plot_title=plot_title,
    )