from learning_fc.training import train

name = "obs_f_df"
plot_title = "OBS={force, deltaf}"

for alg in ["ppo", "td3"]:
    train(
        env_name="gripper_tactile", 
        model_name=alg,
        name=name,
        plot_title=plot_title,
        # train_kw=dict(timesteps=2e3) # quick training for debugging
    )