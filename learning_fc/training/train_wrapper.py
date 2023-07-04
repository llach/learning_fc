import os
import json

from sys import platform
from datetime import datetime

from learning_fc import model_path, datefmt, get_constructor_params
from learning_fc.training import make_env, make_model, tactile_eval, envname2cls

model_defaults = dict(
    ppo=dict(timesteps=5e5),
    td3=dict(timesteps=1e5),
    ddpg=dict(timesteps=1e5),
    sac=dict(timesteps=1e5),
)

env_eval_fn = dict(
    gripper_tactile=tactile_eval
)

def generate_trial_name_and_plot_title(env_name, env_kw, model_name, nenv, frame_stack):
    # get date and env params
    datestr = datetime.utcnow().strftime(datefmt)
    epar = get_constructor_params(envname2cls[env_name]) | env_kw

    # build name and plot title lists
    _name  = [
        datestr, 
        env_name, 
        model_name, 
        epar["control_mode"], 
        f"obs_{'-'.join(epar['obs_config'])}",
        f"nenv-{nenv}",
        f"k-{frame_stack}"
    ]
    _title = [
        model_name.upper(),
        f"{str(epar['control_mode']).replace('.', ': ')}", 
        f"OBS={'{'}{', '.join(epar['obs_config'])}{'}'}x{frame_stack}"
    ]

    return "__".join(_name), " | ".join(_title)

def train(env_name="gripper_tactile", model_name="ppo", nenv=1, frame_stack=1, env_kw={}, model_kw={}, train_kw={}, logdir=model_path):
    # build training parameters dict, store locals
    tkw = model_defaults[model_name] | train_kw

    # build trial name and dir
    trial_name, plot_title = generate_trial_name_and_plot_title(env_name=env_name, model_name=model_name, nenv=nenv, frame_stack=frame_stack, env_kw=env_kw)
    trialdir = f"{logdir}/{trial_name}/"

    print( "##########################")
    print(f"######## Running trial {trial_name}")
    print( "##########################")

    # easy ref to total steps as integer
    timesteps = int(tkw["timesteps"])

    # store all parameters
    fn_params = locals()

    # create log and trial dirs
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(trialdir, exist_ok=True)

    # environment setup
    env, _, eparams = make_env(env_name=env_name, logdir=trialdir, env_kw=env_kw, with_vis=False, training=True, nenv=nenv, frame_stack=frame_stack)

    # model setup
    model, callbacks, mparams = make_model(env=env, model_name=model_name, logdir=trialdir, model_kw=model_kw, timesteps=timesteps, save_periodic=timesteps/20)
    
    # store parameters
    with open(f"{trialdir}/parameters.json", "w") as f:
        f.write(json.dumps(dict(
            make_env=eparams,
            make_model=mparams,
            train=fn_params
        ), indent=2, sort_keys=True))

    # train the agent
    try: model.learn(total_timesteps=timesteps, callback=callbacks)
    except (KeyboardInterrupt) as e: 
        print(f"\ngot exception while training:\n{e}\nattempting evaluation regardless")

    # create evaluation plots
    if env_name in env_eval_fn: 
        agent_rew, base_rew = env_eval_fn[env_name](trialdir, trial_name=trial_name, plot_title=plot_title, with_vis=False)

        if platform == "darwin": # macOS gets notifications
            import pync
            pync.notify(f"BASE {base_rew:.1f} | RL   {agent_rew:.1f}", title="RL Training done!", activate="com.microsoft.VSCode")