import inspect 

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from learning_fc.utils import get_constructor_params, safe_unwrap
from learning_fc.envs import GripperPosEnv, GripperTactileEnv
from learning_fc.live_vis import PosVis, TactileVis


envname2cls = dict(
    gripper_pos=GripperPosEnv,
    gripper_tactile=GripperTactileEnv,
)

default_env_kw = dict(
    gripper_pos={},
    gripper_tactile={},
)

env2vis = dict(
    gripper_pos=PosVis,
    gripper_tactile=TactileVis,
)

def make_env(env_name, logdir, env_kw={}, max_steps=250, nenv=1, frame_stack=1, with_vis=False, training=True):
    fn_params = locals() # get locals right away to avoid "env" or "vis" to be included 

    # get environment class
    assert env_name in envname2cls, f"unknown env {env_name}, available: {list(envname2cls.keys())}"
    ecls = envname2cls[env_name]

    # prepare environment arguments
    if with_vis: env_kw |= dict(render_mode="human")

    # instantiate environment and wrap
    env = ecls(**default_env_kw[env_name] | env_kw)
    env = TimeLimit(env, max_episode_steps=max_steps)
    if training: env = Monitor(env, logdir) # we only need this wrapper during training

    # (optional) create live vis
    vis = env2vis[env_name](env) if with_vis else None

    # get environment creation parameters to replicate the environment config while testing
    init_params = get_constructor_params(ecls, safe_unwrap(env))
    params = fn_params | dict(init_params=init_params)

    return env, vis, params