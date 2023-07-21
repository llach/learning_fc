import inspect 

from gymnasium.wrappers import TimeLimit, FrameStack, FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from learning_fc.utils import get_constructor_params, safe_unwrap
from learning_fc.envs import GripperPosEnv, GripperTactileEnv, ControlMode
from learning_fc.live_vis import PosVis, TactileVis


envname2cls = dict(
    gripper_pos=GripperPosEnv,
    gripper_tactile=GripperTactileEnv,
)

default_env_kw = dict(
    gripper_pos=dict(control_mode=ControlMode.PositionDelta),
    gripper_tactile=dict(control_mode=ControlMode.PositionDelta),
)

env2vis = dict(
    gripper_pos=PosVis,
    gripper_tactile=TactileVis,
)

def make_env(env_name, logdir=None, env_kw={}, max_steps=250, nenv=1, frame_stack=1, with_vis=False, training=True):
    # sanity checks
    assert logdir is not None or not training, "specify logdir for training"

    fn_params = locals() # get locals right away to avoid "env" or "vis" to be included 

    # get environment class
    assert env_name in envname2cls, f"unknown env {env_name}, available: {list(envname2cls.keys())}"
    ecls = envname2cls[env_name]

    # prepare environment arguments
    if with_vis: env_kw = {**env_kw, **dict(render_mode="human")}

    def _make_env():
        # instantiate environment and wrap
        env = ecls(**{**default_env_kw[env_name], **env_kw})
        env = TimeLimit(env, max_episode_steps=max_steps)
        if frame_stack > 1:
            env = FrameStack(env=env, num_stack=frame_stack, lz4_compress=False)
            env = FlattenObservation(env=env)
        return env
    
    env = SubprocVecEnv([lambda: _make_env() for _ in range(nenv)]) if nenv>1 and training else _make_env()
    
    if training: # we only need this wrapper during training
        mon_cls = Monitor if nenv == 1 else VecMonitor
        env = mon_cls(env, logdir) 

    # (optional) create live vis
    vis = env2vis[env_name](env) if with_vis else None

    # get environment creation parameters to replicate the environment config while testing
    init_params = get_constructor_params(ecls, safe_unwrap(env))
    params = {**fn_params, **dict(init_params=init_params)}

    return env, vis, params