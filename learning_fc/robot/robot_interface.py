import os 
import time
import rospy
import pickle
import threading
import numpy as np

from datetime import datetime
from collections import deque

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ta11_readout.msg import ModelDebug

from learning_fc import model_path, datefmt
from learning_fc.utils import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.enums import ControlTask, ControlMode, Observation
from learning_fc.models import ForcePI

class RobotInterface:

    JOINT_NAMES = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

    def __init__(self, model, env, k=1, goal=0.0, fth=0.0075, freq=25, control_mode=None, datadir=None, verbose=False, with_indb=False):
        self.k = k
        self.env = env
        self.fth = fth
        self.goal = goal
        self.freq = freq
        self.model = model
        self.dt = 1/self.freq
        self.dq_max = env.dq_max
        self.verbose = verbose
        self.with_indb = with_indb

        # hacky way of supplying baseline with ROS data
        if isinstance(self.model, ForcePI): self.model.env = self

        self.task = ControlTask.Force if isinstance(env.unwrapped, GripperTactileEnv) else ControlTask.Position
        self.obs_config = env.obs_config
        self.control_mode = control_mode or env.control_mode
        self.obs_buf = deque(maxlen=self.k)

        self.active = False
        self.js_idx = [None, None]

        self.data_dir = datadir or f"{model_path}/data/"
        os.makedirs(self.data_dir, exist_ok=True)

        ### ROS init
        rospy.init_node("model_robot_interface")

        self.r = rospy.Rate(self.freq)
        self.lpub = rospy.Publisher("/finger_left_controller/command", Float64MultiArray, queue_size=1)
        self.rpub = rospy.Publisher("/finger_right_controller/command", Float64MultiArray, queue_size=1)
        self.dpub = rospy.Publisher("/model_debug", ModelDebug, queue_size=1)

        self._setup_subscribers()
        self.reset()

        self.initialized = True
        self.dpub_thread = threading.Thread(target=self._debug_pub)
        self.dpub_thread.start()

    def shutdown(self):
        print("RobotInterface.shutdown()")
        self.initialized = False
        self.dpub.unregister()
        self.dpub_thread.join()
        
    def _debug_pub(self):
        while not rospy.is_shutdown() and self.initialized:
            try:
                if np.any(self.force==None) or np.any(self.q==None): continue

                m = ModelDebug()
                if self.task == ControlTask.Force:
                    m.force = list(self.force.copy())
                elif self.task == ControlTask.Position:
                    m.force = self.q.copy()
                m.goal = float(self.goal)
                m.force_deltas = self._goal_delta()
                m.actions = self.last_a
                m.ftheta = self.fth
                self.dpub.publish(m)
                self.r.sleep()
            except Exception as e:
                print(e)
                break
        print("debug pub shutting down")

    def _js_cb(self, msg):
        # first callback -> store joint indices
        if None in self.js_idx:
            for i, jn in enumerate(self.JOINT_NAMES): self.js_idx[i] = msg.name.index(jn)

        for i, jidx in enumerate(self.js_idx):
            self.q[i] = msg.position[jidx]
            self.qdot[i] = msg.velocity[jidx]

    def force_cb(self, msg):
        self.force = np.array(msg.data)
        assert self.force.shape == (2,), "self.force.shape == (2,)"

        self.in_con  = self.force > self.fth
        self.had_con = self.in_con | self.had_con

    def _setup_subscribers(self): 
        print("setting up subscribers ...")
        self.js_sub = rospy.Subscriber("/joint_states", JointState, self._js_cb)
        if self.task == ControlTask.Force:
            self.force_sub = rospy.Subscriber("/ta11", Float64MultiArray, self.force_cb)
    
    def _goal_delta(self): 
        if self.task == ControlTask.Force:
            return self.goal - self.force
        elif self.task == ControlTask.Position:
            return self.goal - self.q
        else:
            assert False, f"unknown ControlTask {self.task}"

    def _enum2obs_raw(self, on):
        if on == Observation.Pos:           return self.q.copy()
        if on == Observation.Des:           return self.qdes.copy()
        if on == Observation.Vel:           return self.qdot.copy()
        if on == Observation.Force:         return self.force.copy()
        if on == Observation.Action:        return self.last_a.copy()
        if on == Observation.PosDelta:      return self._goal_delta()
        if on == Observation.ForceDelta:    return self._goal_delta()
        if on == Observation.InCon:         return self.in_con.copy()
        if on == Observation.HadCon:        return self.had_con.copy()

    def _enum2obs(self, on):
        if on == Observation.Pos:           return safe_rescale(self.q, [0.0, 0.045])
        if on == Observation.Des:           return safe_rescale(self.qdes, [0.0, 0.045])
        if on == Observation.Vel:           return safe_rescale(self.qdot, [-self.env.vmax, self.env.vmax])
        if on == Observation.Force:         return self.force#safe_rescale(self.force, [0, self.env.max_fmax])
        if on == Observation.Action:        return self.last_a.copy()
        if on == Observation.PosDelta:      return safe_rescale(self._goal_delta(), [-0.045, 0.045])
        if on == Observation.ForceDelta:    return self._goal_delta()#safe_rescale(self._goal_delta(), [-self.goal, self.goal])
        if on == Observation.InCon:         return self.in_con.copy()
        if on == Observation.HadCon:        return self.had_con.copy()

        assert False, f"unknown Observation {on}"

    def _get_obs(self):
        obs = []
        for on in self.obs_config: obs.append(self._enum2obs(on))
        return np.concatenate(obs).astype(np.float32)
    
    def get_goal(self): return self.goal
    
    def actuate(self, action): 
        left = Float64MultiArray(data=[action[0]])
        right = Float64MultiArray(data=[action[1]])

        if np.abs(left.data[0]-self.q[0])>self.env.dq_min: 
            if self.verbose: print(f"publishing left {left.data} | {self.q[0]}")
            self.lpub.publish(left)
        if np.abs(right.data[0]-self.q[1])>self.env.dq_min: 
            if self.verbose: print(f"publishing right {right.data} | {self.q[1]}")
            self.rpub.publish(right)

    def set_goal(self, g):
        if self.task == ControlTask.Position:
            self.goal = np.clip(g, 0.0, 0.045)
        elif self.task == ControlTask.Force:
            self.goal = np.clip(g, 0, 1.0)
        else:
            assert False, f"unknown ControlTask {self.task}"

    def reset(self): 
        """ initialize observation arrays
        """
        self.q     = np.array([None,None])
        self.qdot  = np.array([0,0])
        self.act   = np.array([0,0])

        self.cumr = 0
        self.force   = np.array([None,None])
        self.in_con  = np.array([0,0])
        self.had_con = np.array([0,0])
        self.last_a = np.array([0,0])

        self.set_goal(0)

        # reset observation buffer
        self.obs_buf = deque(maxlen=self.k)

        if self.verbose: print("waiting for joint states ...")
        while not rospy.is_shutdown() and np.any(self.q==None): self.r.sleep()

        if self.task == ControlTask.Force:
            if self.verbose: print("waiting for forces ...")
            while not rospy.is_shutdown() and np.any(self.force==None): self.r.sleep()

        if self.control_mode == ControlMode.Position:
            self.last_a = safe_rescale(self.q, [0.0, 0.045])
        elif self.control_mode == ControlMode.PositionDelta:
            self.last_a  = np.array([0,0])

        self.qdes = self.q.copy()

        if self.verbose: print("reset done!")

    def step(self): 
        obs = self._get_obs()
        obs_ = {on: self._enum2obs_raw(on) for on in self.obs_config} # for history
        
        self.obs_buf.append(obs)
        while len(self.obs_buf) < self.obs_buf.maxlen: 
            if self.verbose: print(f"obs {len(self.obs_buf)}..{self.obs_buf.maxlen}")
            self.obs_buf.append(obs)

        if isinstance(self.model, ForcePI):
            raw_action, _ = self.model.predict(self.q, self.force, self.goal)
        else:
            obs = np.asarray(self.obs_buf, dtype=np.float32).flatten()
            raw_action, _ = self.model.predict(obs, deterministic=True)

        if self.with_indb:
            ab = self.env.action_bias(self.had_con, self.force, self.get_goal())
            raw_action = ab * raw_action

        if self.control_mode == ControlMode.Position:
            self.qdes = safe_rescale(raw_action, [-1, 1], [0.0, 0.045])
        elif self.control_mode == ControlMode.PositionDelta:
            ain = safe_rescale(raw_action, [-1, 1], [-self.env.dq_max, self.env.dq_max])
            self.qdes = np.clip(self.q+ain, 0, 0.045)
        
        self.actuate(self.qdes)
        self.last_a = raw_action.copy()

        self.rew = self.env.rforce(self.get_goal(), self.force.copy())
        self.cumr += self.rew

        # update history
        for k, v in obs_.items():
            self.hist["obs"][k].append(v)
        self.hist["qdes"].append(self.qdes)
        self.hist["net_out"].append(raw_action)
        self.hist["goal"].append(self.goal)
        self.hist["timestamps"].append(datetime.utcnow())
        self.hist["r"].append(self.rew)
        self.hist["force"].append(self.force)
        self.hist["cumr"].append(self.cumr)

    def stop(self): 
        if self.active:
            if self.verbose: print("killing thread ...")
            self.active = False
            self.exec_thread.join()
            if self.verbose: print("done")

    def save_hist(self, name=None):
        now = f"{datetime.utcnow().strftime(datefmt)}"
        if name is None: name = now
        file_path = f"{self.data_dir}/{name}.pkl"

        if self.verbose: print(f"storing {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(self.hist, f)

    def run(self):
        if self.verbose: print("running model ...")
        self.hist = dict(
            obs={on: [] for on in self.obs_config},
            qdes=[],
            net_out=[],
            goal=[],
            timestamps=[],
            force=[],
            r=[],
            cumr=[],
            dq_max=self.env.dq_max,
        )

        def _step_loop():
            self.active=True
            while not rospy.is_shutdown() and self.active:
                self.step()
                self.r.sleep()
            if self.verbose: print("_step_loop() finished")
        
        self.exec_thread = threading.Thread(target=_step_loop)
        self.exec_thread.start()


if __name__ == "__main__":
    from learning_fc.training import make_eval_env_model
    from learning_fc.utils import find_latest_model_in_path

    # trial = f"{model_path}/2023-09-14_10-53-25__gripper_tactile__ppo__k-3__lr-0.0006_M2_inb" 
    trial = f"{model_path}/2023-09-14_11-24-22__gripper_tactile__ppo__k-3__lr-0.0006_M2_noinb" 
    # trial = find_latest_model_in_path(model_path, filters=["ppo"])
    env, model, _, params = make_eval_env_model(trial, with_vis=False, checkpoint="best")
    k = 1 if "frame_stack" not in params["make_env"] else params["make_env"]["frame_stack"]

    from learning_fc.models import PosModel, StaticModel, ForcePI
    # model = PosModel(env)
    # model = StaticModel(safe_rescale(-0.003, [-env.dq_max, env.dq_max], [-1, 1]))

    # model = ForcePI(env, verbose=False)
    ri = RobotInterface(
        model, 
        env, 
        k=k, 
        fth=env.fth,
        goal=0.0, 
        freq=25, 
        verbose=False, 
        with_indb=True
    )

    time.sleep(1.0)
    ri.actuate([0.045, 0.045])

    while not rospy.is_shutdown():
        inp = input("q = kill; sa = save; st = stop; o = open; gXXX = goal; aXXX = actuate\n")
        if inp == "q":
            ri.stop()
            ri.shutdown()
            break

        elif inp == "st":
            ri.stop()

        elif inp=="o":
            ri.stop()
            ri.actuate([0.045, 0.045])
            if isinstance(model, ForcePI): model.reset()

        elif inp[0]=="g":
            goal = inp[1:]
            try:
                goal = float(goal)
                assert goal > 0, "goal > 0"
                ri.set_goal(goal)
                print(f"new goal: {goal}")
            except Exception as e:
                print(f"can't convert {goal} to a number:\n{e}")
                continue
            if not ri.active: ri.reset(); ri.run()

        elif inp[0]=="a":
            ri.stop()

            goal = inp[1:]
            try:
                goal = float(goal)
                print(f"actuate: {goal}")
                ri.actuate(2*[goal])
            except:
                print(f"can't convert {goal} to a number")
                continue

        elif inp[:2]=="sa":
            ri.stop()
            
            name = inp.split(" ")[1] if " " in inp else None
            ri.save_hist(name)