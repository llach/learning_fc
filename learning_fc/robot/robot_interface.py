import rospy
import threading
import numpy as np

from collections import deque
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from learning_fc.utils import safe_rescale
from learning_fc.envs import GripperTactileEnv
from learning_fc.enums import ControlTask, ControlMode, Observation
from learning_fc.models import BaseModel

class RobotInterface:

    JOINT_NAMES = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

    def __init__(self, model, env, goal=0.0, fth=0.001, freq=50, n_action_avg=1):
        self.env = env
        self.fth = fth
        self.goal = goal
        self.freq = freq
        self.model = model
        self.actions = deque(maxlen=n_action_avg)

        self.task = ControlTask.Force if isinstance(env.unwrapped, GripperTactileEnv) else ControlTask.Position
        self.obs_config = env.obs_config
        self.control_mode = env.control_mode

        self.js_idx = [None, None]

        rospy.init_node("model_robot_interface")

        self.r = rospy.Rate(self.freq)
        self.qpub = rospy.Publisher("/gripper_position_controller/command", Float64MultiArray, queue_size=1)

        self._setup_subscribers()
        self.reset()

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
        if self.control_mode == ControlTask.Force:
            self.force_sub = rospy.Subscriber("/ta11", Float64MultiArray, self.force_cb)
    
    def _goal_delta(self): 
        if self.task == ControlTask.Force:
            return self.goal - self.force
        elif self.task == ControlTask.Position:
            return self.goal - self.q
        else:
            assert False, f"unknown ControlTask {self.task}"

    def _enum2obs(self, on):
        if on == Observation.Pos: return safe_rescale(self.q, [0.0, 0.045])
        if on == Observation.Vel: return safe_rescale(self.qdot, [-self.env.vmax, self.env.vmax])
        if on == Observation.Force: return safe_rescale(self.force, [0, self.env.fmax])
        if on == Observation.Action: return self.last_a
        if on == Observation.PosDelta: return safe_rescale(self._goal_delta(), [-0.045, 0.045])
        if on == Observation.ForceDelta: return safe_rescale(self._goal_delta(), [-self.goal, self.goal])
        if on == Observation.InCon: return self.in_con
        if on == Observation.HadCon: return self.had_con

        assert False, f"unknown Observation {on}"

    def _get_obs(self):
        obs = []
        for on in self.obs_config: obs.append(self._enum2obs(on))
        return np.concatenate(obs).astype(np.float32)
    
    def actuate(self, action): self.qpub.publish(Float64MultiArray(data=action))

    def set_goal(self, g):
        if self.task == ControlTask.Position:
            self.goal = np.clip(g, 0.0, 0.045)
        elif self.task == ControlTask.Force:
            self.goal = np.clip(g, 0, self.env.fmax)
        else:
            assert False, f"unknown ControlTask {self.task}"

    def reset(self): 
        """ initialize observation arrays
        """
        self.q     = np.array([None,None])
        self.qdot  = np.array([0,0])
        self.act   = np.array([0,0])

        self.force   = np.array([None,None])
        self.in_con  = np.array([0,0])
        self.had_con = np.array([0,0])

        self.active = True

        print("waiting for joint states ...")
        while not rospy.is_shutdown() and np.any(self.q==None): self.r.sleep()

        if self.task == ControlTask.Force:
            print("waiting for forces ...")
            while not rospy.is_shutdown() and np.any(self.force==None): self.r.sleep()


        if self.control_mode == ControlMode.Position:
            self.last_a = safe_rescale(self.q, [0.0, 0.045])
        elif self.control_mode == ControlMode.PositionDelta:
            self.last_a  = np.array([0,0])
        
        for _ in range(self.actions.maxlen): self.actions.append(self.last_a.copy())

        print("reset done!")

    def step(self): 
        obs = self._get_obs()
        raw_action, _ = self.model.predict(obs)

        if self.control_mode == ControlMode.Position:
            ain = safe_rescale(raw_action, [-1, 1], [0.0, 0.045])
        elif self.control_mode == ControlMode.PositionDelta:
            ain = safe_rescale(raw_action, [-1, 1], [-self.env.dq_max, self.env.dq_max])
            ain = np.clip(self.q+ain, 0, 0.045)

        self.actions.append(ain)
        ain_avg = np.mean(self.actions, axis=0)

        self.actuate(ain_avg)

        self.last_a = raw_action

    def stop(self): 
        print("killing thread ...")
        self.active = False
        self.exec_thread.join()
        print("done")

    def run(self): 
        print("running model ...")

        def _step_loop():
            while not rospy.is_shutdown() and self.active:
                self.step()
                self.r.sleep()
            print("_step_loop() finished")
        
        self.exec_thread = threading.Thread(target=_step_loop)
        self.exec_thread.start()


if __name__ == "__main__":

    from learning_fc import model_path
    from learning_fc.training import make_eval_env_model
    from learning_fc.utils import find_latest_model_in_path

    # trial = find_latest_model_in_path(model_path, filters=["ppo"])
    trial = f"{model_path}/2023-07-21_10-01-57__gripper_pos__ppo__pos_delta__obs_q-dq__nenv-6__k-1"
    env, model, _, _ = make_eval_env_model(trial, with_vis=False, checkpoint="best")

    from learning_fc.models import PosModel
    # model = PosModel(env)

    ri = RobotInterface(model, env, freq=100, goal=0.01, n_action_avg=1)
    ri.run()

    while not rospy.is_shutdown():
        inp = input("q = kill; st = stop; gXXX = goal; aXXX = actuate\n")
        if inp == "q":
            ri.stop()
            break

        elif inp == "st":
            ri.stop()

        elif inp[0]=="g":
            goal = inp[1:]
            try:
                goal = float(goal)
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