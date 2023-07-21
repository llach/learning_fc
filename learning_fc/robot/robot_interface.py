import rospy
import numpy as np

from enum import Enum
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

# from learning_fc.enums import ControlTask, ControlMode

class ControlTask(str, Enum):
    Force="force"
    Position="position"

class Observation(str, Enum):
    Pos="q"
    Vel="qdot"
    Acc="qacc"
    Force="f"
    Action="act"

    PosDelta="dq"
    ForceDelta="df"

    InCon="inC"
    HadCon="hadC"

class RobotInterface:

    JOINT_NAMES = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

    def __init__(self, model, obs_conf, goal=0.0, fth=0.001, task=ControlTask.Force, freq=50):
        self.fth = fth
        self.task = task
        self.goal = goal
        self.freq = freq
        self.model = model
        self.obs_conf = obs_conf
        
        """ initialize observation arrays
        """
        self.q     = np.array([0,0])
        self.qdot  = np.array([0,0])
        self.act   = np.array([0,0])

        self.force   = np.array([0,0])
        self.in_con  = np.array([0,0])
        self.had_con = np.array([0,0])

        self.js_idx = [None, None]

        rospy.init_node("model_robot_interface")

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
        self.js_sub = rospy.Subscriber("/joint_states", JointState, self._js_cb)
        self.force_sub = rospy.Subscriber("/ta11", Float32MultiArray, self.force_cb)
    
    def _goal_delta(self): 
        if self.task == ControlTask.Force:
            return self.goal - self.force
        elif self.task == ControlTask.Position:
            return self.q - self.goal
        else:
            assert False, "unknown ControlTask"


    def _enum2obs(self, on):
        # TODO normalize!

        if on == Observation.Pos: return self.q
        if on == Observation.Vel: return self.qdot
        if on == Observation.Force: return self.force
        if on == Observation.Action: return self.act
        if on == Observation.PosDelta: return self._goal_delta()
        if on == Observation.ForceDelta: return self._goal_delta()
        if on == Observation.InCon: return self.in_con
        if on == Observation.HadCon: return self.had_con

        assert False, "unknown Observation"

    def _get_obs(self):
        obs = []
        for on in self.obs_conf: obs.append(self._enum2obs(on))
        return np.concatenate(obs)
    
    def actuate(self, action): pass

    def reset(self): 
        self.active = True
        self.had_con = np.array([0,0])
        self.last_a  = np.array([0,0]) # TODO change depending on ctrl mode

    def step(self): 
        obs = self._get_obs()

        # action, _ = self.model.predict(obs)
        # self.actuate(action)

        # self.last_a = action

    def run(self): 
        r = rospy.Rate(self.freq)
        while not rospy.is_shutdown() and self.active:
            self.step()
            r.sleep()


if __name__ == "__main__":
    ri = RobotInterface(None, [Observation.Pos, Observation.Vel])
    ri.run()
