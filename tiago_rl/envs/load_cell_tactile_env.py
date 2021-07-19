import numpy as np
from collections import deque

from tiago_rl.envs import BulletRobotEnv


def force_delta(force_a, force_b):
    assert force_a.shape == force_b.shape
    return np.linalg.norm(force_a - force_b, axis=-1)


RAW_FORCES = 'raw'
BINARY_FORCES = 'binary'

SPARSE_REWARDS = 'sparse'
CONT_REWARDS = 'continuous'


class LoadCellTactileEnv(BulletRobotEnv):

    def __init__(self, joints, force_noise_mu=None, force_noise_sigma=None, force_smoothing=None,
                 target_forces=None, force_threshold=None, force_type=None, reward_type=None,
                 *args, **kwargs):

        self.force_smoothing = force_smoothing or 4
        self.force_noise_mu = force_noise_mu or 0.0
        self.force_noise_sigma = force_noise_sigma or 0.0077
        self.force_threshold = force_threshold or 3 * self.force_noise_sigma

        self.force_buffer_r = deque(maxlen=self.force_smoothing)
        self.force_buffer_l = deque(maxlen=self.force_smoothing)

        if target_forces is not None:
            self.target_forces = np.array(target_forces)
        else:
            self.target_forces = np.array([10.0, 10.0])

        self.force_type = force_type or RAW_FORCES
        self.reward_type = reward_type or CONT_REWARDS

        self.current_forces = np.array([0.0, 0.0])
        self.current_forces_raw = np.array([0.0, 0.0])

        if self.force_type not in [RAW_FORCES, BINARY_FORCES]:
            print(f"unknown force type: {self.force_type}")
            exit(-1)

        if self.reward_type not in [SPARSE_REWARDS, CONT_REWARDS]:
            print(f"unknown reward type: {self.reward_type}")
            exit(-1)

        BulletRobotEnv.__init__(self, joints=joints, *args, **kwargs)

    # BulletRobotEnv methods
    # ----------------------------

    def _transform_forces(self, force):
        return (force / 100) + np.random.normal(self.force_noise_mu, self.force_noise_sigma)

    def _get_obs(self):
        # get joint positions and velocities from superclass
        joint_states = super(LoadCellTactileEnv, self)._get_obs()['observation']

        if self.objectId:
            # get current contact forces
            self.force_buffer_r.append(self._get_contact_force(self.robotId, self.objectId,
                                       self.robot_link_to_index['gripper_right_finger_link'],
                                       self.object_link_to_index['object_link']))
            self.force_buffer_l.append(self._get_contact_force(self.robotId, self.objectId,
                                       self.robot_link_to_index['gripper_left_finger_link'],
                                       self.object_link_to_index['object_link']))

            # although forces are called "raw", the are averaged to be as close as possible to the real data.
            self.current_forces_raw = np.array([
                np.mean(self.force_buffer_r),
                np.mean(self.force_buffer_l)
            ])

            # calculate current forces based on force type
            if self.force_type == BINARY_FORCES:
                self.current_forces = (np.array(self.current_forces_raw) > self.force_threshold).astype(np.float32)
            elif self.force_type == RAW_FORCES:
                self.current_forces = self.current_forces_raw.copy()
            else:
                print(f"unknown force type: {self.force_type}")
                exit(-1)

        obs = np.concatenate([joint_states, self.current_forces])
        return {
            'observation': obs
        }

    def _is_success(self):
        """If the force delta between target and current force is smaller than the force threshold, it's a success.
        Note, that we use the observation forces here that are averaged over the last k samples. This may lead to
        this function returning False even though the desired force was hit for one sample (see tactile_demo). The
        alternative would be to calculate the success on data that differs from the observation, which an agent would
        not have access too. We assume that that behavior would be more confusing for an agent than it would be helpful.
        """
        delta_f = force_delta(self.current_forces_raw, self.target_forces)
        return (np.abs(delta_f) < self.force_threshold).astype(np.float32)

    def _compute_reward(self):
        delta_f = force_delta(self.current_forces_raw, self.target_forces)
        if self.reward_type == SPARSE_REWARDS:
            return -(delta_f < self.force_threshold).astype(np.float32)
        else:
            return -delta_f


class GripperTactileEnv(LoadCellTactileEnv):

    def __init__(self, initial_state=None, *args, **kwargs):

        joints = [
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
            'torso_to_arm'
        ]

        initial_state = initial_state or [
            0.045,
            0.045,
            0.0
        ]

        LoadCellTactileEnv.__init__(self,
                                    joints=joints,
                                    initial_state=initial_state,
                                    cam_yaw=120.5228271484375,
                                    cam_pitch=-68.42454528808594,
                                    cam_distance=1.1823151111602783,
                                    cam_target_position=(-0.2751278877258301, -0.15310688316822052, -0.27969369292259216),
                                    robot_model="gripper_tactile.urdf",
                                    robot_pos=[0.0, 0.0, 0.27],
                                    object_model="objects/object.urdf",
                                    object_pos=[0.04, 0.02, 0.6],
                                    *args,
                                    **kwargs)


class TIAGoTactileEnv(LoadCellTactileEnv):

    def __init__(self, initial_state=None, *args, **kwargs):

        joints = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint',
            'gripper_right_finger_joint',
            'gripper_left_finger_joint',
        ]
        
        initial_state = initial_state or [
             0.,
             2.71,
             -0.173,
             1.44,
             1.79,
             0.23,
             -0.0424,
             -0.0209,
             0.045,
             0.045
        ]
        
        LoadCellTactileEnv.__init__(self,
                                    joints=joints,
                                    initial_state=initial_state,
                                    cam_yaw=89.6000747680664,
                                    cam_pitch=-35.40000915527344,
                                    cam_distance=1.6000027656555176,
                                    robot_model="tiago_tactile.urdf",
                                    object_model="objects/object.urdf",
                                    object_pos=[0.73, 0.07, 0.6],
                                    table_model="objects/table.urdf",
                                    table_pos=[0.7, 0, 0.27],
                                    *args, **kwargs)