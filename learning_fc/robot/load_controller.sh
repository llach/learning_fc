#!/bin/bash

export ROS_MASTER_URI=http://tiago-72c:11311
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rosparam load $SCRIPT_DIR/tiago_position_controller.yaml

ssh -t pal@tiago-72c << EOF
source /opt/pal/ferrum/setup.bash
rosservice call /controller_manager/load_controller "name: 'finger_left_controller'"
rosservice call /controller_manager/load_controller "name: 'finger_right_controller'"
rosservice call /controller_manager/switch_controller "start_controllers:
- 'finger_left_controller'
- 'finger_right_controller'
stop_controllers:
- 'gripper_controller'
- 'gripper_position_controller'
- 'myrmex_gripper_controller'
- 'gripper_force_controller'
strictness: 0"
EOF
