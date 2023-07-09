from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path
from learning_fc.training import tactile_eval, pos_eval

# latest = find_latest_model_in_path(model_path, filters=["ppo"])
# print(latest)
# tactile_eval(latest, nrollouts=10, with_vis=0, training=False)

trial  = "2023-07-08_20-18-25__gripper_pos__ppo__pos__obs_q-dq__nenv-1__k-3"
latest = f"{model_path}/{trial}"
# latest = find_latest_model_in_path(model_path, filters=["gripper_pos"])
# print(latest)
pos_eval(latest, nrollouts=10, with_vis=0, training=False)