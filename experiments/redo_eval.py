from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path
from learning_fc.training import tactile_eval, pos_eval

# trial  = "2023-07-09_16-31-37__gripper_tactile__ppo__pos_delta__obs_q-qdot-f-df-inC-hadC__nenv-6__k-1"
# latest = f"{model_path}/{trial}"
latest = find_latest_model_in_path(model_path, filters=["ppo"])

tactile_eval(latest, nrollouts=10, with_vis=0, training=False)
# pos_eval(latest, nrollouts=10, with_vis=0, training=False)