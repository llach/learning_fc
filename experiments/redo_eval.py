from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path
from learning_fc.training import tactile_eval, pos_eval

trial  = "2023-07-19_14-28-19__centered__minimal_reward__nenv-6__k-1"
latest = f"{model_path}/{trial}"
# latest = find_latest_model_in_path(model_path, filters=["ppo"])

# tactile_eval(latest, nrollouts=10, with_vis=0, training=False, checkpoint="best")
# pos_eval(latest, nrollouts=10, with_vis=0, training=False)

import os
# for x in os.listdir(model_path):
#     if "2023" not in x: continue
#     paf = f"{model_path}/{x}"
#     tactile_eval(paf, nrollouts=5, with_vis=0, training=False, checkpoint="best")
#     tactile_eval(paf, nrollouts=5, with_vis=0, training=False, checkpoint="latest")

# for x in os.listdir(f"{model_path}/_archive/"):
#     if "2023" not in x: continue
#     paf = f"{model_path}/_archive/{x}"
#     tactile_eval(paf, nrollouts=5, with_vis=0, training=False, checkpoint="best")
#     tactile_eval(paf, nrollouts=5, with_vis=0, training=False, checkpoint="latest")