from learning_fc import model_path
from learning_fc.utils import find_latest_model_in_path
from learning_fc.training import tactile_eval

latest = find_latest_model_in_path(model_path, filters=["ppo"])
print(latest)
tactile_eval(latest, with_vis=0, training=False)