from learning_fc.utils import find_latest_model_in_path
from learning_fc.training import tactile_eval

latest = find_latest_model_in_path("/tmp/tactile2/")
tactile_eval(latest, with_vis=0)