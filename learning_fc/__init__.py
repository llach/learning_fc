import os 

datefmt='%Y-%m-%d_%H-%M-%S'
model_path = f"{os.environ['HOME']}/fc_models/"
tmp_model_path = "/tmp/fc_models/"

from .utils import *