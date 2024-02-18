import os 

datefmt='%Y-%m-%d_%H-%M-%S'
model_path = f"{os.environ['HOME']}/fc_models/"
tmp_model_path = "/tmp/fc_models/"

from .utils import *
from .enums import *

from collections import OrderedDict

cm_uni = OrderedDict()
cm_uni['red'] = '#A51E37'
cm_uni['gold'] = '#B4A069'
cm_uni['black'] = '#32414B'
cm_uni['darkblue'] = '#415A8C'
cm_uni['blue'] = '#0069AA'
cm_uni['lightblue'] = '#50AAC8'
cm_uni['cyan'] = '#82B9A0'
cm_uni['green'] = '#7DA54B'
cm_uni['darkgreen'] = '#326E1E'
cm_uni['lightred'] = '#C8503C'
cm_uni['magenta'] = '#AF6E96'
cm_uni['gray'] = '#B4A096'
cm_uni['lightorange'] = '#D7B469'
cm_uni['orange'] = '#D29600'
cm_uni['brown'] = '#916946'