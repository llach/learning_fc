from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

class Colors(str, Enum):
    # greys
    grey   = "#656565"
    davy_grey = "#5C5B5C"
    timberwolf = "#B3B3B3"
    onyx = "#3B3B3B"

    # instable trajectories colors
    purple = "#bf2dd2"
    green  = "#74BB44"

    fgoal = "#11922B"

    mean_r = "#3465d6"
    sigma_r = "#517ADB"

    act_var_mean = "#05A143"
    act_var_var = "#03692C"

    """ blue / orange pairs
    """
    tab10_0 = "#0192F8"
    tab10_1 = "#F96801"

    actions0 = "#6B9CFF", # cornflower
    actions1 = "#E37C22" # sunglow


class PLOTMODE(str, Enum):
    debug="debug"
    camera_ready="camera_ready"

class FIGTYPE(str, Enum):
    single="single"
    multicol="multicol"
    multirow="multirow"

EPS_SEP_LINE_KW = dict(
    lw=.7, 
    ls="dashed",
    c="grey"
)

SEND_LINE_KW = dict(
    color=Colors.onyx,
    ls="dashed",
    lw=1.2
)

def set_rcParams(mode: PLOTMODE = PLOTMODE.debug, ftype: FIGTYPE = FIGTYPE.single, nrows=None):
    # legend 
    plt.rcParams['legend.fancybox']  = False
    plt.rcParams['legend.edgecolor'] = "#6C6C6D"
    plt.rcParams['legend.borderpad'] = 0.7

    # axes
    plt.rcParams['xtick.direction'] = "in"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['axes.spines.top']   = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['lines.linewidth'] = 1.5

    # axes background grid
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.75
    plt.rcParams['grid.color'] = Colors.timberwolf
    plt.rcParams['grid.linestyle'] = (0, (1, 4)) # loosely dotted; 1pt line, 4pt spacing
    plt.rcParams['grid.linewidth'] = 0.7
    
    # font sizes
    plt.rcParams['font.size'] = 21
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    # figure config
    if ftype == FIGTYPE.single or ftype == FIGTYPE.multicol:
        figsize = (7.8, 5.5)
    elif ftype == FIGTYPE.multirow:
        figsize = (7.8, nrows*3.5)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.constrained_layout.use'] = True

    # camera ready: slower but nicer
    if mode == PLOTMODE.camera_ready:
        # same font as text 
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Palatino']

        # high quality figure
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.format"] = "pdf"

        # use LaTeX
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    return plt.rcParams['text.usetex']

def setup_axis(
        ax: plt.Axes,
        xlim=None, 
        ylim=None,
        xlabel=None,
        ylabel=None,
        xticks=None,
        yticks=None,
        xticklabels=None,
        yticklabels=None,
        legend_items=[],
        legend_loc=None,
        remove_xticks=False,
        remove_first_ytick=False,
    ): 
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    if xticks is not None:
        ax.set_xticks(xticks) 
        lbls = xticklabels or [str(ti) for ti in xticks]
        ax.set_xticklabels(lbls)

    if yticks is not None:
        ax.set_yticks(yticks) 
        lbls = yticklabels or [str(ti) for ti in yticks]
        ax.set_yticklabels(lbls)

    if remove_first_ytick: 
        labels = ax.get_yticklabels()
        labels[0].set(text="")
        ax.set_yticks(ax.get_yticks()) 
        ax.set_yticklabels(labels)

    if remove_xticks:
        ax.set_xticklabels([])

    if len(legend_items)>0:
        legend = ax.legend(
            *legend_items,
            loc=legend_loc,
            handler_map={tuple: HandlerTuple(ndivide=None)}
        )
        legend.get_frame().set_linewidth(1.0)