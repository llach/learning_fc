from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

grey   = "#656565"
purple = "#bf2dd2"
green  = "#74BB44"
davy_grey = "#5C5B5C"
timberwolf = "#B3B3B3"

class PLOTMODE(str, Enum):
    debug="debug"
    camera_ready="camera_ready"

class FIGTYPE(str, Enum):
    single="single"
    multicol="multicol"
    multirow="multirow"


def set_rcParams(mode: PLOTMODE = PLOTMODE.debug, ftype: FIGTYPE = FIGTYPE.single, nrows=None):
    # legend 
    plt.rcParams['legend.fancybox']  = False
    plt.rcParams['legend.edgecolor'] = "#6C6C6D"
    plt.rcParams['legend.borderpad'] = 0.7

    # axes setup
    plt.rcParams['xtick.direction'] = "in"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['axes.spines.top']   = False
    plt.rcParams['axes.spines.right'] = False

    # font sizes
    plt.rcParams['font.size'] = 21
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    if ftype == FIGTYPE.single or ftype == FIGTYPE.multicol:
        figsize = (7.8, 5.5)
    elif ftype == FIGTYPE.multirow:
        figsize = (7.8, nrows*5.5)
    plt.rcParams['figure.figsize'] = figsize

    if mode == PLOTMODE.camera_ready:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Palatino']

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.format"] = "pdf"

        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def setup_axis(
        ax, 
        xlabel=None, 
        ylabel=None, 
        xlim=None, 
        ylim=None, 
        remove_first_ytick=True,
        draw_grid=True,
        legend_items=[],
        legend_loc=None,
    ): 

    if ylabel: ax.set_xlabel(xlabel)
    if xlabel: ax.set_ylabel(ylabel)

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    if remove_first_ytick: 
        labels = ax.get_yticklabels()
        labels[0].set(text="")
        ax.set_yticks(ax.get_yticks()) 
        ax.set_yticklabels(labels)

    if draw_grid: ax.grid(visible=True, ls="dotted", lw=1.0, c=timberwolf)

    if len(legend_items)>0:
        legend = ax.legend(
            *legend_items,
            loc=legend_loc,
            handler_map={tuple: HandlerTuple(ndivide=None)}
        )
        legend.get_frame().set_linewidth(1.0)

def finish_fig(fig, suptitle=None): 
    if suptitle: fig.suptitle(suptitle)
    fig.tight_layout(pad=0.01)