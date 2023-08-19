import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

grey   = "#656565"
purple = "#bf2dd2"
green  = "#74BB44"
davy_grey = "#5C5B5C"
timberwolf = "#B3B3B3"

def set_rcParams(usetex=True):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino']

    if usetex:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rc('text', usetex=True)

def setup_axis(
        ax, 
        xlabel=None, 
        ylabel=None, 
        xlim=None, 
        ylim=None, 
        fs_label=20, 
        fs_tick=13,
        fs_legend=14,
        remove_outer_borders=True,
        inner_ticks=True,
        remove_first_ytick=True,
        draw_grid=True,
        legend_items=[],
        legend_loc=None,
    ): 
    ax.tick_params(axis='both', labelsize=fs_tick)

    if ylabel: ax.set_xlabel(xlabel, fontsize=fs_label)
    if xlabel: ax.set_ylabel(ylabel, fontsize=fs_label)

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

    if remove_outer_borders: ax.spines[['right', 'top']].set_visible(False)
    if inner_ticks: ax.tick_params(axis="both",direction="in")

    if remove_first_ytick: 
        labels = ax.get_yticklabels()
        labels[0].set(text="")
        ax.set_yticklabels(labels)

    if draw_grid: ax.grid(visible=True, ls="dotted", lw=1.0, c=timberwolf)

    if len(legend_items)>0:
        legend = ax.legend(
            *legend_items,
            loc=legend_loc,
            prop={'size': fs_legend},
            fancybox=False,
            edgecolor="#6C6C6D",
            borderpad=0.7,
            handler_map={tuple: HandlerTuple(ndivide=None)}
        )
        legend.get_frame().set_linewidth(1.0)

def finish_fig(fig, suptitle=None): 
    if suptitle: fig.suptitle(suptitle)
    fig.tight_layout()