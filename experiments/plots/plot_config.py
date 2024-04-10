import matplotlib.pyplot as plt

# legend 
plt.rcParams['legend.fancybox']  = True
plt.rcParams['legend.handlelength']  = 1.2
plt.rcParams['legend.edgecolor'] = "#000000"

# axes
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = "medium"
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.borderaxespad'] = 1.2

plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage[T1]{fontenc}\usepackage{mathpazo}'