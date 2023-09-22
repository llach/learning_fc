import pyqtgraph as pg

try: 
    from PyQt6 import QtWidgets, QtCore
except:
    from PyQt5 import QtWidgets, QtCore

class VisBase:

    def __init__(self, env, title, windim=[1000, 600]):
        # store reference to environment
        self.env = env

        # create QT Application
        self.app = QtWidgets.QApplication([])

        # configure window
        self.win = pg.GraphicsLayoutWidget(show=True, )
        self.win.resize(*windim)
        self.win.setWindowTitle(title)

        # enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        # initialize counter
        self.t = 0

    def reset(self, hard=False):
        if hard:
            for plot in self.all_plots:
                plot.clear()
        else:
            self.draw_goal()

            for plot in self.all_plots:
                if self.t == 0: break
                plot.draw_line(
                    name=f"ep{self.t}", 
                    pos=self.t,
                    pen=dict(color="#D3D3D3", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                    angle=90
                    )
                
    def draw_goal(self): raise NotImplementedError

class PlotItemWrapper:

    def __init__(self, win: pg.GraphicsLayoutWidget, 
                 pens, title, yrange=None, ticks=None):
        
        self.data = []
        self.plot = win.addPlot(title=title)
        self.lines = {}

        self.pens = pens
        self._setup_pens()

        if yrange: self.plot.setYRange(*yrange)
        if ticks:  self.draw_ticks(ticks)

    def _setup_pens(self):
        # number of pens = number of curves
        self.curves = []
        self.curve_data = []
    
        if isinstance(self.pens, list) and len(self.pens)>1:
            for p in self.pens:
                self.curves.append(self.plot.plot(pen=p))
                self.curve_data.append([])
        elif isinstance(self.pens, str):
            self.curves = self.plot.plot(pen=self.pens)
        else:
            assert False, f"unsupported type for 'pens' {type(self.pens)}"

    def _remove_line(self, name): 
        self.plot.removeItem(self.lines[name])
        self.lines.pop(name)

    def clear(self):
        self.plot.clear()
        self._setup_pens()

    def draw_line(self, name, **lnargs):
        if name in self.lines: self._remove_line(name)
        self.lines = {**self.lines, name: pg.InfiniteLine(**lnargs)}
        self.plot.addItem(self.lines[name])

    def draw_ticks(self, ticks, axis="left"):
        ax = self.plot.getAxis(axis)
        ax.setTicks([[(v, str(v)) for v in ticks]])

    def update(self, v): 
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
            self.curve_data.append(v)
            self.curves.setData(self.curve_data)
        else:
            for x, curve, data in zip(v, self.curves, self.curve_data):
                data.append(x)
                curve.setData(data)