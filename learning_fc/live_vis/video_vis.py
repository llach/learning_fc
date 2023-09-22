import numpy as np

try: 
    from PyQt6 import QtCore
except:
    from PyQt5 import QtCore
from learning_fc.live_vis import VisBase, PlotItemWrapper as PIWrapper


class VideoVis(VisBase):

    def __init__(self, env):
        VisBase.__init__(
            self,
            env=env,
            title="",
            windim=[960,720]
        )

        self.win.setBackground("w")

        self.plt_force = PIWrapper(
            self.win, 
            title="", 
            pens=[
                dict(color="#3387bc", width=2),
                dict(color="#d83742", width=2),
                dict(color="#38d248", width=2)
            ], 
            yrange=[-0.0, 1.0], 
            ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        self.win.nextRow()
        
        self.plt_act = PIWrapper(
            self.win, 
            title="", 
            pens=[
                dict(color="#f18d1d", width=2),
                dict(color="#e665c1", width=2),
            ], 
            yrange=[-1.0, 1.0], 
            ticks=[-1,0,1]
        )
        
        self.plt_act.draw_line(
            name="a_zero",
            pos=0,
            angle=0,
            pen=dict(color="#000000", width=0.5)
        )


        self.all_plots = [
            self.plt_force, self.plt_act
        ]

    def draw_goal(self): pass

    def update_plot(self, action):
        self.t += 1

        # store new data
        self.plt_force.update(np.concatenate([self.env.force, [self.env.fgoal]]))
        self.plt_act.update(action)
