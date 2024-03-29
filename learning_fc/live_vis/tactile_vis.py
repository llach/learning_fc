import platform
import numpy as np

try: 
    from PyQt6 import QtCore
except:
    from PyQt5 import QtCore
from learning_fc import safe_rescale
from learning_fc.live_vis import VisBase, PlotItemWrapper as PIWrapper


class TactileVis(VisBase):

    def __init__(self, env):
        VisBase.__init__(
            self,
            env=env,
            title="Force Visualisation"
        )

        # create plots and curves
        self.plt_force = PIWrapper(self.win, title="Contact Forces", pens=["r", "y"], yrange=[-0.05, 1.2*env.max_fmax])
        self.plt_cntct = PIWrapper(self.win, title="Actions", pens=["r", "y"], yrange=[-1.05, 1.05], ticks=[-1,1])
        
        # draw lines at threshold and goal force
        self.draw_goal()
        self.plt_force.draw_line(
                name="ftheta",
                pos=env.fth,
                angle=0
            )

        self.plt_cntct.draw_line(
            name="a_min",
            pos=-1,
            angle=0
        )
        self.plt_cntct.draw_line(
            name="a_max",
            pos=1,
            angle=0
        )

        self.plt_cntct.draw_line(
            name="a_zero",
            pos=0,
            angle=0
        )
        self.plt_cntct.draw_line(
            name="a_zero_lower",
            pos=-0.09,
            angle=0,
            pen=dict(color="#D3D3D3", width=1, style=QtCore.Qt.PenStyle.DotLine)
        )
        self.plt_cntct.draw_line(
            name="a_zero_upper",
            pos=0.09,
            angle=0,
            pen=dict(color="#D3D3D3", width=1, style=QtCore.Qt.PenStyle.DotLine)
        )
        
        self.win.nextRow()

        self.plt_pos = PIWrapper(self.win, title="Joint Positions", pens=["r", "y", "c", "b"], yrange=[-0.005, 0.05], ticks=[0.045, 0.02, 0.0])

        vmax = self.env.vmax
        self.plt_vel = PIWrapper(self.win, title="Joint Velocities", pens=["r", "y"], yrange=[-1.2*vmax, 1.2*vmax], ticks=[-vmax, vmax])
        self.plt_vel.draw_line(
            name="upper_limit",
            pos=vmax,
            angle=0
        )
        self.plt_vel.draw_line(
            name="lower_limit",
            pos=-vmax,
            angle=0
        )

        self.win.nextRow()

        self.plt_acc = PIWrapper(self.win, title="Joint Accelerations", pens=["r", "y"])
        self.plt_vobj = PIWrapper(self.win, title="Object Velocity", pens=["r", "g", "b"])


        self.win.nextRow()

        self.plt_r = PIWrapper(self.win, title="r(t)", pens="g")
        self.plt_r_parts = PIWrapper(self.win, title="r_force | r_obj_pos | r_prox", pens=["b", "y", "c", "g"])

        # self.win.nextRow()

        # self.plt_r_obj_prx = PIWrapper(self.win, title="r_obj_prx", pens="b", yrange=[-0.1, 2.2], ticks=[0,2])
        # self.plt_r_qdot = PIWrapper(self.win, title="r_qdot & r_qacc", pens=["r", "c"], yrange=[0.1, -2.2], ticks=[0,-2])

        self.all_plots = [
            self.plt_force, self.plt_cntct, 
            self.plt_pos, self.plt_vel, 
            self.plt_acc, self.plt_vobj, 
            self.plt_r, self.plt_r_parts, 
        ]

    def draw_goal(self):
        fth    = self.env.fth
        fgoal  = self.env.fgoal

        self.plt_force.draw_line(
            name="fgoal",
            pos=round(fgoal, 3),
            angle=0,
            pen=dict(color="#00FF00", width=1)
        )
        self.plt_force.draw_line(
            name="noise_upper",
            pos=round(fgoal+fth, 3),
            angle=0,
            pen=dict(color="#D3D3D3", width=1, style=QtCore.Qt.PenStyle.DotLine)
        )
        self.plt_force.draw_line(
            name="noise_lower",
            pos=round(fgoal-fth, 3),
            angle=0,
            pen=dict(color="#D3D3D3", width=1, style=QtCore.Qt.PenStyle.DotLine)
        )

        self.plt_force.draw_line(
            name="fmax",
            pos=round(self.env.fmax, 3),
            angle=0,
            pen=dict(color="#FF0000", width=1, style=QtCore.Qt.PenStyle.DotLine)
        )

        self.plt_force.draw_ticks([0, round(fgoal, 2), round(self.env.max_fmax, 2)])

    def update_plot(self, action, reward):
        self.t += 1
        action = safe_rescale(action, [-1,1], [-0.045,0.045])

        # store new data
        self.plt_force.update(self.env.force)
        self.plt_cntct.update(self.env.last_a)

        self.plt_pos.update(np.concatenate([self.env.q, self.env.qdes]))
        self.plt_vel.update(self.env.qdot)

        self.plt_acc.update(self.env.fdot)
        self.plt_vobj.update(np.abs(self.env.obj_v))

        self.plt_r.update(reward)
        self.plt_r_parts.update([self.env.r_force, self.env.r_obj_pos, self.env.r_obj_prox, self.env.r_act])

        # self.plt_r_obj_prx.update(self.env.r_obj_prx)
        # self.plt_r_qdot.update([-self.env.r_qdot, -self.env.r_qacc])

        # on macOS, calling processEvents() is unnecessary
        # and even results in an error. only do so on Linux
        if platform.system() == 'Linux':
            self.app.processEvents()

