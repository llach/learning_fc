import numpy as np
import mujoco as mj
import learning_fc
import mujoco_viewer

import xml.etree.ElementTree as ET

with_vis = 1

xmlmodel = ET.parse(learning_fc.__path__[0]+"/assets/pal_force.xml")
root = xmlmodel.getroot()

root.find("compiler").attrib["meshdir"] = learning_fc.__path__[0]+"/assets/meshes/"

model = mj.MjModel.from_xml_string(ET.tostring(xmlmodel.getroot(), encoding='utf8', method='xml'))

data = mj.MjData(model)

if with_vis:
    viewer = mujoco_viewer.MujocoViewer(model, data)

    viewer.cam.azimuth      = 153
    viewer.cam.distance     = 0.33
    viewer.cam.elevation    = -49
    viewer.cam.lookat       = [-0.00099796, -0.00137387, 0.04537828]

# fingers start in open position
data.joint("finger_joint_l").qpos = 0.045
data.joint("finger_joint_r").qpos = 0.045

steps = 150000
forces_l = np.zeros((5,steps)) # collect all individual pad forces, summation is done below
forces_r = np.zeros((5,steps))

for i in range(steps):
    data.ctrl = 2*[0.045]

    for j, c in enumerate(data.contact):
        name1 = data.geom(c.geom1).name
        name2 = data.geom(c.geom2).name

        if name1 != "object":  continue # lowest ID geoms come first
        if name2[:3] != "pad": continue # geoms for force measurements need to have "pad" in their name

        c_ft = np.zeros((6,))
        mj.mj_contactForce(model, data, j, c_ft)
        f = c_ft[0] # only consider normal force

        pidx = int(name2.split("_")[1])-1
        if name2[-2:] == "_l":
            forces_l[pidx, i] = f 
        elif name2[-2:] == "_r":
            forces_r[pidx, i] = f
        else: print(f"unknown pad {name2}")

    mj.mj_step(model, data)
    if with_vis: viewer.render()
    print(viewer.cam)

if with_vis: viewer.close()