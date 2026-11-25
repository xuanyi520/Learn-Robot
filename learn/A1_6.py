import time
import mujoco
import numpy as np
import mujoco.viewer

#<option gravity="0 0 -9.81" timestep="0.002"/>设置重力和步长
#<light pos="0 1 3" dir="0 -1 -1"/>,放置一盏灯，位置（0,1,3）,往-y,-z方向照
#<geom type="plane" size="5 5 0.1" rgba="0.35 0.35 0.40 1"/>
#地面，灰色，rgba是红、绿、蓝、透明度，每个0~1，最后的1是不透明度
#<geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"
#                  rgba="0 0.7 1 1" mass="1"/>
#腿变成蓝色，并且强制质量=1KG
xml = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.002"/>
    <worldbody>
        <light pos="0 1 3" dir="0 -1 -1"/>
        <geom type="plane" size="5 5 0.1" rgba="0.35 0.35 0.40 1"/>
        <body pos="0 0 1.2">
            <joint name="hinge" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"
                  rgba="0 0.7 1 1" mass="1"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="hinge" ctrlrange="-50 50"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

KP, KD, TARGET = 80.0, 8.0, np.pi

#打开一个窗口，把model和data交给它，
#passive（被动）的意思是：仿真循环还是你控制，viewer 只负责画
with mujoco.viewer.launch_passive(model,data) as viewer:
    #只要窗口还开着就跑
    while viewer.is_running():
        #data.qpos[0]:当前位置
        err = (TARGET - data.qpos[0] + np.pi) % (2 * np.pi) -np.pi
        #data.qvel[0]:当前角速度
        data.ctrl[0] = np.clip(KP * err - KD * data.qvel[0], -50, 50)

        mujoco.mj_step(model,data)
        viewer.sync()
        time.sleep(1 / 240)