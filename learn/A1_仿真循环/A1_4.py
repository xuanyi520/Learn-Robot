import mujoco
import numpy as np
#axis 是铰链转轴方向的意思，其中0 1 0,指的是铰链绕着y轴转动
#<joint name="hinge" .../>给关节起了一个名字，电机要靠名字找到他
#<actuator>*<motor joint="hinge" ctrlrange="-50 50"/></actuator>
#装一个电机，接在hinge关节上，输出力矩范围到-50~50牛·米
#actuator就是“驱动器”，现实中是电机
#model.nu 从0变成1,nu = number of controls,控制量的个数，有几个电机就是几


xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1.2">
     <joint name="hinge" type="hinge" axis="0 1 0"/>
     <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("nq = ",model.nu)

for i in range(500):
    data.ctrl[0] = 10.0 #一直给10牛.米的力矩
    mujoco.mj_step(model, data)

print("500 步后角度:", round(data.qpos[0], 4), "弧度 =",
      round(np.degrees(data.qpos[0]), 2), "度")