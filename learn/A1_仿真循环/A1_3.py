import mujoco
import numpy as np
#<body pos="...">这个刚体的坐标系原点
#<geom pos="...">这块形状的位置
#<geom pos="0 0 0.1">是说这个球的中心在刚体原点上方10厘米，不是在世界上方

#<joint type="hinge" axis="0 1 0"/>指的是铰链关节：只能绕y轴旋转，其他方向不动。这个关节的原点在刚体原点

xml = """
<mujoco>
    <worldbody>
        <body pos="0 0 1.2">
            <joint  type="hinge"  axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("nq =",model.nq," nv = ",model.nv)

data.qpos[0] = 1.0 #手动给它一个初始角度

for i in range(500):
    mujoco.mj_step(model, data)

print("500 步后角度:", round(data.qpos[0], 4), "弧度 =",
      round(np.degrees(data.qpos[0]), 2), "度")
print("当前角速度:", round(data.qvel[0], 4))