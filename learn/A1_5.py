import mujoco
import numpy as np

#capsule胶囊体，圆柱体两端是半球,fromto定义它的两个端点,size是它的半径
#fromto="0 0 0 0 0 -0.5"是六个数字
#前面三个数字是起点。后面三个数字是终点，（0,0,0）->(0,0,-0.5)

xml = """
<mujoco>
    <worldbody>
        <body pos="0 0 1.2">
            <joint  name="hinge" type="hinge" axis="0 1 0"/>
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

KP = 80.0
KD = 8.0
TARGET = np.pi #目标角度：竖起来

for i in range(4000):
    #1.读当前的状态
    q = data.qpos[0] #当前位置、角度
    qd = data.qvel[0] #当前角速度
    #2.计算误差,归一到[-pi, pi]
    #公式1（角度归一化):err = (TARGET - q + np.pi) % (2 * np.pi) - np.pi
    #（目标角度-当前角度+圆周率）%（2*圆周率）-圆周率，这样得到的结果落点在[-pi, pi]区间内
    err = (TARGET - q + np.pi) % (2 * np.pi) - np.pi
    #3.计算力矩
    #PD控制器公式:力矩 = KP * 误差 - KD * 角速度
    #np.clip()函数的作用是将力矩限制在[-50, 50]范围内，防止力矩过大导致仿真不稳定
    data.ctrl[0] = np.clip(KP * err - KD * qd, -50, 50) #限制力矩范围
    #4.推进仿真
    mujoco.mj_step(model, data)

    if i in (0, 20, 50, 100, 200, 400):
        print(f"第 {i+1:4d} 步: 角度 {np.degrees(data.qpos[0]):8.1f} 度, "
              f"力矩 {data.ctrl[0]:7.2f}")

print("最终角度:", round(np.degrees(data.qpos[0]), 2), "度")