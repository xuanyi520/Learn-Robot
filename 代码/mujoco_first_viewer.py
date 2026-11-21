"""MuJoCo 第一个可视化 demo —— S2 阶段用。

运行：conda activate ai && python mujoco_first_viewer.py

窗口操作：
    鼠标左键拖动 = 旋转视角
    鼠标右键拖动 = 平移
    滚轮          = 缩放
    双击物体      = 施加一个力
    空格          = 暂停

场景里有一个箱子和一个球自由下落，落到地面停住。
"""

import time

import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.35 0.35 0.40 1"/>
    <body name="box" pos="-0.5 0 3">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" rgba="0.0 0.7 1.0 1"/>
    </body>
    <body name="ball" pos="0.7 0 4.5">
      <freejoint/>
      <geom type="sphere" size="0.3" rgba="1.0 0.55 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("窗口已打开。关闭窗口即退出。")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1 / 240)
print("窗口已关闭。")
