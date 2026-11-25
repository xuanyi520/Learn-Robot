"""A1 练习：扫描 PD 增益，看单摆能不能站稳

运行方式：
    PyCharm 里右键 -> 运行 'A1_练习'
    或者终端：conda activate ai && python A1_练习.py

改下面的参数组合，看输出怎么变。
"""

import mujoco
import numpy as np

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="arm" pos="0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrlrange="-50 50"/></actuator>
</mujoco>
"""


def trial(kp, kd, steps=4000):
    """跑一段仿真，返回最终角度（度）、角速度、最后 500 步的角度波动"""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    angles = []
    for _ in range(steps):
        # 角度误差归一到 [-pi, pi]
        err = (np.pi - data.qpos[0] + np.pi) % (2 * np.pi) - np.pi
        data.ctrl[0] = np.clip(kp * err - kd * data.qvel[0], -50, 50)
        mujoco.mj_step(model, data)
        angles.append(np.degrees(data.qpos[0]))

    tail = angles[-500:]
    return np.degrees(data.qpos[0]), data.qvel[0], max(tail) - min(tail)


print(f"{'kp':>6} {'kd':>6} {'最终角度':>10} {'角速度':>10} {'波动':>8}  判断")
print("-" * 60)

combos = [
    (0.5, 0.1),
    (2.0, 0.5),
    (10.0, 1.0),
    (80.0, 0.0),
    (80.0, 0.5),
    (80.0, 8.0),
    (300.0, 30.0),
]

for kp, kd in combos:
    ang, vel, swing = trial(kp, kd)
    if abs(abs(ang) - 180) < 1.0:
        verdict = "站住了"
    elif swing < 0.5:
        verdict = "卡住了，上不去"
    else:
        verdict = "在动，没稳住"

    print(f"{kp:6.1f} {kd:6.1f} {ang:10.1f} {vel:10.4f} {swing:8.1f}  {verdict}")

print()
print("要试别的参数，改上面 combos 列表，或者把 kd 改成 0 试试。")
