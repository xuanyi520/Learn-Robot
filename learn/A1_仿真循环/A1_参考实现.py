"""A1 仿真循环：让单摆站起来

运行（带窗口）：
    conda activate ai
    python learn/A1_仿真循环/A1_参考实现.py

运行（无窗口，只打印结果）：
    python learn/A1_仿真循环/A1_参考实现.py --headless
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------- 模型
# 一个挂在世界坐标上的摆：hinge 关节绕 y 轴转，杆朝下。
# qpos = 0   时杆朝下（自然下垂，稳定平衡）
# qpos = pi  时杆朝上（倒立，不稳定平衡）
XML = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 1 3" dir="0 -1 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.35 0.35 0.40 1"/>
    <body name="arm" pos="0 0 1.2">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"
            rgba="0 0.7 1 1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""

TARGET = np.pi          # 目标角度：竖起来
KP, KD = 80.0, 8.0      # 比例增益、微分增益
MAX_TORQUE = 50.0


def controller(model, data, kp=KP, kd=KD, target=TARGET):
    """PD 控制：力矩 = 位置误差 × kp - 速度 × kd"""
    q = data.qpos[0]
    qd = data.qvel[0]

    # 角度误差归一到 [-pi, pi]，避免从 359 度绕远路
    err = (target - q + np.pi) % (2 * np.pi) - np.pi

    data.ctrl[0] = np.clip(kp * err - kd * qd, -MAX_TORQUE, MAX_TORQUE)


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # ---------- 无窗口模式：只跑物理，打印结果 ----------
    if "--headless" in sys.argv:
        trace = []
        for step in range(4000):
            controller(model, data)
            mujoco.mj_step(model, data)
            if step % 200 == 0:
                trace.append((step, np.degrees(data.qpos[0]), data.ctrl[0]))

        print("步数    角度(度)   力矩(N·m)")
        for s, ang, tau in trace:
            print(f"{s:5d}   {ang:8.2f}   {tau:7.2f}")
        print(f"\n最终角度 {np.degrees(data.qpos[0]):.2f} 度（目标 180 度）")
        print(f"最终角速度 {data.qvel[0]:.4f} rad/s（越小越稳）")
        return

    # ---------- 带窗口模式 ----------
    print("窗口已打开：单摆会从下垂位置摆到竖起。关闭窗口退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            controller(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 240)
    print("窗口已关闭。")


if __name__ == "__main__":
    main()
