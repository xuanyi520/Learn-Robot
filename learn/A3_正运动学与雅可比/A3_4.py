import numpy as np
import mujoco

np.set_printoptions(precision=4, suppress=True)

L1 = 0.3
L2 = 0.3

XML="""
<mujoco>
    <compiler angle="radian"/>
    <worldbody>
        <body name="link1" pos="0 0 0">
            <joint name="j1" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
            <body name="link2" pos="0 0 -0.3">
                <joint name="j2" type="hinge" axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
                <site name="tip" pos="0 0 -0.3" size="0.01"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

def fk_hand(q1,q2):
    """两段杆的正运动学：关节角 -> 末端 (x, z)。"""
    x = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    z = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    return np.array([x, z])

def set_q(q1, q2):
    """设好两个关节角，让 MuJoCo 算一遍所有派生量，返回末端世界坐标。"""
    data.qpos[0] = q1
    data.qpos[1] = q2
    mujoco.mj_forward(model, data)      # 只算运动学/动力学，不推进时间
    return data.site_xpos[tip_id].copy()

for q1, q2 in [(0.0, 0.0), (np.pi / 2, 0), (0, np.pi / 2),
               (0.3, -0.7), (np.pi / 4, np.pi / 3)]:
    hand = fk_hand(q1, q2)
    sim = set_q(q1, q2)

    print(f"q1={q1:+.3f}  q2={q2:+.3f} rad")
    print(f"    手算   x={hand[0]:+.4f}  z={hand[1]:+.4f}")
    print(f"    MuJoCo x={sim[0]:+.4f}  z={sim[2]:+.4f}")
    # 只比较 x 和 z 两个分量（y 恒为 0）
    print(f"    差     {np.linalg.norm(hand - np.array([sim[0], sim[2]])):.2e}")
