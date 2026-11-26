"""A3 正运动学与雅可比：两条杆的手臂

运行：
    conda activate ai
    python learn/A3_正运动学与雅可比/A3_1.py

本课不训练、不控制，只回答两个问题：
    关节转多少  ->  末端在哪          （正运动学）
    关节动一点  ->  末端动多少        （雅可比）

四个实验：
    实验 1  正运动学：自己算的末端位置 vs MuJoCo 读出来的
    实验 2  雅可比是"每个关节各出多少力"：转 1 度看末端位移
    实验 3  三种算法算雅可比：手写解析 / 数值差分 / MuJoCo 自带
    实验 4  奇异位形：手臂伸直时，末端有一个方向动不了
"""

import numpy as np
import mujoco

np.set_printoptions(precision=4, suppress=True)

# 两段杆各长 0.3 米，绕 y 轴转，所以在 x-z 平面里摆（像腿）
L1 = 0.3
L2 = 0.3

XML = """
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


# ============================================================ 正运动学
def fk_hand(q1, q2):
    """自己推的公式：两个关节角 -> 末端 (x, z)

    初始方向朝下 (0, 0, -L)。绕 y 轴转 a 度后：
        x = -L * sin(a)
        z = -L * cos(a)
    第二段的角度是 q1 + q2（角度叠加）。
    """
    x = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    z = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    return np.array([x, z])


def jacobian_hand(q1, q2):
    """解析雅可比：对正运动学公式求偏导。

    第 1 行是 x 对 (q1, q2) 的偏导，第 2 行是 z 的。
    形状 (2, 2)：2 个方向 × 2 个关节
    """
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-L1 * c1 - L2 * c12, -L2 * c12],
        [L1 * s1 + L2 * s12,  L2 * s12],
    ])


def jacobian_fd(q1, q2, h=1e-6):
    """数值雅可比：不懂公式也能算，直接"动一点点看变多少"。

    对每个关节单独加 h，看末端位置变化了多少，除以 h 就是这一列。
    """
    x0 = fk_hand(q1, q2)
    col1 = (fk_hand(q1 + h, q2) - x0) / h
    col2 = (fk_hand(q1, q2 + h) - x0) / h
    return np.column_stack([col1, col2])


# ============================================================ 模型
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")


def set_q(q1, q2):
    """设好关节角，并调用 mj_forward 更新所有派生量（site 位置就在这里算）"""
    data.qpos[0] = q1
    data.qpos[1] = q2
    mujoco.mj_forward(model, data)
    return data.site_xpos[tip_id].copy()


# ============================================================ 实验 1
print("=" * 60)
print("实验 1：正运动学——手算 vs MuJoCo")
print("=" * 60)

for q1, q2 in [(0, 0), (np.pi / 2, 0), (0, np.pi / 2), (0.3, -0.7), (np.pi / 4, np.pi / 3)]:
    hand = fk_hand(q1, q2)
    sim = set_q(q1, q2)
    print(f"q1={q1:+.3f}  q2={q2:+.3f} rad  ({np.degrees(q1):+5.1f}, {np.degrees(q2):+5.1f} 度)")
    print(f"    手算   x={hand[0]:+.4f}  z={hand[1]:+.4f}")
    print(f"    MuJoCo x={sim[0]:+.4f}  z={sim[2]:+.4f}")
    print(f"    差     {np.linalg.norm(hand - np.array([sim[0], sim[2]])):.2e}")
print("结论：公式和 MuJoCo 完全一致，误差在 1e-16 量级（浮点误差）。\n")

# ============================================================ 实验 2
print("=" * 60)
print("实验 2：雅可比在说什么——转 1 度，末端动多少")
print("=" * 60)

q1, q2 = 0.3, 0.5
J = jacobian_hand(q1, q2)
print(f"当前姿态 q1={q1}, q2={q2}")
print(f"解析雅可比 J（每列对应一个关节）:\n{J}\n")

dq1 = np.radians(1.0)   # 只让 j1 转 1 度
p0 = fk_hand(q1, q2)
p1 = fk_hand(q1 + dq1, q2)
print(f"只转 j1 一度：末端移动 {np.linalg.norm(p1 - p0)*1000:.2f} 毫米")
print(f"用雅可比预测 J 的第 1 列 × dq1 = {J[:, 0] * dq1}  （单位：米）")
print(f"实际位移                      = {p1 - p0}  （单位：米）")
print("前三位对得上，最后一位差一点——雅可比是「一阶近似」，转得越少越准。")
print("这正好说明雅可比第 1 列就是「j1 转一点，末端往哪走、走多快」。")
print("第 2 列同理是 j2 的影响。\n")

# 速度层面的用法：关节速度 × 雅可比 = 末端速度
qd = np.array([1.0, -0.5])   # 关节速度 rad/s
v_end = J @ qd
print(f"关节速度 qd = {qd} rad/s")
print(f"雅可比 @ qd = 末端速度 {v_end} m/s  ← 后续整个运动学都是这个式子\n")

# ============================================================ 实验 3
print("=" * 60)
print("实验 3：三种算法算同一个雅可比")
print("=" * 60)

J_hand = jacobian_hand(q1, q2)
J_fd = jacobian_fd(q1, q2)

set_q(q1, q2)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, tip_id)
# MuJoCo 返回 3×2（x,y,z 三行），y 方向是 0，取第 0、2 行换成 (x,z)
J_mj = jacp[[0, 2], :]

print(f"手写解析:\n{J_hand}")
print(f"数值差分:\n{J_fd}")
print(f"MuJoCo  :\n{J_mj}")
print(f"解析 vs 差分   最大差 {np.abs(J_hand - J_fd).max():.2e}")
print(f"解析 vs MuJoCo 最大差 {np.abs(J_hand - J_mj).max():.2e}")
print("结论：三条路结果一样（差分的差异来自 h=1e-6 的截断误差，量级 1e-7）。")
print("实际工程里：能推导就用解析，推不动就用差分或直接调库。\n")

# ============================================================ 实验 4
print("=" * 60)
print("实验 4：奇异位形——伸直手臂，有个方向动不了")
print("=" * 60)

print("q2 从 90 度回到 0 度（手臂从折叠到伸直），看 det(J) 怎么变小：\n")
print("   q2(度)      det(J)")
for q2_deg in [90, 60, 30, 10, 1, 0]:
    q2r = np.radians(q2_deg)
    Jn = jacobian_hand(0.0, q2r)
    det = np.linalg.det(Jn)
    print(f"   {q2_deg:5.0f}   {det:+.6f}")

print("\n完全伸直（q1=0, q2=0）时的雅可比：")
J_sing = jacobian_hand(0.0, 0.0)
print(J_sing)
print("第 2 行全是 0 -> 末端在 z 方向的速度恒为 0。")
print("也就是说：手臂伸直朝下时，末端只能左右平移，没法上下动。")
print("这就是奇异位形（singularity）：雅可比掉秩，逆解会给出无穷大的关节速度。\n")

# 用 MuJoCo 数值确认：伸直状态下让末端往上走，需要关节转多少？
p_start = set_q(0.0, 0.0)
target = p_start + np.array([0, 0, 0.01])   # 往上抬 1 厘米
print(f"伸直状态下，目标点上升 1 厘米：{target}")
print("用雅可比反解需要 J 的逆，但 det(J)=0，逆不存在。")
print("下一课 A4 就是解决这个问题：加一个阻尼项，让它在奇异点附近「尽力而为」而不是爆炸。")

# 数值演示：在 det 很小的位置，逆解需要的关节速度有多大
for q2_deg in [30, 5, 1, 0.1]:
    q2r = np.radians(q2_deg)
    Jn = jacobian_hand(0.0, q2r)
    v_des = np.array([0.0, 0.1])       # 想让末端往上走 0.1 m/s
    dq = np.linalg.solve(Jn, v_des)    # J^-1 @ v
    print(f"q2={q2_deg:5.1f} 度  末端要 0.1 m/s 向上  ->  关节速度 {np.abs(dq).max():10.2f} rad/s")
print("q2 越小，需要的关节速度越夸张——0.1 度时已经离谱。实机到这里会触发保护。")
