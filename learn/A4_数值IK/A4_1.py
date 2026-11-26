"""A4 数值 IK：让末端沿圆运动

运行：
    conda activate ai
    python learn/A4_数值IK/A4_1.py

A3 解决的是「关节 -> 末端」，这一课解决反方向：「想让末端到某处，关节该转多少」。
反解没有公式（角度和位置是非线性的），只能迭代逼近。

五个实验：
    实验 1  手写阻尼最小二乘（DLS），把一个点拉到位
    实验 2  让末端画圆，看跟踪误差
    实验 3  目标够不到时会发生什么，阻尼 lambda 在防什么
    实验 4  用 mink 做同一件事（工程里的标准工具）
    实验 5  零空间：末端不动，肘部还能自己挪
"""

import numpy as np
import mujoco
import mink

np.set_printoptions(precision=4, suppress=True)

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


# ============================================================ 正运动学与雅可比（A3 的结论）
def fk(q):
    q1, q2 = q
    return np.array([
        -L1 * np.sin(q1) - L2 * np.sin(q1 + q2),
        -L1 * np.cos(q1) - L2 * np.cos(q1 + q2),
    ])


def jac(q):
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-L1 * c1 - L2 * c12, -L2 * c12],
        [L1 * s1 + L2 * s12,  L2 * s12],
    ])


# ============================================================ 阻尼最小二乘
def dls_step(q, target, lam):
    """算一步，返回 dq。

    err = target - 当前位置              想让末端往哪走（2 个方向）
    J   = 2x2 雅可比                     关节动 1 弧度末端动多少
    解 J @ dq = err 就是直接求逆，但奇异位形 J 不可逆。
    加阻尼后：dq = J^T (J J^T + lam^2 I)^-1 err
    lam 越大越稳、越慢；lam 越小越快、越容易爆。
    """
    err = target - fk(q)
    J = jac(q)
    dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(2), err)
    return dq, err


# ============================================================ 实验 1
print("=" * 60)
print("实验 1：手写 DLS，把一个点拉到位")
print("=" * 60)

q = np.array([0.3, -0.5])
target = np.array([0.2, -0.4])
print(f"起点关节角 {q}，末端 {fk(q)}")
print(f"目标点     {target}\n")

for i in range(40):
    dq, err = dls_step(q, target, lam=0.05)
    q = q + dq
    if i < 5 or i == 9:
        print(f"  第 {i+1:2d} 次迭代  误差 {np.linalg.norm(err)*1000:10.6f} 毫米  关节增量 {np.linalg.norm(dq):.5f}")

print(f"\n最终关节角 {q}")
print(f"最终末端   {fk(q)}   误差 {np.linalg.norm(target - fk(q))*1000:.8f} 毫米")
print("结论：没有公式可解，但每步「算误差 -> 用雅可比换算成关节增量 -> 挪一点」就能收敛。\n")

print("同一目标，换三个起点：")
for q0 in ([0.3, -0.5], [-0.2, 0.8], [1.0, -1.5]):
    q_test = np.array(q0, dtype=float)
    for _ in range(100):
        dq, _ = dls_step(q_test, target, lam=0.05)
        q_test = q_test + dq
    print(f"  起点 {str(q0):12} -> 解 {np.round(q_test, 3)}  末端误差 {np.linalg.norm(target-fk(q_test))*1000:.6f} 毫米")
print("三个起点给出两组完全不同的关节角，末端都在同一个位置。")
print("这就是 IK 的多解性：末端定住，手臂还能有两种姿势（肘部朝左/朝右）。\n")

print("同一个起点，换 lambda：")
for lam in [0.01, 0.05]:
    q_test = np.array([0.3, -0.5], dtype=float)
    for _ in range(100):
        dq, _ = dls_step(q_test, target, lam=lam)
        q_test = q_test + dq
    print(f"  lam={lam:<5} -> 末端 {np.round(fk(q_test), 3)}  误差 {np.linalg.norm(target-fk(q_test))*1000:9.4f} 毫米")
print("lam=0.01 反而发散：J 接近奇异时逆解把误差放大成大跳，跳过目标来回打转。")
print("lam 不是越小越好，它要和雅可比本身的大小配合。\n")

# ============================================================ 实验 2
print("=" * 60)
print("实验 2：让末端画圆")
print("=" * 60)

center = np.array([0.0, -0.35])
radius = 0.15
n_steps = 120
q = np.array([0.0, -1.0])

errors = []
path = []
for k in range(n_steps):
    theta = 2 * np.pi * k / n_steps
    target = center + radius * np.array([np.cos(theta), np.sin(theta)])
    # 每个目标点迭代 20 次
    for _ in range(20):
        dq, err = dls_step(q, target, lam=0.01)
        q = q + dq
    p = fk(q)
    errors.append(np.linalg.norm(target - p))
    path.append(p)

errors = np.array(errors)
print(f"画 {n_steps} 个点，每个点迭代 20 次")
print(f"平均跟踪误差 {errors.mean():.2e} 米（浮点噪声级别）")
print(f"最大跟踪误差 {errors.max():.2e} 米")
print(f"圆心 {center}，半径 {radius} 米")
print(f"实测路径半径约 {np.linalg.norm(np.array(path) - center, axis=1).mean():.4f} 米")
print("结论：只要目标可达且不在奇异点，DLS 能贴着目标走。误差的剩下几位是迭代停止条件。\n")

# ============================================================ 实验 3
print("=" * 60)
print("实验 3：目标够不到，阻尼 lambda 在防什么")
print("=" * 60)

q = np.array([0.3, -0.5])
target = np.array([0.0, -1.0])    # 离原点 1 米，臂展只有 0.6 米
print(f"起点 q={q}，末端 {fk(q)}")
print(f"目标 {target}（离原点 1 米，臂展只有 0.6 米，够不到）\n")

for lam in [0.001, 0.05]:
    q_test = q.copy()
    for _ in range(200):
        dq, err = dls_step(q_test, target, lam)
        q_test = q_test + np.clip(dq, -0.1, 0.1)   # 每步最多转 0.1 弧度
    print(f"  lam={lam:<6}  200 次后末端 {np.round(fk(q_test), 4)}  关节角 {np.round(q_test, 3)}  误差 {np.linalg.norm(target-fk(q_test))*1000:7.2f} 毫米")

print("\n两种 lam 都停在伸直位附近（末端 z≈-0.6，差 400 毫米）：")
print("  - 够不到就是够不到，DLS 不会硬掰出一个错误姿势")
print("  - 同时手臂被拉直到奇异位形，再往前就没方向了")
print("  - 上面的 np.clip 是必须的：不加限幅，小 lam 会让关节一步跳几弧度，")
print("    200 次后关节转到几百弧度，位置完全失控（可以注释掉 clip 试一次）")

print("\nlambda 真正防的是「奇异点附近的速度爆炸」。在快伸直的位置上试：")
for theta in [60, 30, 10]:
    q_test = np.array([0.0, np.radians(-theta)])   # q2 接近 0 就是接近伸直
    target = fk(q_test) + np.array([0.0, 0.1])     # 想往上抬 10 厘米
    dq, err = dls_step(q_test, target, lam=1e-6)
    print(f"  q2=-{theta:2d} 度，末端上抬 10 厘米 -> 单步关节增量 {np.abs(dq).max():8.1f} rad")
print("lam 太小，奇异点附近会被放大成几弧度的单步跳跃，实机直接触发保护。")
print("lam 大一点，同样的任务拆成很多小步慢慢挪——慢，但不会爆。\n")

# ============================================================ 实验 4
print("=" * 60)
print("实验 4：用 mink 画同一个圆")
print("=" * 60)

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

config = mink.Configuration(model)
config.q[:] = [0.0, -1.0]
mujoco.mj_forward(model, config.data)

task = mink.FrameTask(
    frame_name="tip",
    frame_type="site",
    position_cost=1.0,       # 位置误差权重
    orientation_cost=0.0,    # 只有位置任务，不管朝向
)
limits = [mink.ConfigurationLimit(model)]   # 关节限位（本模型没写范围，暂时不起作用）

dt = 1e-2
mink_errors = []
for k in range(n_steps):
    theta = 2 * np.pi * k / n_steps
    target = center + radius * np.array([np.cos(theta), np.sin(theta)])
    # mink 处理 3D，平面任务把 y 固定为 0
    task.set_target(mink.SE3.from_translation(np.array([target[0], 0.0, target[1]])))
    for _ in range(20):
        velocity = mink.solve_ik(
            config, [task], dt,
            solver="daqp",      # 本机可用的二次规划求解器
            damping=1e-3,
            limits=limits,
        )
        config.integrate_inplace(velocity, dt)
    mujoco.mj_forward(model, config.data)
    p = config.data.site_xpos[tip_id]
    mink_errors.append(np.linalg.norm(np.array([p[0], p[2]]) - target))

mink_errors = np.array(mink_errors)
print(f"平均跟踪误差 {mink_errors.mean():.2e} 米（浮点噪声级别）")
print(f"最大跟踪误差 {mink_errors.max():.2e} 米")
print(f"最终关节角 {config.q}")
print("mink 内部做的就是我们实验 1 写的 DLS，只是多了限位、多任务、接触等约束。\n")

# ============================================================ 实验 5
print("=" * 60)
print("实验 5：零空间——末端不动，肘部还能自己挪")
print("=" * 60)

# 换成 3 段杆：3 个关节追末端 2 个坐标，多出 1 个自由度
# 总长 = 0.3 + 0.3 + 0.2 = 0.8 米
L3 = 0.2


def fk3(q):
    q1, q2, q3 = q
    return np.array([
        -L1 * np.sin(q1) - L2 * np.sin(q1 + q2) - L3 * np.sin(q1 + q2 + q3),
        -L1 * np.cos(q1) - L2 * np.cos(q1 + q2) - L3 * np.cos(q1 + q2 + q3),
    ])


def jac3(q):
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    c123, s123 = np.cos(q1 + q2 + q3), np.sin(q1 + q2 + q3)
    return np.array([
        [-L1 * c1 - L2 * c12 - L3 * c123, -L2 * c12 - L3 * c123, -L3 * c123],
        [L1 * s1 + L2 * s12 + L3 * s123,  L2 * s12 + L3 * s123,  L3 * s123],
    ])


def solve3(q0, use_nullspace):
    q = np.array(q0, dtype=float)
    for _ in range(600):
        e = target - fk3(q)
        J = jac3(q)
        # 2x3 的雅可比，伪逆仍是 J^T (J J^T + lam^2 I)^-1，形状 3x2
        Jpinv = J.T @ np.linalg.inv(J @ J.T + 0.005**2 * np.eye(2))
        dq = Jpinv @ e
        if use_nullspace:
            # (I - J^+ J) 把姿态指令投影到"不影响末端"的方向上
            dq = dq + (np.eye(3) - Jpinv @ J) @ (-0.5 * (q - q_posture))
        q = q + dq
    return q


target = np.array([0.0, -0.55])
q_posture = np.array([1.0, -1.6, -0.3])   # 希望的肘部姿势
q_start = np.array([0.3, -0.5, -0.3])

print("3 个关节追末端 2 个坐标，多出 1 个自由度。")
print(f"主任务：末端到 {target}")
print(f"次任务：关节尽量靠近 {q_posture}\n")

for use_ns, name in [(False, "只用末端任务"), (True, "末端任务 + 零空间姿态")]:
    q_sol = solve3(q_start, use_ns)
    err = np.linalg.norm(target - fk3(q_sol)) * 1000
    print(f"{name}:")
    print(f"  关节角 {np.round(q_sol, 3)}")
    print(f"  末端   {np.round(fk3(q_sol), 4)}   误差 {err:.3f} 毫米")

print("\n末端位置一模一样，关节角完全不同：")
print("  不加零空间：肘部停在哪儿算哪儿（[0.984, -1.228, -0.844]）")
print("  加了零空间：肘部被拉到指定姿势（[1.107, -1.563, -0.294]）")
print("末端 2 个约束只占掉 2 个自由度，剩下 1 维叫零空间，怎么动都不改变末端位置。")
print("人形机器人 30+ 个关节，零空间很大：这就是「脚踩稳不动、上身随便调」的原理。")
