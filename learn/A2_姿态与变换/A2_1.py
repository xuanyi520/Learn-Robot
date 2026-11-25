"""A2 姿态与变换：旋转矩阵、欧拉角、四元数

运行：
    conda activate ai
    python code.py

本课不涉及物理仿真，纯数学。四个实验：
    实验 1  三种姿态表示互相转换，结果一致
    实验 2  万向锁：pitch = 90 度时 yaw 和 roll 分不开
    实验 3  四元数双覆盖：q 和 -q 是同一个旋转
    实验 4  四元数插值为什么不走直线
"""

import numpy as np
import mujoco

np.set_printoptions(precision=4, suppress=True)


# ============================================================ 基础工具
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_R_zyx(yaw, pitch, roll):
    """内旋 ZYX：先绕 z 转 yaw，再绕新 y 转 pitch，最后绕新 x 转 roll"""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def quat_from_axis_angle(axis, angle):
    """单位轴 + 角度 -> 四元数 [w, x, y, z]"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_mul(a, b):
    """四元数乘法，表示先 b 后 a 的复合旋转"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_R(q):
    """四元数 -> 旋转矩阵"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    """旋转矩阵 -> 四元数（用 MuJoCo 的实现，省得自己处理分支）"""
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(R).flatten())
    return out


def quat_slerp(q0, q1, t):
    """球面插值，t 在 [0, 1] 之间"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:                 # 走短弧
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-6:
        return q0
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


# ============================================================ 实验 1
print("=" * 60)
print("实验 1：三种表示互相转换")
print("=" * 60)

yaw, pitch, roll = np.radians([30, 20, 10])

R_euler = euler_to_R_zyx(yaw, pitch, roll)
q = R_to_quat(R_euler)
R_from_quat = quat_to_R(q)

print(f"欧拉角 (yaw, pitch, roll) = ({np.degrees(yaw):.1f}, "
      f"{np.degrees(pitch):.1f}, {np.degrees(roll):.1f}) 度")
print(f"四元数 q = {q}")
print(f"欧拉角转矩阵 和 四元数转矩阵 的最大差异："
      f"{np.abs(R_euler - R_from_quat).max():.2e}")

# 用 MuJoCo 的实现交叉验证
q_mj = np.zeros(4)
mujoco.mju_mat2Quat(q_mj, R_euler.flatten())
print(f"MuJoCo 算出的四元数 = {q_mj}")
print("结论：三种表示等价，只是坐标不同。\n")


# ============================================================ 实验 2
print("=" * 60)
print("实验 2：万向锁")
print("=" * 60)

# pitch = 90 度时，绕 y 轴转到头了（三个轴共面）
target = euler_to_R_zyx(0.5, np.pi / 2, 0.0)

best_err, best_roll = 1e9, None
for r in np.linspace(-np.pi, np.pi, 36001):
    R_test = euler_to_R_zyx(0.0, np.pi / 2, r)
    err = np.abs(target - R_test).max()
    if err < best_err:
        best_err, best_roll = err, r

print(f"目标：yaw=0.5 rad, pitch=90 度, roll=0")
print(f"找到一个只用 roll 的欧拉角也能得到同一个姿态：roll = {best_roll:.4f} rad "
      f"（{np.degrees(best_roll):.1f} 度）")
print(f"最大差异 {best_err:.2e}")
print("结论：pitch = 90 度时，yaw 和 roll 的作用合并了，"
      "一个自由度丢失。这就是万向锁。\n")


# ============================================================ 实验 3
print("=" * 60)
print("实验 3：四元数双覆盖")
print("=" * 60)

q1 = quat_from_axis_angle([0, 0, 1], np.radians(45))
q2 = -q1
print(f"q  = {q1}")
print(f"-q = {q2}")
print(f"两个四元数转出的矩阵最大差异：{np.abs(quat_to_R(q1) - quat_to_R(q2)).max():.2e}")
print("结论：q 和 -q 表示同一个旋转。比较四元数时不能直接比数值，"
      "要么比较矩阵，要么先统一符号。\n")


# ============================================================ 实验 4
print("=" * 60)
print("实验 4：插值")
print("=" * 60)

q_start = quat_from_axis_angle([1, 0, 0], 0.0)              # 不转
q_end = quat_from_axis_angle([0, 0, 1], np.radians(90))     # 绕 z 转 90 度

print("t     四元数插值的旋转角     直接线性插值的旋转角")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q_s = quat_slerp(q_start, q_end, t)
    angle_slerp = np.degrees(2 * np.arccos(np.clip(q_s[0], -1, 1)))

    q_lin = q_start * (1 - t) + q_end * t
    q_lin = q_lin / np.linalg.norm(q_lin)
    angle_lin = np.degrees(2 * np.arccos(np.clip(q_lin[0], -1, 1)))

    print(f"{t:4.2f}   {angle_slerp:8.2f} 度          {angle_lin:8.2f} 度")

print("结论：slerp 匀速转（每 1/4 走 22.5 度），线性插值中间快、两端慢。"
      "这就是为什么动画和状态插值都用 slerp。\n")
