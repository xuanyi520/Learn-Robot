import numpy as np
import mujoco

np.set_printoptions(precision=4, suppress=True)   # 打印保留 4 位小数

def Rx(a):
    """绕 x 轴转 a 弧度。x 轴不动，所以 1 在左上角。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def Ry(a):
    """绕 y 轴转 a 弧度。y 轴不动，所以 1 在正中间。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def Rz(a):
    """绕 z 轴转 a 弧度。z 轴不动，所以 1 在右下角。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def euler_to_R_zyx(yaw, pitch, roll):
    """内旋 ZYX：先绕 z 转 yaw，再绕新 y 转 pitch，最后绕新 x 转 roll"""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def quat_from_axis_angle(axis, angle):
    """轴 + 角度 -> 四元数 [w, x, y, z]。
    axis 是一根方向，angle 是绕它转的角度（弧度）。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)          # 归一化：只看方向，不看长度
    return np.concatenate([[np.cos(angle / 2)],        # 实部 w = cos(半角)
                           np.sin(angle / 2) * axis])  # 虚部 = sin(半角) × 轴


def quat_mul(a, b):
    """四元数乘法，表示"先做 b 的旋转，再做 a 的旋转"。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_R(q):
    """四元数 -> 旋转矩阵。公式是固定的，不用背。"""
    w, x, y, z = q / np.linalg.norm(q)          # 先归一化，保证是单位四元数
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    """旋转矩阵 -> 四元数。
    自己手写要按最大对角元分 4 种情况，容易错，直接借 MuJoCo 的实现。"""
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(R).flatten())
    return out

q1 = quat_from_axis_angle([0,0,1],np.radians(45))
q2 = -q1

print(f"q = {q1}")
print(f"-q = {q2}")
#这两个四元数转成矩阵，看差异多少
print(f"矩阵最大差异:{np.abs(quat_to_R(q1)-quat_to_R(q2)).max():.2e}")
def quat_slerp(q0, q1, t):
    """球面插值：t=0 返回 q0，t=1 返回 q1，中间匀速。"""
    q0 = q0 / np.linalg.norm(q0)          # 归一化
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)                  # 两个四元数的"相似度"
    if dot < 0:                           # 负值说明两个四元数在球的两边
        q1 = -q1                          # 翻符号，走短的那一边（双覆盖）
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)         # 防止数值误差超出范围
    theta = np.arccos(dot)                # 两个姿态之间的夹角
    if theta < 1e-6:                      # 几乎重合，直接返回
        return q0
    # 按球面弧长分配权重
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


q_start = quat_from_axis_angle([1, 0, 0], 0.0)            # 不转
q_end = quat_from_axis_angle([0, 0, 1], np.radians(90))   # 绕 z 转 90 度

print("t     四元数插值     直接线性插值")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q_s = quat_slerp(q_start, q_end, t)
    angle_slerp = np.degrees(2 * np.arccos(np.clip(q_s[0], -1, 1)))   # 四元数 -> 角度

    q_lin = q_start * (1 - t) + q_end * t
    q_lin = q_lin / np.linalg.norm(q_lin)
    angle_lin = np.degrees(2 * np.arccos(np.clip(q_lin[0], -1, 1)))

    print(f"{t:4.2f}   {angle_slerp:8.2f} 度    {angle_lin:8.2f} 度")
