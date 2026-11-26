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

# 目标姿态：偏航 0.5 弧度，俯仰 90 度，翻滚 0
target = euler_to_R_zyx(0.5, np.pi / 2, 0.0)

best_err, best_roll = 1e9, None
for r in np.linspace(-np.pi, np.pi, 36001):       # 从 -180 度扫到 180 度，共 36001 个候选
    R_test = euler_to_R_zyx(0.0, np.pi / 2, r)    # 只改 roll，yaw 固定为 0
    err = np.abs(target - R_test).max()           # 和目标的差异
    if err < best_err:                            # 记下差异最小的一次
        best_err, best_roll = err, r

print(f"找到的等价角度：roll = {best_roll:.4f} rad （{np.degrees(best_roll):.1f} 度）")
print(f"最大差异 {best_err:.2e}")