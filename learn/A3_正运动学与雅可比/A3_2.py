import numpy as np
import mujoco

np.set_printoptions(precision=4,suppress=True) #打印保留四位小数

L1 = 0.3 #第一段杆长0.3米
L2 = 0.3 #第二段杆长0.3米

def one_link(q):
    """一根长L1的杆,绕y轴转q弧度,返回末端(x,z)"""
    x = -L1 * np.sin(q)
    z = -L1 * np.cos(q)
    return np.array([x,z])


def fk_hand(q1, q2):
    """两段杆的正运动学：关节角 -> 末端 (x, z)"""
    x = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    z = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    return np.array([x, z])


print(fk_hand(0.0, np.pi / 2))    # 预期 x=-0.3, z=-0.3


