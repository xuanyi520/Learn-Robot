import numpy as np

L1 = 0.3
L2 = 0.3

def fk_hand(q1,q2):
    """两段杆的正运动学：关节角 -> 末端 (x, z)。"""
    x = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    z = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    return np.array([x, z])

print(fk_hand(0.0,np.pi / 2)) 