import numpy as np
import mujoco

np.set_printoptions(precision=4,suppress=True)

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def euler_to_R_zyx(yaw,pitch,roll):
    """内旋 ZYX: 先绕 Z 转 yaw,再绕新 Y 转 pitch,最后绕新X转roll"""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

for deg in (0, 30, 45, 60, 90, 180, 270):
    a = np.radians(deg)
    p = Rz(a) @ np.array([1.0, 0.0, 0.0])
    print(f"{deg:4d}度  c={c if False else np.cos(a):6.3f}  s={np.sin(a):6.3f}  "
          f"x轴落到 ({p[0]:6.3f}, {p[1]:6.3f})")