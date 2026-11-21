"""检查 PyCharm 的 jdk.table.xml 里注册的解释器，哪些路径已经失效。

用法：
    python _check_pycharm_sdk.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

TABLE = Path.home() / "AppData/Roaming/JetBrains/PyCharm2026.1/options/jdk.table.xml"
HOME = Path.home().as_posix()

root = ET.parse(TABLE).getroot()
jdks = list(root.iter("jdk"))
bad = []

for i, jdk in enumerate(jdks, 1):
    name = hp = flav = ""
    for c in jdk:
        if c.tag == "name":
            name = c.get("value", "")
        elif c.tag == "homePath":
            hp = c.get("value", "")
        elif c.tag == "additional":
            for s in c.iter("setting"):
                if s.get("name") == "FLAVOR_ID":
                    flav = s.get("value", "")
    real = hp.replace("$USER_HOME$", HOME)
    ok = os.path.exists(real) if real else False
    if not ok:
        bad.append((i, name, hp, flav))
    print(f"{i:2d} {'OK ' if ok else '坏 '} {flav:24s} {name[:40]:40s} {hp}")

print()
print(f"共 {len(jdks)} 条，其中 {len(bad)} 条路径不存在")
for i, name, hp, flav in bad:
    print(f"  第 {i:2d} 条  {name}  ->  {hp}")
