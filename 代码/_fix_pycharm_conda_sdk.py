"""清掉 PyCharm 里 FLAVOR_DATA 为空的 Conda 解释器记录。

背景：手工往 jdk.table.xml 里加 conda 解释器时如果 FLAVOR_DATA 填 {}，
PyCharm 2026.1 读取时会抛 PyCondaEnv.getFullCondaPathOnTarget() NPE，
连带把 Jupyter 内核启动也弄挂。

用法（必须先完全退出 PyCharm）：
    python _fix_pycharm_conda_sdk.py           # 只检查，不修改
    python _fix_pycharm_conda_sdk.py --apply   # 备份并删除
"""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TABLE = Path.home() / "AppData/Roaming/JetBrains/PyCharm2026.1/options/jdk.table.xml"
BACKUP = TABLE.with_suffix(".xml.bak2")


def collect_bad(root):
    """返回 [(元素, 名称, homePath)]，条件是 Conda 且 FLAVOR_DATA 为空"""
    bad = []
    for jdk in root.iter("jdk"):
        name = hp = ""
        flav_id = flav_data = None
        for c in jdk:
            if c.tag == "name":
                name = c.get("value", "")
            elif c.tag == "homePath":
                hp = c.get("value", "")
            elif c.tag == "additional":
                for s in c.iter("setting"):
                    k = s.get("name")
                    if k == "FLAVOR_ID":
                        flav_id = s.get("value")
                    elif k == "FLAVOR_DATA":
                        flav_data = s.get("value")
        if flav_id == "Conda" and (flav_data or "").strip() in ("", "{}"):
            bad.append((jdk, name, hp))
    return bad


def main():
    apply = "--apply" in sys.argv
    tree = ET.parse(TABLE)
    root = tree.getroot()

    total = len(list(root.iter("jdk")))
    bad = collect_bad(root)

    print(f"当前共 {total} 条解释器记录，其中损坏 {len(bad)} 条：")
    for _, name, hp in bad:
        print(f"  - {name}  ->  {hp}")

    if not bad:
        print("没有需要清理的记录。")
        return

    if not apply:
        print("\n这是检查模式，没有改动。加 --apply 才执行删除。")
        return

    shutil.copy2(TABLE, BACKUP)
    print(f"\n已备份到 {BACKUP}")

    for jdk, _, _ in bad:
        for parent in root.iter():
            if jdk in list(parent):
                parent.remove(jdk)
                break

    ET.indent(tree, space="  ")
    tree.write(TABLE, encoding="utf-8", xml_declaration=False)
    print(f"已删除 {len(bad)} 条，剩余 {len(list(root.iter('jdk')))} 条")


if __name__ == "__main__":
    main()
