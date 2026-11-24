import mujoco
#<worldbody>是世界坐标系，本身还是容器
#<body>是刚体，一块不会变形的物体，body是可以套body的
#<freejoint/>是自由关节
#自由关节的方块
xml_freejoint = """
<mujoco>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom type="box" size=".2 .2 .2"/>
      </body>
    </worldbody>
</mujoco>
"""

#铰链关节:只能绕一根轴转
#type="hinge" = 合页那种关节，只能绕一根轴转
#axis="0 0 1" = 绕的那根轴是"z 方向"（竖直的）
#name="h" = 给它起个名，叫 h（现在没用上，但以后要按名字找关节）
xml_hinge = """
<mujoco>
    <worldbody>
        <body pos="0 0 1">
            <joint name="h" type="hinge" axis="0 0 1"/>
            <geom type="box" size=".2 .2 .2"/>
        </body>
    </worldbody>
</mujoco>
"""

#mujoco.MjModel.from_xml_string(xml),照着这张图纸，造一个模型出来
# mujoco.MjData(model),给这个模型开一本状态记录本


for name, xml in [("自由关节", xml_freejoint), ("铰链关节", xml_hinge)]:
    model = mujoco.MjModel.from_xml_string(xml)  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    print(name, "→ 自由度 nv =", model.nv, "| qpos 长度 nq =", model.nq, "| qpos =", data.qpos)
