#导入物理引擎
import mujoco
#xml="""...""",一个字符串，可以理解为机器人的装配手册
#<worldbody>是世界坐标系，
#<body>是机器人坐标系，
#<geom>是几何体，
#<freejoint/>是自由关节
xml = """
<mujoco>
    <worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 2">
          <freejoint/>
          <geom type="box" size=".2 .2 .2"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)  # pyright: ignore[reportAttributeAccessIssue]
data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]

for i in range(1000):
    mujoco.mj_step(model, data)  # pyright: ignore[reportAttributeAccessIssue]

print(data.qpos)