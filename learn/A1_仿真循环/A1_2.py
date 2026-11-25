import mujoco

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

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

for i in range(500):
    mujoco.mj_step(model, data)
    if i % 100 == 0:
        print(f"第{i+1:3d}步,高度{data.qpos[2]:.4f}米")