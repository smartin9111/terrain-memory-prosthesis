from pathlib import Path

import mujoco
from myoassist_utils.env_spec import EnvSpec


workspace = Path(__file__).resolve().parents[2]
spec_path = (
    workspace
    / "myoassist"
    / "docs"
    / "examples"
    / "env_prosthesis_rough.json"
)

print("Environment:", spec_path)

spec = EnvSpec.load(spec_path)
spec.validate()

xml = spec.compose()
model = mujoco.MjModel.from_xml_string(xml)

print()
print("=== MODEL ===")
print("nq:", model.nq)
print("nv:", model.nv)
print("nu (number of actions/actuators):", model.nu)

print()
print("=== ACTUATORS ===")

for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i,
    )

    marker = ""
    if name == "osl_ankle_torque_actuator":
        marker = "  <--- PROSTHESIS"

    print(
        f"{i:2d}: {name} "
        f"ctrlrange={model.actuator_ctrlrange[i]}{marker}"
    )

print()
print("=== OSL PROSTHESIS ===")

actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "osl_ankle_torque_actuator",
)

print("Action index:", actuator_id)

joint_id = int(model.actuator_trnid[actuator_id, 0])

joint_name = mujoco.mj_id2name(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    joint_id,
)

print("Controlled joint:", joint_name)
print("Control range:", model.actuator_ctrlrange[actuator_id])
