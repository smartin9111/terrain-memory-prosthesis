from assist_sim import load_combined
import mujoco
import mujoco.viewer
import time

# Human model + powered ankle prosthesis
model, data = load_combined(
    "myolegs22",
    "OpenSourceLeg_A_L1"
)

# Open interactive MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)