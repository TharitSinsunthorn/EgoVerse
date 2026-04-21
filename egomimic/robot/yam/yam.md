# YAM / EgoVerse Setup

## 1. Python environment

All EgoVerse commands run inside the `emimic` venv:

```bash
source emimic/bin/activate
```

### 1.1. Install i2rt and CAN dependencies (one-time)

```bash
source emimic/bin/activate
pip install python-can==4.5.0
pip install -e external/i2rt --no-deps   # --no-deps skips the numpy==2.2.6 pin (emimic already has 2.4.x)
```

---

## 2. One-time host setup

### 2.1. Bring up CAN interfaces

YAM uses native CAN (not USB-serial). Bring up both interfaces at 1 Mbit/s:

```bash
sudo ip link set can0 up type can bitrate 1000000   # left arm
sudo ip link set can1 up type can bitrate 1000000   # right arm
```

To reset unresponsive adapters, use the script provided by i2rt:

```bash
sudo sh external/i2rt/scripts/reset_all_can.sh
```

Check that interfaces are up:

```bash
ip link show | grep can
```

---

## 3. Operational notes

These apply every time you connect to hardware (verified against i2rt source):

- **CAN must be up before instantiation.** The i2rt `MotorChainRobot` constructor blocks waiting for the first motor state message. Run `can-yam` before starting any Python script.

- **Gripper calibration runs on first connect.** `LINEAR_4310` has `needs_calibration=True` — the gripper will sweep to both mechanical stops (~4 s total). Keep the workspace clear.

- **`zero_gravity_mode=True` by default.** After `__init__` the arm floats under gravity compensation. The first `set_joints` / `set_pose` call engages PD control — if the target is far from the current pose the snap can be abrupt. Always call `set_home(current_joints, current_gripper)` first to lock in position before running a policy.

- **Joint limits are enforced at ±0.1 rad buffer.** i2rt raises `RuntimeError` every update cycle if any arm joint is outside its limits. The arm must start in a valid configuration; otherwise the motor chain shuts down immediately.

### Gripper convention

| Direction | `get_joints()[6]` | `set_joints(...)[6]` |
|-----------|-------------------|----------------------|
| Closed    | 0.0               | 0.0                  |
| Open      | 1.0               | 1.0                  |

Normalization is handled internally by i2rt's `JointMapper` remapper — you never need to convert to/from raw motor units.

---

## 4. Hardware smoke test (read-only, no commands sent)

Bring up CAN, then:

```bash
source emimic/bin/activate
python egomimic/robot/yam/yam_interface.py --channel can1
```

Expected output:

```
num_dofs : 7
joints   : [ ... 6 arm joints in rad ... gripper 0..1 ]
pose_6d  : [ x  y  z  yaw  pitch  roll ]
```

The arm stays in gravity-compensation (zero-torque) mode throughout.

---

## 5. Finding the home pose

There is no built-in home pose — it depends on your workspace setup. To find one:

1. With the arm in zero-gravity mode, physically guide it to a neutral pose.
2. Read the current joints:
   ```bash
   python egomimic/robot/yam/yam_interface.py --channel can1
   ```
3. Note the printed `joints` array — that is your `arm_joints` (first 6 values) and `gripper` (7th value) for `set_home()`.

Alternatively, use i2rt's MuJoCo viewer to find a pose in simulation:

```bash
cd external/i2rt
python examples/control_with_mujoco/main.py
```

---

## 6. Uploading demos to AWS

```bash
source emimic/bin/activate
python egomimic/scripts/data_download/sync_s3.py --local-dir <demo_dir> --filters <filter>
```
