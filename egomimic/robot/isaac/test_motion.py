"""
Simple motion test for Isaac Sim — no policy server needed.

Launches the sim, moves both arms through a short sequence:
  1. Home  (all zeros, gripper half-open)
  2. Lift up
  3. Reach forward
  4. Close gripper
  5. Open gripper
  6. Return home

Run:
    conda activate ego_sim
    python egomimic/robot/isaac/test_motion.py \
        --usd-path /home/tharit/eva_ws/EVA_room.usd
"""

import argparse
import os

# SimulationApp MUST be created before any other Isaac Sim imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

# ruff: noqa: E402
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction

# -----------------------------------------------------------------------
# Joint waypoints  [joint1..joint6 (rad), gripper (0=closed 1=open)]
# -----------------------------------------------------------------------

HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
LIFT = np.array([0.0, 0.6, 0.2, 0.0, 0.0, 0.0, 0.8])
REACH = np.array([0.0, 1.0, 0.6, 0.2, 0.0, 0.0, 0.8])
GRIP_CLOSE = np.array([0.0, 1.0, 0.6, 0.2, 0.0, 0.0, 0.0])
GRIP_OPEN = np.array([0.0, 1.0, 0.6, 0.2, 0.0, 0.0, 0.8])

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_M = 0.044  # metres — 0=closed, 0.044=open (both joint7 and joint8)
GRIPPER_CLOSE_M = 0.00


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def build_indices(robot):
    """Return (arm_indices [6], gripper_indices [2]) for this robot."""
    names = list(robot.dof_names)
    arm_idx = [names.index(n) for n in ARM_JOINT_NAMES]
    gripper_idx = [names.index(n) for n in GRIPPER_JOINT_NAMES]
    return arm_idx, gripper_idx


def apply_joints(robot, arm_idx, gripper_idx, joints_7):
    """
    Drive robot to joints_7 = [joint1..joint6 (rad), gripper (0-1 norm)].
    """
    gripper_width = GRIPPER_OPEN_M - GRIPPER_CLOSE_M
    gripper_m = float(joints_7[6]) * gripper_width + GRIPPER_CLOSE_M

    current = robot.get_joint_positions()
    if current is None:
        current = np.zeros(robot.num_dof, dtype=np.float64)
    target = current.copy()

    for i, idx in enumerate(arm_idx):
        target[idx] = joints_7[i]
    for idx in gripper_idx:
        target[idx] = gripper_m

    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=target)
    )


def hold(world, robot, arm_idx, gripper_idx, joints_7, steps=60):
    """Apply target and step for `steps` frames. Returns False if window closed."""
    for _ in range(steps):
        apply_joints(robot, arm_idx, gripper_idx, joints_7)
        world.step(render=True)
        if not simulation_app.is_running():
            return False
    return True


# -----------------------------------------------------------------------
# Motion sequence
# -----------------------------------------------------------------------

SEQUENCE = [
    ("Home", HOME, 120),
    ("Lift up", LIFT, 90),
    ("Reach fwd", REACH, 90),
    ("Close gripper", GRIP_CLOSE, 60),
    ("Open gripper", GRIP_OPEN, 60),
    ("Return home", HOME, 120),
]


def run_sequence(world, robots_info):
    for name, target, steps in SEQUENCE:
        print(f"  → {name}")
        for arm, (robot, arm_idx, gripper_idx) in robots_info.items():
            t = target.copy()
            if arm == "right":
                t[0] = -target[0]  # mirror base rotation for right arm
            ok = hold(world, robot, arm_idx, gripper_idx, t, steps=steps)
            if not ok:
                return False
    return True


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main(args):
    arms_list = ["left", "right"] if args.arms == "both" else [args.arms]
    arm_paths = {
        "left": args.left_arm_path,
        "right": args.right_arm_path,
    }

    print("[test_motion] Loading USD stage...")
    import omni.usd

    omni.usd.get_context().open_stage(
        str(args.usd_path)
    )  # preserves absolute prim paths

    world = World(stage_units_in_meters=1.0)

    robots_info = {}
    for arm in arms_list:
        robot = Robot(prim_path=arm_paths[arm], name=f"eva_{arm}")
        world.scene.add(robot)
        robots_info[arm] = robot

    world.reset()
    for robot in robots_info.values():
        robot.initialize()

    # warm-up frames
    for _ in range(10):
        world.step(render=True)

    # build DOF index maps
    indexed = {}
    for arm, robot in robots_info.items():
        arm_idx, gripper_idx = build_indices(robot)
        indexed[arm] = (robot, arm_idx, gripper_idx)
        print(
            f"[test_motion] {arm} arm DOFs — joints: {arm_idx}  gripper: {gripper_idx}"
        )

    print("[test_motion] Starting motion sequence. Close the window to quit.\n")

    try:
        while simulation_app.is_running():
            print("[test_motion] --- Running sequence ---")
            ok = run_sequence(world, indexed)
            if not ok:
                break

            if not args.loop:
                print("[test_motion] Done. Close the window to exit.")
                while simulation_app.is_running():
                    world.step(render=True)
                break

    except KeyboardInterrupt:
        print("\n[test_motion] Interrupted.")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        os._exit(0)


def parse_args():
    p = argparse.ArgumentParser(
        description="Isaac Sim motion test — no policy server needed."
    )
    p.add_argument("--usd-path", required=True)
    p.add_argument("--arms", default="both", choices=["left", "right", "both"])
    p.add_argument("--loop", action="store_true", help="Repeat the sequence in a loop.")
    p.add_argument("--scene-prim-path", default="/World")
    p.add_argument("--left-arm-path", default="/World/X5A_L")
    p.add_argument("--right-arm-path", default="/World/X5A_R")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
