"""
YAM hardware interface.

Operational notes (verified against i2rt source, 2026-04-21):
  * CAN must be up before instantiation — the i2rt MotorChainRobot constructor
    blocks waiting for the first motor state message.
  * Gripper calibration runs on first connect (linear_4310 has
    `needs_calibration=True`). The gripper will move to both mechanical
    stops (~4s total). Ensure the workspace is clear.
  * Default `zero_gravity_mode=True` means the arm floats after __init__.
    First `set_joints`/`set_pose` call engages PD control — snap may be
    abrupt if the target is far from the current pose. Call `set_home` with
    the arm's current joints to lock position before running a policy.
  * i2rt raises RuntimeError on every update loop if any arm joint is
    outside its limits (±0.1 rad buffer). The arm must start in a valid
    configuration; otherwise the motor chain shuts down immediately.

Gripper convention (confirmed via i2rt's JointMapper remapper):
  * get_joints()[6] returns gripper normalized to [0, 1] (0=closed, 1=open)
  * set_joints(...)[6] expects gripper normalized to [0, 1]
"""

from __future__ import annotations

import time

# from abc import ABC, abstractclassmethod
# from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.robot.yam.yam_kinematics import YAMMinkKinematicsSolver

try:
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
except ImportError:
    get_yam_robot = None
    ArmType = None
    GripperType = None


class YAMArm:
    """Single YAM arm on one CAN channel."""

    def __init__(
        self,
        channel: str,
        arm_type: ArmType = ArmType.YAM,
        gripper_type: GripperType = GripperType.LINEAR_4310,
        ik_solver: Optional[YAMMinkKinematicsSolver] = None,
        zero_gravity_mode: bool = True,
    ):
        self.channel = channel
        self.arm_type = arm_type
        self.gripper_type = gripper_type

        self.robot = get_yam_robot(
            channel=channel,
            arm_type=arm_type,
            gripper_type=gripper_type,
            zero_gravity_mode=zero_gravity_mode,
        )
        self.num_arm_joints = 6
        self.num_dofs = self.robot.num_dofs()  # 7 with gripper

        self.ik = ik_solver or YAMMinkKinematicsSolver(
            arm_variant=arm_type.name.lower(),
            gripper_variant=gripper_type.name.lower(),
        )

    # ----- joint-space -----

    def get_joints(self) -> np.ndarray:
        """Return (7,): 6 arm joints + gripper, in i2rt native units."""
        return np.asarray(self.robot.get_joint_pos(), dtype=np.float64)

    def set_joints(self, desired_position: np.ndarray) -> None:
        """Send a 7-DOF command. i2rt clips arm joints to joint_limits."""
        desired_position = np.asarray(desired_position, dtype=np.float64)
        if desired_position.shape != (self.num_dofs,):
            raise ValueError(
                f"Expected shape ({self.num_dofs},), got {desired_position.shape}"
            )
        self.robot.command_joint_pos(desired_position)

    # ----- cartesian-space (arm base frame) -----

    def get_pose(self, se3: bool = False):
        """FK on current arm joints.

        Returns (pos[3], scipy Rotation) when se3=False, or a 4x4 SE(3)
        matrix when se3=True. Matches ARXInterface.get_pose signature.
        """
        jnts = self.get_joints()[: self.num_arm_joints]
        pos, rot = self.ik.fk(jnts)
        pos = np.asarray(pos, dtype=np.float64)
        if se3:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            return T
        return pos, rot

    def get_pose_6d(self) -> np.ndarray:
        """[xyz, y, p, r] in arm base frame (6,)."""
        pos, rot = self.get_pose(se3=False)
        return np.concatenate([pos, rot.as_euler("ZYX", degrees=False)])

    def solve_ik(self, ee_pose: np.ndarray) -> np.ndarray:
        """Solve IK for a 6D base-frame pose [xyz, y, p, r] → 6 arm joints.

        Seeded from current joints; retries with perturbations on failure.
        Returns current arm joints if IK ultimately fails.
        """
        ee_pose = np.asarray(ee_pose, dtype=np.float64)
        if ee_pose.shape != (6,):
            raise ValueError(f"Expected shape (6,), got {ee_pose.shape}")
        rot_mat = R.from_euler("ZYX", ee_pose[3:6], degrees=False).as_matrix()
        cur_arm = self.get_joints()[: self.num_arm_joints]
        solved = self.ik.ik_with_retries(ee_pose[:3], rot_mat, cur_arm)
        if solved is None:
            print(f"[YAMArm {self.channel}] IK failed; holding arm pose")
            return cur_arm
        return solved

    def set_pose(self, pose: np.ndarray) -> np.ndarray:
        """Cartesian command: [xyz, y, p, r, gripper] in arm base frame.

        Returns the 7-DOF joint command actually sent.
        """
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"Expected shape (7,), got {pose.shape}")
        arm_joints = self.solve_ik(pose[:6])
        joints = np.concatenate([arm_joints, [pose[6]]])
        self.set_joints(joints)
        return joints

    # ----- utilities -----

    def set_home(
        self,
        arm_joints: np.ndarray,
        gripper: float,
        time_s: float = 3.0,
    ) -> None:
        """Smoothly interpolate to a home pose via i2rt's move_joints.

        `arm_joints` must be (6,) rad and `gripper` a scalar. There's no
        safe default — determine these per-setup (e.g. via i2rt's MuJoCo
        viewer at examples/control_with_mujoco, or by hand in zero-gravity
        mode and reading get_joints()).
        """
        target = np.concatenate([np.asarray(arm_joints, dtype=np.float64), [gripper]])
        if hasattr(self.robot, "move_joints"):
            self.robot.move_joints(target, time_interval_s=time_s)
        else:
            self.set_joints(target)
            time.sleep(time_s)

    def close(self) -> None:
        # DMChainCanInterface.close() sets running=False then immediately closes
        # the CAN socket, without joining its own thread. Pre-signal the thread
        # to stop so it has time to exit before the socket fd is invalidated.
        if hasattr(self.robot, "motor_chain") and hasattr(
            self.robot.motor_chain, "running"
        ):
            self.robot.motor_chain.running = False
            time.sleep(0.05)
        self.robot.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    # Read-only smoke test: connects to one arm, prints joints + FK.
    # No commands are sent; arm stays in zero-gravity (gravity-comp) mode.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can1")
    parser.add_argument(
        "--loop", action="store_true", help="Print joint + pose at 10 Hz until Ctrl-C."
    )
    args = parser.parse_args()

    arm = YAMArm(channel=args.channel)
    try:
        print(f"num_dofs : {arm.num_dofs}")
        if args.loop:
            print("Reading at 10 Hz — Ctrl-C to stop.\n")
            while True:
                j = arm.get_joints()
                p = arm.get_pose_6d()
                print(
                    f"joints  : {np.round(j, 4)}  |  pose_6d : {np.round(p, 4)}",
                    end="\r",
                )
                time.sleep(0.1)
        else:
            print(f"joints   : {arm.get_joints()}")
            print(f"pose_6d  : {arm.get_pose_6d()}")
    except KeyboardInterrupt:
        print()
    finally:
        arm.close()
