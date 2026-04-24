"""
YAM-specific kinematics solvers.
"""

import os
from typing import Optional

import numpy as np
from mink.exceptions import NoSolutionFound

from egomimic.robot.kinematics import MinkKinematicsSolver, TracKinematicsSolver

_I2RT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "i2rt")
)
YAM_URDF_PATH = os.path.join(_I2RT_ROOT, "i2rt/robot_models/arm/yam/yam.urdf")
YAM_XML_PATH = os.path.join(_I2RT_ROOT, "i2rt/robot_models/arm/yam/yam.xml")


def get_yam_combined_xml(
    arm_variant: str = "yam",
    gripper_variant: str = "linear_4310",
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
) -> str:
    """Produce the arm+gripper MJCF path using i2rt's combiner.

    Returns a path to a temp XML file with the TCP site available.
    """
    from i2rt.robots.utils import (
        ArmType,
        GripperType,
        combine_arm_and_gripper_xml,
    )

    arm_type = ArmType[arm_variant.upper()]
    gripper_type = GripperType[gripper_variant.upper()]
    return combine_arm_and_gripper_xml(
        arm_type,
        gripper_type,
        ee_mass=ee_mass,
        ee_inertia=ee_inertia,
    )


class YAMTracKinematicsSolver(TracKinematicsSolver):
    """
    TracIK solver for YAM. Uses the URDF; EEF defaults to link_6 (last
    actuated link). Switch eef_link_name to a gripper link if a URDF for the
    gripper is merged in.
    """

    def __init__(
        self,
        urdf_path: str = YAM_URDF_PATH,
    ):
        """
        Inintialize YAM TracIK solver.

        Args:
            urdf_path: Path to YAM's URDF file.
        """
        super().__init__(
            urdf_path=urdf_path,
            base_link_name="base_link",
            eef_link_name="link_6",
            num_joints=6,
        )
        self.urdf_path = urdf_path
        self.base_transform = None


class YAMMinkKinematicsSolver(MinkKinematicsSolver):
    """
    Yam-specific kinematics solver using Mink.

    Uses the combined arm+gripper XML so the EEF frame is the gripper TCP
    site, which is what the policy's Cartesian actions are expressed in.
    """

    # Joint names for YAM arm
    JOINT_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    # Conservative velocity limits (rad/s).
    DEFAULT_VELOCITY_LIMITS = {
        "joint1": 1.0,
        "joint2": 1.0,
        "joint3": 1.0,
        "joint4": 1.0,
        "joint5": 1.5,
        "joint6": 1.5,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        arm_variant: str = "yam",
        gripper_variant: str = "linear_4310",
        eef_link_name: Optional[str] = None,
        eef_frame_type: str = "site",
        velocity_limits: Optional[dict] = None,
        solver: str = "daqp",
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-3,
    ):
        """
        Args:
            model_path: Path to a pre-combined MuJoCo XML. If None, calls
                i2rt's combine_arm_and_gripper_xml() to produce one.
            arm_variant: "yam", "yam_pro", "yam_ultra", or "big_yam".
            gripper_variant: gripper name under i2rt/robot_models/gripper/.
            eef_link_name: EEF frame. If None, auto-selects to match i2rt
                convention: "tcp_site" for yam_teaching_handle, "grasp_site"
                otherwise. Grasp site is ~13.5cm forward of tcp and is the
                point actions should target for normal grippers.
            eef_frame_type: "site" (default) or "body".
        """
        if model_path is None:
            model_path = get_yam_combined_xml(arm_variant, gripper_variant)

        if eef_link_name is None:
            eef_link_name = (
                "tcp_site" if gripper_variant == "yam_teaching_handle" else "grasp_site"
            )

        if velocity_limits is None:
            velocity_limits = self.DEFAULT_VELOCITY_LIMITS.copy()

        super().__init__(
            model_path=model_path,
            base_link_name="world",
            eef_link_name=eef_link_name,
            num_joints=6,
            joint_names=self.JOINT_NAMES,
            eef_frame_type=eef_frame_type,
            velocity_limits=velocity_limits,
            solver=solver,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )

        self.model_path = model_path
        self.arm_variant = arm_variant
        self.gripper_variant = gripper_variant
        self.base_transform = None

    def set_base_transform(self, transform):
        """
        Set a base-to-world transform for mobile-base deployments.

         Args:
            transform: 4x4 homogeneous transform from the robot's base frame to the world frame.
        """
        self.base_transform = transform

    def ik_with_retries(self, pos_xyz, rot_mat, cur_jnts, num_retries=3, dt=0.1):
        """
        Solve IK with multiple retries and different random seeds on failure.

        Args:
            pos_xyz: Desired end-effector position (3,).
            rot_mat: Desired end-effector rotation as a 3x3 matrix.
            cur_jnts: Current joint values (6,).
            num_retries: Number of retries with random perturbations if IK fails.
            dt: Per-iteration integration step for Mink. Larger values allow
                more joint travel per call, which can help recover from large
                target jumps at the cost of potentially less precision.

        Returns:
            solved_jnts: Solution joint valuse or None if all retries fail.
        """

        def _try_ik(seed):
            try:
                return self.ik(pos_xyz, rot_mat, seed, dt=dt)
            except NoSolutionFound:
                return None

        # Loose acceptance: position within 3 cm, rotation-matrix Frobenius
        # error within 0.3 (~12°). Tight tolerances reject near-miss solutions
        # that are still useful for tracking a policy target.
        pos_thresh = 0.03
        rot_thresh = 0.3

        # First try with current configuration
        result = _try_ik(cur_jnts)

        # Verify the solution
        if result is not None:
            # Check if solution is valid by computing FK
            fk_pos, fk_rot = self.fk(result)
            pos_error = np.linalg.norm(fk_pos - pos_xyz)
            rot_error = np.linalg.norm(fk_rot.as_matrix() - rot_mat)

            if pos_error < pos_thresh and rot_error < rot_thresh:
                return result

        # Try with random perturbations
        for _ in range(num_retries):
            # Add small random perturbation to seed
            perturbed = cur_jnts + np.random.randn(self.num_joints) * 0.1
            result = _try_ik(perturbed)

            if result is not None:
                fk_pos, fk_rot = self.fk(result)
                pos_error = np.linalg.norm(fk_pos - pos_xyz)
                rot_error = np.linalg.norm(fk_rot.as_matrix() - rot_mat)

                if pos_error < pos_thresh and rot_error < rot_thresh:
                    return result

        # All retries failed
        return None
