"""
YAM bimanual robot interface — mirrors ARXInterface from EVA.

Wraps two YAMArm instances and three RealSense D405 cameras, and provides
get_obs() / set_joints() / set_pose() in the same format as ARXInterface
so it can be used directly with PolicyRollout.

Also provides OfflineYAMInterface for testing inference without hardware:
    ri = OfflineYAMInterface(arms=["left", "right"], dataset_path="path/to/episode.zarr")
    obs = ri.get_obs()   # returns obs from the recorded episode
    ri.set_pose(action, "left")  # no-op, just stores the command
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# YAMArm and i2rt types are imported lazily inside YAMInterface.__init__
# so that OfflineYAMInterface can be used without i2rt installed.

# RealSenseRecorder lives in the EVA directory; add it to the path once.
_EVA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "eva", "eva_ws", "src", "eva")
)
if _EVA_DIR not in sys.path:
    sys.path.insert(0, _EVA_DIR)

try:
    from stream_d405 import RealSenseRecorder, list_connected_serials
except ImportError:
    RealSenseRecorder = None
    list_connected_serials = None


class YAMInterface:
    """Bimanual YAM interface with cameras.

    Parameters
    ----------
    arms:
        List of arms to control, e.g. ["left", "right"] or ["right"].
    left_channel / right_channel:
        CAN interface names (e.g. "can0", "can1").
    front_cam_serial:
        Serial number of the front/egocentric RealSense D405.
    left_wrist_serial / right_wrist_serial:
        Serial numbers of the wrist cameras. Only created for arms in `arms`.
    """

    def __init__(
        self,
        arms: List[str],
        left_channel: str = "can0",
        right_channel: str = "can1",
        front_cam_serial: Optional[str] = None,
        left_wrist_serial: Optional[str] = None,
        right_wrist_serial: Optional[str] = None,
        arm_type=None,
        gripper_type=None,
    ):
        # Lazy imports — keep i2rt off the critical path for offline users.
        try:
            from i2rt.robots.utils import ArmType, GripperType

            from egomimic.robot.yam.yam_interface import YAMArm
        except ImportError as exc:
            raise ImportError(
                "i2rt is not installed. Run: pip install -e external/i2rt --no-deps"
            ) from exc

        if arm_type is None:
            arm_type = ArmType.YAM
        if gripper_type is None:
            gripper_type = GripperType.LINEAR_4310

        if RealSenseRecorder is None:
            raise ImportError(
                "pyrealsense2 is not installed. Run: pip install pyrealsense2"
            )

        self.arms = arms
        channels = {"left": left_channel, "right": right_channel}

        # Cameras before arms: RealSense USB enumeration holds the GIL for seconds,
        # which would starve the 250 Hz motor control threads and trip DM watchdogs.
        cam_specs = {"front_img_1": front_cam_serial}
        if "left" in arms:
            cam_specs["left_wrist_img"] = left_wrist_serial
        if "right" in arms:
            cam_specs["right_wrist_img"] = right_wrist_serial

        self._cameras: dict[str, RealSenseRecorder] = {
            key: RealSenseRecorder(serial)
            for key, serial in cam_specs.items()
            if serial is not None
        }

        self._arms: dict = {
            arm: YAMArm(
                channel=channels[arm],
                arm_type=arm_type,
                gripper_type=gripper_type,
            )
            for arm in arms
        }

    # ----- observations -----

    def get_obs(self) -> dict:
        """Return obs dict matching PolicyRollout's expected format.

        Keys: joint_positions [14], ee_poses [14],
              front_img_1, left_wrist_img, right_wrist_img (BGR uint8).
        """
        joint_positions = np.zeros(14, dtype=np.float64)
        ee_poses = np.zeros(14, dtype=np.float64)

        for arm in self.arms:
            off = 7 if arm == "right" else 0
            joints = self._arms[arm].get_joints()  # (7,)
            joint_positions[off : off + 7] = joints
            pos, rot = self._arms[arm].get_pose()
            ee_poses[off : off + 6] = np.concatenate(
                [pos, rot.as_euler("ZYX", degrees=False)]
            )
            ee_poses[off + 6] = joints[6]

        obs: dict = {
            "joint_positions": joint_positions,
            "ee_poses": ee_poses,
        }
        for key, cam in self._cameras.items():
            img = cam.get_image()
            if img is not None:
                obs[key] = img
        return obs

    # ----- per-arm control (matches ARXInterface signatures) -----

    def get_joints(self, arm: str) -> np.ndarray:
        return self._arms[arm].get_joints()

    def set_joints(self, desired_position: np.ndarray, arm: str) -> None:
        self._arms[arm].set_joints(desired_position)

    def set_pose(self, pose: np.ndarray, arm: str) -> np.ndarray:
        return self._arms[arm].set_pose(pose)

    def get_pose(self, arm: str, se3: bool = False):
        return self._arms[arm].get_pose(se3=se3)

    def get_pose_6d(self, arm: str) -> np.ndarray:
        return self._arms[arm].get_pose_6d()

    def solve_ik(self, ee_pose: np.ndarray, arm: str) -> np.ndarray:
        return self._arms[arm].solve_ik(ee_pose)

    # ----- utilities -----

    def set_home(
        self,
        arm_joints: Optional[dict] = None,
        gripper: float = 1.0,
        time_s: float = 3.0,
    ) -> None:
        """Move all arms to home.

        arm_joints: {"left": q6, "right": q6}. If None, locks each arm at
        its current configuration — safe first call before running a policy.
        """
        for arm in self.arms:
            q6 = (
                np.asarray(arm_joints[arm], dtype=np.float64)
                if arm_joints is not None
                else self._arms[arm].get_joints()[:6]
            )
            self._arms[arm].set_home(q6, gripper, time_s)

    def close(self) -> None:
        for arm_obj in self._arms.values():
            arm_obj.close()
        for cam in self._cameras.values():
            cam.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class OfflineYAMInterface:
    """Fake robot interface that replays a pre-recorded zarr episode.

    No hardware is touched.  Use this to test the full inference pipeline
    (policy load → obs processing → action output) before connecting to the
    real arms.

    Parameters
    ----------
    arms:
        List of arm names, e.g. ["left", "right"].
    dataset_path:
        Path to a zarr episode directory.  Can be an EVA episode recorded
        with the EVA embodiment keymap — the obs format is identical.
    embodiment_keymap:
        Keymap mode passed to the embodiment's get_keymap().  Defaults to
        "cartesian", matching PolicyRollout's expected obs keys.
    """

    DEFAULT_IMAGE_SHAPE = (480, 640, 3)

    def __init__(
        self,
        arms: List[str],
        dataset_path: Optional[str] = None,
        embodiment: str = "eva",
    ):
        self.arms = arms
        self._joint_positions = {
            arm: np.zeros(7, dtype=np.float64) for arm in ("left", "right")
        }
        self._ee_pose = {
            arm: np.zeros(7, dtype=np.float64) for arm in ("left", "right")
        }
        self.frame_idx = 0
        self.dataset = None

        if dataset_path is not None:
            self.dataset = self._build_dataset(dataset_path, embodiment)
            if len(self.dataset) == 0:
                raise ValueError(f"Offline dataset is empty: {dataset_path}")
            print(
                f"[OfflineYAMInterface] Loaded {len(self.dataset)} frames from {dataset_path}"
            )

    def _build_dataset(self, dataset_path: str, embodiment: str):
        from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

        if embodiment == "eva":
            from egomimic.rldb.embodiment.eva import Eva

            key_map = Eva.get_keymap(keymap_mode="cartesian")
        else:
            raise ValueError(f"Unknown embodiment '{embodiment}'. Use 'eva'.")

        episode_path = Path(dataset_path)
        if not episode_path.exists():
            raise FileNotFoundError(f"Offline episode not found: {dataset_path}")
        return ZarrDataset(episode_path, key_map=key_map, transform_list=None)

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _image_to_bgr_uint8(cls, image):
        image = cls._to_numpy(image)
        if image.ndim != 3:
            raise ValueError(f"Expected 3-dim image, got {image.shape}")
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image[..., [2, 1, 0]]  # RGB → BGR

    @classmethod
    def _pose_wxyz_to_xyzypr(cls, pose):
        pose = cls._to_numpy(pose).astype(np.float64, copy=False).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f"Expected xyz+quat(wxyz) pose, got {pose.shape}")
        quat_xyzw = pose[[4, 5, 6, 3]]
        ypr = R.from_quat(quat_xyzw).as_euler("ZYX", degrees=False)
        return np.concatenate([pose[:3], ypr], axis=-1)

    def _next_sample(self):
        if self.dataset is None:
            return None
        sample = self.dataset[self.frame_idx % len(self.dataset)]
        self.frame_idx += 1
        return sample

    def get_obs(self) -> dict:
        sample = self._next_sample()
        if sample is not None:
            sample = {k: self._to_numpy(v) for k, v in sample.items()}
            left_pose = self._pose_wxyz_to_xyzypr(sample["left.obs_ee_pose"])
            right_pose = self._pose_wxyz_to_xyzypr(sample["right.obs_ee_pose"])
            left_gripper = float(np.asarray(sample["left.obs_gripper"]).reshape(-1)[0])
            right_gripper = float(
                np.asarray(sample["right.obs_gripper"]).reshape(-1)[0]
            )

            ee_poses = np.zeros(14, dtype=np.float64)
            ee_poses[:6] = left_pose
            ee_poses[6] = left_gripper
            ee_poses[7:13] = right_pose
            ee_poses[13] = right_gripper

            return {
                "front_img_1": self._image_to_bgr_uint8(
                    sample["observations.images.front_img_1"]
                ),
                "left_wrist_img": self._image_to_bgr_uint8(
                    sample["observations.images.left_wrist_img"]
                ),
                "right_wrist_img": self._image_to_bgr_uint8(
                    sample["observations.images.right_wrist_img"]
                ),
                "joint_positions": np.concatenate(
                    [self.get_joints("left"), self.get_joints("right")]
                ),
                "ee_poses": ee_poses,
            }

        # No dataset — return zeros
        return {
            "front_img_1": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "left_wrist_img": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "right_wrist_img": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "joint_positions": np.concatenate(
                [self.get_joints("left"), self.get_joints("right")]
            ),
            "ee_poses": np.concatenate([self._ee_pose["left"], self._ee_pose["right"]]),
        }

    def get_joints(self, arm: str) -> np.ndarray:
        return self._joint_positions[arm].copy()

    def set_joints(self, desired_position: np.ndarray, arm: str) -> None:
        self._joint_positions[arm] = np.asarray(
            desired_position, dtype=np.float64
        ).copy()

    def set_pose(self, pose: np.ndarray, arm: str) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        self._ee_pose[arm] = pose.copy()
        joints = np.zeros(7, dtype=np.float64)
        joints[6] = pose[6]
        self._joint_positions[arm] = joints
        return joints

    def get_pose(self, arm: str, se3: bool = False):
        pose = self._ee_pose[arm]
        pos = pose[:3].copy()
        rot = R.from_euler("ZYX", pose[3:6], degrees=False)
        if se3:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            return T
        return pos, rot

    def solve_ik(self, ee_pose: np.ndarray, arm: str) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    def set_home(
        self, arm_joints=None, gripper: float = 1.0, time_s: float = 3.0
    ) -> None:
        for arm in ("left", "right"):
            self._joint_positions[arm] = np.zeros(7, dtype=np.float64)
            self._ee_pose[arm] = np.zeros(7, dtype=np.float64)
        self.frame_idx = 0
        print("[OfflineYAMInterface] set_home: state reset, frame_idx=0")

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    # List connected RealSense serials — useful for finding camera serial numbers.
    if list_connected_serials is None:
        print("pyrealsense2 not installed.")
    else:
        serials = list_connected_serials()
        print(f"Connected RealSense devices ({len(serials)}):")
        for s in serials:
            print(f"  {s}")
