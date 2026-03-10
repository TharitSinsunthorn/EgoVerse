from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


class DummyARXInterface:
    DEFAULT_IMAGE_SHAPE = (480, 640, 3)

    def __init__(self, arms, dataset_path=None):
        self.arms = arms
        self.recorders = {}
        self._joint_positions = {
            arm: np.zeros(7, dtype=np.float64) for arm in ("left", "right")
        }
        self._ee_pose = {
            arm: np.zeros(7, dtype=np.float64) for arm in ("left", "right")
        }
        self.dataset_path = str(dataset_path) if dataset_path is not None else None
        self.dataset = None
        self.frame_idx = 0
        if self.dataset_path is not None:
            self.dataset = self._build_dataset(self.dataset_path)
            if len(self.dataset) == 0:
                raise ValueError(f"Offline dataset is empty: {self.dataset_path}")

    def _build_dataset(self, dataset_path):
        episode_path = Path(dataset_path)
        if not episode_path.exists():
            raise FileNotFoundError(f"Offline episode path not found: {dataset_path}")
        return ZarrDataset(
            episode_path,
            key_map=Eva.get_keymap(),
            transform_list=None,
        )

    def _create_controllers(self, cfg):
        return None

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _image_to_bgr_uint8(cls, image):
        image = cls._to_numpy(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] != 3:
            raise ValueError(f"Expected image channels-last RGB, got {image.shape}")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image[..., [2, 1, 0]]

    @classmethod
    def _pose_wxyz_to_xyzypr(cls, pose):
        pose = cls._to_numpy(pose).astype(np.float64, copy=False).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f"Expected xyz+quat(wxyz) pose, got shape {pose.shape}")
        quat_xyzw = pose[[4, 5, 6, 3]]
        ypr = R.from_quat(quat_xyzw).as_euler("ZYX", degrees=False)
        return np.concatenate([pose[:3], ypr], axis=-1)

    def _next_dataset_sample(self):
        if self.dataset is None:
            return None
        sample = self.dataset[self.frame_idx % len(self.dataset)]
        self.frame_idx += 1
        return sample

    def _obs_from_dataset_sample(self, sample):
        sample = {k: self._to_numpy(v) for k, v in sample.items()}
        left_pose = self._pose_wxyz_to_xyzypr(sample["left.obs_ee_pose"])
        right_pose = self._pose_wxyz_to_xyzypr(sample["right.obs_ee_pose"])
        left_gripper = float(np.asarray(sample["left.obs_gripper"]).reshape(-1)[0])
        right_gripper = float(np.asarray(sample["right.obs_gripper"]).reshape(-1)[0])

        ee_poses = np.zeros(14, dtype=np.float64)
        ee_poses[:6] = left_pose
        ee_poses[6] = left_gripper
        ee_poses[7:13] = right_pose
        ee_poses[13] = right_gripper

        obs = {
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
                [self.get_joints("left"), self.get_joints("right")], axis=0
            ),
            "ee_poses": ee_poses,
        }
        return obs

    def set_joints(self, desired_position, arm):
        desired_position = np.asarray(desired_position, dtype=np.float64)
        if desired_position.shape != (7,):
            raise ValueError(
                "For Eva, desired position must be of shape (7,) for single arm"
            )
        self._joint_positions[arm] = desired_position.copy()

    def set_pose(self, pose, arm):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(
                f"For Eva, target position must be of shape (7,), current shape: {pose.shape}"
            )
        self._ee_pose[arm] = pose.copy()
        joints = np.zeros(7, dtype=np.float64)
        joints[6] = pose[6]
        self._joint_positions[arm] = joints
        return joints

    def get_obs(self):
        sample = self._next_dataset_sample()
        if sample is not None:
            return self._obs_from_dataset_sample(sample)

        obs = {
            "front_img_1": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "left_wrist_img": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
            "right_wrist_img": np.zeros(self.DEFAULT_IMAGE_SHAPE, dtype=np.uint8),
        }
        joint_positions = np.concatenate(
            [self.get_joints("left"), self.get_joints("right")], axis=0
        )
        ee_poses = np.concatenate([self._ee_pose["left"], self._ee_pose["right"]])
        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses
        return obs

    def solve_ik(self, ee_pose, arm):
        ee_pose = np.asarray(ee_pose, dtype=np.float64)
        if ee_pose.shape != (6,):
            raise ValueError(
                "For Eva, target position must be of shape (6,) for single arm"
            )
        return np.zeros(6, dtype=np.float64)

    def get_joints(self, arm):
        return self._joint_positions[arm].copy()

    def get_pose(self, arm, se3=False):
        pose = self._ee_pose[arm]
        pos = pose[:3].copy()
        rot = R.from_euler("ZYX", pose[3:6], degrees=False)
        if se3:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot.as_matrix()
            T[:3, 3] = pos
            return T
        return pos, rot

    def set_home(self):
        for arm in ("left", "right"):
            self._joint_positions[arm] = np.zeros(7, dtype=np.float64)
            self._ee_pose[arm] = np.zeros(7, dtype=np.float64)
        self.frame_idx = 0
