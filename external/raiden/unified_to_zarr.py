"""
Convert a Raiden processed dataset to EgoVerse Zarr v3 format.

Run this from the EgoVerse Python environment (requires zarr==3.1.5, simplejpeg):

    python scripts/convert_to_egoverse.py \\
        --input ~/raiden/data/processed/cup_on_saucer \\
        --output ~/egoverse_datasets/cup_on_saucer

Field mapping
─────────────
Raiden lowdim                         EgoVerse key            Shape/frame
────────────────────────────────────  ──────────────────────  ───────────
joints[:6]       (obs arm joints)     left.obs_joints         (6,)
joints[6:7]      (obs gripper)        left.obs_gripper        (1,)
joints[7:13]                          right.obs_joints        (6,)
joints[13:14]                         right.obs_gripper       (1,)
action_joints[:6]  (cmd arm joints)   left.cmd_joints         (6,)
action_joints[6:7] (cmd gripper)      left.cmd_gripper        (1,)
action_joints[7:13]                   right.cmd_joints        (6,)
action_joints[13:14]                  right.cmd_gripper       (1,)
actual_poses[0:12] (obs EE, left)     left.obs_ee_pose        (7,) [x,y,z,qw,qx,qy,qz]
actual_poses[13:25] (obs EE, right)   right.obs_ee_pose       (7,)
action[0:12]   (cmd EE, left)         left.cmd_ee_pose        (7,)
action[13:25]  (cmd EE, right)        right.cmd_ee_pose       (7,)
rgb/left_wrist_camera/                images.left_wrist       JPEG bytes
rgb/right_wrist_camera/               images.right_wrist      JPEG bytes
rgb/scene_camera/                     images.front_1          JPEG bytes
language_prompt                       annotations             annotation_v1

The 13-dim EE representation per arm is [x, y, z, R00..R22, gripper], where
R is a 3×3 rotation matrix stored row-major. We extract [0:12] for pose
(ignoring the per-arm gripper at dim 12 in favour of joints[6] / action_joints[6]).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from egomimic.rldb.zarr.zarr_writer import ZarrWriter  # pyright: ignore[reportMissingImports]

# ── constants ────────────────────────────────────────────────────────────────

CAMERAS = {
    "left_wrist_camera": "images.left_wrist",
    "right_wrist_camera": "images.right_wrist",
    "scene_camera": "images.front_1",  # Eva/YAM embodiment hardcodes this key name
}
IMAGE_SHAPE = [480, 640, 3]
FPS = 30
EMBODIMENT = "yam_bimanual"
CHUNK_T = 100

# ── helpers ───────────────────────────────────────────────────────────────────


def _rotmat_to_quat_wxyz(poses_13: np.ndarray) -> np.ndarray:
    """Extract [x,y,z,qw,qx,qy,qz] from (T,13) arm pose arrays.

    Each row: [x, y, z, R00..R22, gripper]. Rotation matrix converted to
    quaternion; gripper (dim 12) is excluded here.
    """
    T = len(poses_13)
    out = np.zeros((T, 7), dtype=np.float32)
    out[:, :3] = poses_13[:, :3]
    xyzw = Rotation.from_matrix(poses_13[:, 3:12].reshape(T, 3, 3)).as_quat()
    out[:, 3] = xyzw[:, 3]   # qw
    out[:, 4:] = xyzw[:, :3]  # qx, qy, qz
    return out


# ── main converter ────────────────────────────────────────────────────────────


def convert_episode(episode_dir: Path, output_path: Path, task_description: str) -> None:
    pkl_files = sorted((episode_dir / "lowdim").glob("*.pkl"))
    if not pkl_files:
        print(f"  skip {episode_dir.name}: no lowdim files")
        return

    T = len(pkl_files)
    print(f"  {episode_dir.name}: {T} frames → {output_path.name}")

    # ── load lowdim ──────────────────────────────────────────────────────────
    joints_buf = np.zeros((T, 14), dtype=np.float32)
    action_joints_buf = np.zeros((T, 14), dtype=np.float32)
    actual_poses_buf = np.zeros((T, 26), dtype=np.float32)
    action_buf = np.zeros((T, 26), dtype=np.float32)
    language_prompt = task_description

    for i, pkl_path in enumerate(pkl_files):
        with open(pkl_path, "rb") as f:
            frame = pickle.load(f)
        joints_buf[i] = frame["joints"]
        action_joints_buf[i] = frame["action_joints"]
        actual_poses_buf[i] = frame["actual_poses"]
        action_buf[i] = frame["action"]
        if not language_prompt:
            language_prompt = str(frame.get("language_prompt", ""))

    # ── load images (read raw JPEG bytes, no decode/re-encode) ───────────────
    pre_encoded: dict[str, tuple[np.ndarray, list[int]]] = {}
    for cam_dir, zarr_key in CAMERAS.items():
        cam_path = episode_dir / "rgb" / cam_dir
        jpg_files = sorted(cam_path.glob("*.jpg"))
        if len(jpg_files) != T:
            print(f"    WARNING: {cam_dir} has {len(jpg_files)} frames, expected {T} — truncating")
        encoded = np.empty((T,), dtype=object)
        for i, p in enumerate(sorted(jpg_files)[:T]):
            encoded[i] = p.read_bytes()
        pre_encoded[zarr_key] = (encoded, IMAGE_SHAPE)

    # ── write via ZarrWriter ─────────────────────────────────────────────────
    writer = ZarrWriter(
        episode_path=output_path,
        embodiment=EMBODIMENT,
        fps=FPS,
        task_name=episode_dir.parent.name,
        task_description=language_prompt,
        annotations=[(language_prompt, 0, T - 1)],
        chunk_timesteps=CHUNK_T,
    )
    writer.write(
        metadata_override={"task": episode_dir.parent.name},
        numeric_data={
            "left.obs_joints":   joints_buf[:, :6],
            "left.obs_gripper":  joints_buf[:, 6:7],
            "left.obs_ee_pose":  _rotmat_to_quat_wxyz(actual_poses_buf[:, :13]),
            "left.cmd_joints":   action_joints_buf[:, :6],
            "left.cmd_gripper":  action_joints_buf[:, 6:7],
            "left.cmd_ee_pose":  _rotmat_to_quat_wxyz(action_buf[:, :13]),
            "right.obs_joints":  joints_buf[:, 7:13],
            "right.obs_gripper": joints_buf[:, 13:14],
            "right.obs_ee_pose": _rotmat_to_quat_wxyz(actual_poses_buf[:, 13:]),
            "right.cmd_joints":  action_joints_buf[:, 7:13],
            "right.cmd_gripper": action_joints_buf[:, 13:14],
            "right.cmd_ee_pose": _rotmat_to_quat_wxyz(action_buf[:, 13:]),
        },
        pre_encoded_image_data=pre_encoded,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Raiden processed dataset → EgoVerse Zarr v3"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Raiden processed task dir, e.g. data/processed/cup_on_saucer",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory; one <episode>.zarr per episode is written here",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="Episode IDs to convert (e.g. 0000 0003). Default: all",
    )
    args = parser.parse_args()

    input_dir: Path = args.input.expanduser().resolve()
    output_dir: Path = args.output.expanduser().resolve()

    # shared task description from metadata
    task_description = ""
    meta_path = input_dir / "metadata_shared.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        task_description = (
            meta.get("language", {}).get("prompt", "")
            or meta.get("task_description", "")
        )

    # episode list from split_all.json → "files" key
    split_path = input_dir / "split_all.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        all_episodes = sorted(split.get("files", {}).keys())
    else:
        all_episodes = sorted(
            p.name for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )

    episodes = args.episodes if args.episodes else all_episodes

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Episodes: {len(episodes)}\n")

    for ep_id in episodes:
        ep_dir = input_dir / f"{int(ep_id):04d}"
        if not ep_dir.exists():
            print(f"  skip {ep_id}: not found")
            continue
        convert_episode(ep_dir, output_dir / f"{ep_dir.name}.zarr", task_description)

    print(f"\nDone — {len(episodes)} episode(s) written to {output_dir}")


if __name__ == "__main__":
    main()
