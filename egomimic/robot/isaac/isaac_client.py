"""
Isaac Sim 5.1 client for ZMQ policy rollout.

Runs in the ego_sim conda env. Drives the physics simulation,
renders camera observations, and communicates with policy_server.py
(running in emimic venv) over a ZMQ socket.

--- How to run (start policy_server.py FIRST, then this) ---

Terminal 1 (emimic venv):
    source /home/tharit/EgoVerse/emimic/bin/activate
    python egomimic/robot/isaac/policy_server.py \\
        --policy-path logs/test/test_2026-04-10_20-39-35/checkpoints/last.ckpt \\
        --arms both --cartesian

Terminal 2 (ego_sim env):
    conda activate ego_sim
    python egomimic/robot/isaac/isaac_client.py \\
        --usd-path /home/tharit/eva_ws/EVA_room.usd \\
        --arms both --cartesian

--- First run: check joint names ---
    python egomimic/robot/isaac/isaac_client.py \\
        --usd-path /home/tharit/eva_ws/EVA_room.usd --print-dofs

--- USD structure (discovered) ---
  /World/X5A_L   left arm  articulation  — DOFs: joint1..joint6, joint7, joint8
  /World/X5A_R   right arm articulation  — DOFs: joint1..joint6, joint7, joint8
  joint1-joint6 = arm joints
  joint7+joint8 = two gripper fingers (commanded together)
  /World/Eva/egocentric/ego_cam                      front/ego camera (USD path /World/egocentric/ego_cam, prefixed by our /World/Eva reference)
  /World/X5A_L/cam_bracket/cam_wrist/Camera       left wrist camera
  /World/X5A_R/cam_bracket/cam_wrist/Camera       right wrist camera
"""

import argparse
import os
import pickle

# SimulationApp MUST be the very first Isaac Sim import
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

# All other imports after SimulationApp
# ruff: noqa: E402
import numpy as np
import zmq
from omni.isaac.core import World
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------
# USD transform helper
# -----------------------------------------------------------------------


def _get_world_pose_se3(prim_path: str) -> np.ndarray:
    """Return 4x4 SE3 transform (column-vector convention) for a prim path.

    Uses Isaac Sim's XFormPrim.get_world_pose() which returns a clean
    [w,x,y,z] quaternion — avoids the GfMatrix4d numpy conversion ambiguity.
    """
    from omni.isaac.core.prims import XFormPrim

    xform = XFormPrim(prim_path=prim_path)
    pos, quat_wxyz = xform.get_world_pose()  # pos=[x,y,z], quat=[w,x,y,z]
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pos
    # scipy expects [x,y,z,w]; Isaac Sim returns [w,x,y,z]
    T[:3, :3] = R.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    return T


# -----------------------------------------------------------------------
# SimRobot  — two separate articulations, one per arm
# -----------------------------------------------------------------------


class SimRobot:
    """
    Minimal Isaac Sim wrapper for the EVA bimanual robot.

    The EVA USD uses two separate physics articulations:
      /World/X5A_L  (left arm)
      /World/X5A_R  (right arm)
    Each has 8 DOFs:  joint1..joint6 (arm) + joint7, joint8 (gripper fingers).

    No EgoVerse imports — only Isaac Sim + numpy + scipy.
    """

    GRIPPER_OPEN = 0.044  # metres — confirmed prismatic range: 0=closed, 0.044=open
    GRIPPER_CLOSE = 0.00

    # DOF names inside each arm's articulation (same for both)
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    GRIPPER_JOINT_NAMES = ["joint7", "joint8"]  # two fingers, commanded together

    def __init__(
        self,
        arms: list,
        world: World,
        scene_prim_path: str = "/World",  # not used for loading; kept for compat
        left_arm_path: str = "/World/X5A_L",  # left articulation root
        right_arm_path: str = "/World/X5A_R",  # right articulation root
        front_cam_path: str = "/World/egocentric/ego_cam",
        left_cam_path: str = "/World/X5A_L/cam_bracket/cam_wrist/Camera",
        right_cam_path: str = "/World/X5A_R/cam_bracket/cam_wrist/Camera",
        camera_resolution: tuple = (640, 480),
        skip_cameras: bool = False,
    ):
        from omni.isaac.core.robots import Robot
        from omni.isaac.sensor import Camera

        self.arms = arms
        self.scene_prim_path = scene_prim_path
        self._world = world
        self._gripper_width = self.GRIPPER_OPEN - self.GRIPPER_CLOSE
        self._arm_paths = {"left": left_arm_path, "right": right_arm_path}

        # ---- One Robot per arm ----
        self._robots = {}
        for arm in arms:
            path = self._arm_paths[arm]
            robot = Robot(prim_path=path, name=f"eva_{arm}")
            self._world.scene.add(robot)
            self._robots[arm] = robot

        # ---- Cameras (skipped for --print-dofs to avoid crash) ----
        self._cameras = {}
        self._cam_paths = {}  # key -> prim path string (for extrinsics)
        if not skip_cameras:
            import omni.usd

            stage = omni.usd.get_context().get_stage()

            W, H = camera_resolution
            cam_specs = [("front_img_1", front_cam_path)]
            if "left" in arms:
                cam_specs.append(("left_wrist_img", left_cam_path))
            if "right" in arms:
                cam_specs.append(("right_wrist_img", right_cam_path))

            for key, path in cam_specs:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise ValueError(
                        f"\n[SimRobot] Camera prim not found: '{path}'\n"
                        f"Run --print-cameras to list all Camera prims in the USD.\n"
                        f"Then override with e.g. --front-cam-path /correct/path"
                    )
                self._cameras[key] = Camera(prim_path=path, resolution=(W, H))
                self._cam_paths[key] = path

        # ---- Initialize ----
        self._world.reset()
        for robot in self._robots.values():
            robot.initialize()
        for cam in self._cameras.values():
            cam.initialize()

        if not skip_cameras:
            for _ in range(5):
                self._world.step(render=True)

        # ---- Build DOF index maps (per arm) ----
        if not skip_cameras:
            self._arm_indices = self._build_arm_indices()
            self._gripper_indices = self._build_gripper_indices()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_dof_names(self):
        print()
        for arm, robot in self._robots.items():
            names = list(robot.dof_names)
            print(
                f"[SimRobot] {arm.upper()} arm  ({self._arm_paths[arm]})  — {len(names)} DOFs:"
            )
            for i, n in enumerate(names):
                print(f"  [{i:2d}]  {n}")
        print(f"\nExpected arm joints   : {self.ARM_JOINT_NAMES}")
        print(f"Expected gripper DOFs : {self.GRIPPER_JOINT_NAMES}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_obs(self) -> dict:
        """
        Return obs dict for policy_server:
          front_img_1, left_wrist_img, right_wrist_img — BGR uint8 [H,W,3]
          joint_positions  — float64 [14]  (left 7, right 7)
          ee_poses         — float64 [14]  (left xyz+ypr+gripper, right ...)
        """
        obs = {key: self._render_bgr(cam) for key, cam in self._cameras.items()}

        joint_positions = np.zeros(14, dtype=np.float64)
        ee_poses = np.zeros(14, dtype=np.float64)

        for arm in self.arms:
            off = 7 if arm == "right" else 0
            jnts = self._get_joints(arm)  # [6 joints + 1 gripper]
            joint_positions[off : off + 7] = jnts

            pos, rot = self._fk(arm)
            ee_poses[off : off + 6] = np.concatenate([pos, rot.as_euler("ZYX")])
            ee_poses[off + 6] = jnts[6]

        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses
        return obs

    def apply_joints(self, joints_7: np.ndarray, arm: str):
        """
        Drive one arm to joint targets.
        joints_7: [joint1..joint6 (rad), gripper (0=close, 1=open)]
        """
        from omni.isaac.core.utils.types import ArticulationAction

        robot = self._robots[arm]
        joints_6 = joints_7[:6].astype(np.float64)
        gripper_m = float(joints_7[6]) * self._gripper_width + self.GRIPPER_CLOSE

        current = robot.get_joint_positions()
        if current is None:
            current = np.zeros(robot.num_dof, dtype=np.float64)
        target = current.copy()

        # Set arm joints
        for i, idx in enumerate(self._arm_indices[arm]):
            target[idx] = joints_6[i]

        for idx in self._gripper_indices[arm]:
            target[idx] = gripper_m

        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=target)
        )

    def set_home(self):
        """Reset all arms to zero joints, gripper half-open."""
        for arm in self.arms:
            home = np.zeros(7, dtype=np.float64)
            home[6] = 0.5
            self.apply_joints(home, arm)
        for _ in range(60):
            self._world.step(render=True)

    def step(self, render: bool = True):
        self._world.step(render=render)

    def align_egocam_to_hardware(self):
        """
        Reposition ego_cam to match x5Dec13_2 hardware camera calibration.

        x5Dec13_2 left arm: camera-to-left-base transform.
          - position in left base frame: [-0.044, -0.232, 0.573] m
          - orientation: looking forward+downward at workspace

        This ensures the sim camera viewpoint matches training data, so the
        policy receives visual observations consistent with what it was trained on.
        """
        from omni.isaac.core.prims import XFormPrim

        # x5Dec13_2 left arm extrinsics (from egomimicUtils.py)
        # Convention: base_pt = T @ cam_pt
        T_leftbase_from_cam = np.array(
            [
                [0.013, -0.718, 0.696, -0.044],
                [-0.999, -0.027, -0.009, -0.232],
                [0.025, -0.696, -0.718, 0.573],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Get left arm base world pose from USD
        left_base_path = self._arm_paths["left"] + "/X5A/base_link"
        T_world_leftbase = _get_world_pose_se3(left_base_path)

        # Camera world pose = T_world_leftbase @ T_leftbase_from_cam
        T_world_cam = T_world_leftbase @ T_leftbase_from_cam

        pos = T_world_cam[:3, 3]
        qxyzw = R.from_matrix(T_world_cam[:3, :3]).as_quat()  # [x,y,z,w]
        qwxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])

        ego_cam_path = self._cam_paths.get("front_img_1", "/Worlds/egocentric/ego_cam")
        XFormPrim(prim_path=ego_cam_path).set_world_pose(
            position=pos, orientation=qwxyz
        )

        print(f"[SimRobot] ego_cam aligned to x5Dec13_2  world_pos={np.round(pos, 3)}")

    def compute_extrinsics(self) -> dict:
        """
        Compute front-camera-to-base 4x4 transforms from actual USD prim poses.
        Returns {"left": 4x4, "right": 4x4}.

        Convention: base_pt = T @ cam_pt  (matches EgoVerse EXTRINSICS).

        Uses the FRONT/EGO camera (not wrist cameras) relative to each arm's
        base_link — this matches x5Dec13_2 which was calibrated from the front
        camera.  Both arms get the same camera, but expressed in their own
        base_link frame.
        """
        front_cam_path = self._cam_paths["front_img_1"]
        T_world_cam_gl = _get_world_pose_se3(front_cam_path)

        # Isaac Sim camera prims use OpenGL convention (-Z forward, Y up).
        # EgoVerse / x5Dec13_2 use OpenCV convention (+Z forward, Y down).
        # Apply a 180° rotation around X to convert OpenGL → OpenCV camera frame.
        R_gl_to_cv = np.eye(4, dtype=np.float64)
        R_gl_to_cv[1, 1] = -1.0  # flip Y
        R_gl_to_cv[2, 2] = -1.0  # flip Z  → camera now looks in +Z (OpenCV)
        T_world_cam = T_world_cam_gl @ R_gl_to_cv

        ext = {}
        for arm in self.arms:
            base_path = self._arm_paths[arm] + "/X5A/base_link"
            T_world_base = _get_world_pose_se3(base_path)
            ext[arm] = np.linalg.inv(T_world_base) @ T_world_cam

        print("[SimRobot] Computed extrinsics from sim (front cam → arm base):")
        for arm, T in ext.items():
            print(f"  {arm}:\n{np.round(T, 4)}")
        return ext

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_arm_indices(self) -> dict:
        """Map ARM_JOINT_NAMES to DOF indices for each arm's articulation."""
        indices = {}
        for arm, robot in self._robots.items():
            dof_names = list(robot.dof_names)
            idx = []
            for jname in self.ARM_JOINT_NAMES:
                if jname not in dof_names:
                    raise ValueError(
                        f"Arm joint '{jname}' not found in {arm} arm.\n"
                        f"Available: {dof_names}\n"
                        f"Run --print-dofs to inspect."
                    )
                idx.append(dof_names.index(jname))
            indices[arm] = idx
        return indices

    def _build_gripper_indices(self) -> dict:
        """Map GRIPPER_JOINT_NAMES to DOF indices for each arm's articulation."""
        indices = {}
        for arm, robot in self._robots.items():
            dof_names = list(robot.dof_names)
            idx = []
            for jname in self.GRIPPER_JOINT_NAMES:
                if jname not in dof_names:
                    raise ValueError(
                        f"Gripper joint '{jname}' not found in {arm} arm.\n"
                        f"Available: {dof_names}"
                    )
                idx.append(dof_names.index(jname))
            indices[arm] = idx
        return indices

    def _get_joints(self, arm: str) -> np.ndarray:
        """Return [6 arm joints (rad), gripper normalized (0-1)]."""
        robot = self._robots[arm]
        all_pos = robot.get_joint_positions()
        if all_pos is None:
            return np.zeros(7, dtype=np.float64)

        joints_6 = np.array(
            [all_pos[i] for i in self._arm_indices[arm]], dtype=np.float64
        )
        # Average two gripper finger positions → single normalized value
        gripper_m = float(np.mean([all_pos[i] for i in self._gripper_indices[arm]]))
        gripper_n = np.clip(
            (gripper_m - self.GRIPPER_CLOSE) / self._gripper_width, 0.0, 1.0
        )
        return np.concatenate([joints_6, [gripper_n]])

    def _fk(self, arm: str):
        """
        Compute EE pose (position, rotation) from USD prim world transforms.
        End-effector = link6 (wrist, after last revolute joint).
        Returns (pos [3], R) in the arm's base_link frame.
        """
        base_path = self._arm_paths[arm] + "/X5A/base_link"
        ee_path = self._arm_paths[arm] + "/X5A/link6"
        T_world_base = _get_world_pose_se3(base_path)
        T_world_ee = _get_world_pose_se3(ee_path)
        T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee
        return T_base_ee[:3, 3], R.from_matrix(T_base_ee[:3, :3])

    @staticmethod
    def _render_bgr(cam) -> np.ndarray:
        rgba = cam.get_rgba()
        if rgba is None:
            h, w = cam.get_resolution()
            return np.zeros((h, w, 3), dtype=np.uint8)
        rgb = rgba[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (
                (rgb * 255).clip(0, 255).astype(np.uint8)
                if rgb.max() <= 1.0
                else rgb.clip(0, 255).astype(np.uint8)
            )
        return rgb[:, :, [2, 1, 0]]  # RGB → BGR


# -----------------------------------------------------------------------
# ZMQ helpers
# -----------------------------------------------------------------------


def zmq_ping(socket, timeout_ms: int = 5000) -> bool:
    socket.send(pickle.dumps({"cmd": "ping"}))
    if socket.poll(timeout_ms):
        return pickle.loads(socket.recv()).get("status") == "ok"
    return False


def zmq_reset(socket):
    socket.send(pickle.dumps({"cmd": "reset"}))
    pickle.loads(socket.recv())


def zmq_set_extrinsics(socket, extrinsics: dict):
    socket.send(pickle.dumps({"cmd": "set_extrinsics", "extrinsics": extrinsics}))
    pickle.loads(socket.recv())


def zmq_step(socket, step_i: int, obs: dict):
    socket.send(pickle.dumps({"cmd": "step", "step_i": step_i, "obs": obs}))
    reply = pickle.loads(socket.recv())
    if "error" in reply:
        print(f"[isaac_client] Server error: {reply['error']}")
        return None
    action = reply["action"]
    if not isinstance(action, np.ndarray):
        action = np.array(action, dtype=np.float32)
    return action


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------


def _apply_action(robot: SimRobot, action: np.ndarray, arms: str):
    arms_list = ["left", "right"] if arms == "both" else [arms]
    for arm in arms_list:
        off = 7 if (arm == "right" and arms == "both") else 0
        robot.apply_joints(action[off : off + 7], arm)


def main(args):
    arms_list = ["left", "right"] if args.arms == "both" else [args.arms]

    # ---- --print-dofs: no policy server needed ----
    if args.print_dofs:
        import omni.usd

        omni.usd.get_context().open_stage(str(args.usd_path))
        world = World(stage_units_in_meters=1.0)
        robot = SimRobot(
            arms=arms_list,
            world=world,
            scene_prim_path=args.scene_prim_path,
            left_arm_path=args.left_arm_path,
            right_arm_path=args.right_arm_path,
            skip_cameras=True,
        )
        robot.print_dof_names()
        os._exit(0)

    # ---- Connect to policy server ----
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{args.port}")

    try:
        print(f"[isaac_client] Connecting to policy server on port {args.port}...")
        if not zmq_ping(socket):
            print("[isaac_client] ERROR: policy_server not responding.")
            print(
                "  Start it first: python egomimic/robot/isaac/policy_server.py --policy-path ..."
            )
            os._exit(1)
        print("[isaac_client] Policy server is ready.")

        # ---- Build sim ----
        print("[isaac_client] Loading USD...")
        import omni.usd

        omni.usd.get_context().open_stage(str(args.usd_path))
        world = World(stage_units_in_meters=1.0)
        robot = SimRobot(
            arms=arms_list,
            world=world,
            scene_prim_path=args.scene_prim_path,
            left_arm_path=args.left_arm_path,
            right_arm_path=args.right_arm_path,
            front_cam_path=args.front_cam_path,
            left_cam_path=args.left_cam_path,
            right_cam_path=args.right_cam_path,
            camera_resolution=(args.cam_width, args.cam_height),
        )

        # ---- Send sim extrinsics to policy server (optional) ----
        if args.use_sim_extrinsics:
            print("[isaac_client] Computing extrinsics from sim camera poses...")
            zmq_set_extrinsics(socket, robot.compute_extrinsics())

        # ---- Rollout ----
        print("[isaac_client] Starting rollout. Ctrl-C or close window to stop.")
        robot.set_home()
        zmq_reset(socket)
        step_i = 0

        while simulation_app.is_running():
            robot.step(render=True)
            obs = robot.get_obs()
            action = zmq_step(socket, step_i, obs)
            if action is not None:
                _apply_action(robot, action, args.arms)
            step_i += 1

    except KeyboardInterrupt:
        print("\n[isaac_client] Interrupted.")
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        socket.close()
        context.term()
        os._exit(0)  # bypass Isaac Sim's buggy Py_FinalizeEx / syntheticdata crash


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Isaac Sim 5.1 ZMQ client for EVA policy rollout."
    )

    p.add_argument("--usd-path", required=True)
    p.add_argument("--arms", default="both", choices=["left", "right", "both"])
    p.add_argument("--cartesian", action="store_true")
    p.add_argument("--port", type=int, default=5555)

    # Prim paths
    p.add_argument("--scene-prim-path", default="/World")
    p.add_argument("--left-arm-path", default="/World/X5A_L")
    p.add_argument("--right-arm-path", default="/World/X5A_R")
    p.add_argument("--front-cam-path", default="/World/egocentric/ego_cam")
    p.add_argument(
        "--left-cam-path", default="/World/X5A_L/cam_bracket/cam_wrist/Camera"
    )
    p.add_argument(
        "--right-cam-path", default="/World/X5A_R/cam_bracket/cam_wrist/Camera"
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)

    # Extrinsics
    p.add_argument(
        "--use-sim-extrinsics",
        action="store_true",
        help="Compute camera-to-base extrinsics from sim and send to server.",
    )

    # Diagnostic
    p.add_argument(
        "--print-dofs",
        action="store_true",
        help="Print DOF names for both arms and exit.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
