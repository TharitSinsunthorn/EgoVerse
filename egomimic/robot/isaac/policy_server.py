"""
ZMQ policy server for HPT inference.

Runs in the emimic venv. Loads an EgoVerse checkpoint and serves
action predictions over a ZMQ REP socket.

The isaac_client.py (in ego_sim env) sends observations and receives
actions — it never imports EgoVerse directly.

--- How to run ---
    source /home/tharit/EgoVerse/emimic/bin/activate
    python egomimic/robot/isaac/policy_server.py \\
        --policy-path logs/test/test_2026-04-10_20-39-35/checkpoints/last.ckpt \\
        --arms both \\
        --cartesian

--- Protocol ---
Client sends (pickle):
    {"cmd": "step",  "step_i": int, "obs": obs_dict}
    {"cmd": "reset"}
    {"cmd": "set_extrinsics", "extrinsics": {"left": 4x4, "right": 4x4}}
    {"cmd": "ping"}

Server replies (pickle):
    step  -> {"action": np.array [14], "inference_time": float}
    reset -> {"status": "ok"}
    set_extrinsics -> {"status": "ok"}
    ping  -> {"status": "ok"}
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import zmq

# Make EgoVerse importable when running from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# rollout.py does `from robot_utils import RateLoop` (bare import)
# so egomimic/robot/ must also be on sys.path
_ROBOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROBOT_DIR not in sys.path:
    sys.path.insert(0, _ROBOT_DIR)

# robot_interface.py lives in eva/eva_ws/src/eva/ and is imported by rollout.py
_EVA_DIR = os.path.join(_ROBOT_DIR, "eva/eva_ws/src/eva")
if _EVA_DIR not in sys.path:
    sys.path.insert(0, _EVA_DIR)

# ruff: noqa: E402
from scipy.spatial.transform import Rotation as R

from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
from egomimic.robot.rollout import PolicyRollout


def _model_xml_path() -> str:
    candidates = [
        "/home/robot/robot_ws/egomimic/resources/model_x5.xml",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../resources/model_x5.xml")
        ),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


def cartesian_to_joints(
    action_cartesian: np.ndarray,
    current_joints: np.ndarray,
    ik_solver: EvaMinkKinematicsSolver,
    arms: str,
) -> np.ndarray:
    """
    Convert Cartesian EE pose action → joint positions via IK.

    action_cartesian: [14] for bimanual — left [x,y,z,yaw,pitch,roll,gripper]
                                          right [x,y,z,yaw,pitch,roll,gripper]
    current_joints:   [14] current arm joint positions + gripper (from sim obs)
    Returns joint array [14]: [joint1..joint6 (rad), gripper_norm (0-1)] × 2
    """
    arms_list = ["left", "right"] if arms == "both" else [arms]
    result = current_joints[:14].copy().astype(np.float64)

    for arm in arms_list:
        off = 7 if arm == "right" else 0
        cart = action_cartesian[off : off + 7]
        cur_q6 = current_joints[off : off + 6].astype(np.float64)
        gripper = float(cart[6])

        pos_xyz = cart[:3].astype(np.float64)
        rot_mat = R.from_euler("ZYX", cart[3:6].astype(np.float64)).as_matrix()

        # Try 1: full pose IK (position + orientation from policy)
        q_sol = None
        try:
            q_sol = ik_solver.ik_with_retries(pos_xyz, rot_mat, cur_q6)
        except Exception:
            pass

        # Try 2: position-only IK — keep current FK orientation, ignore policy rotation.
        # Use larger dt so the solver can make bigger steps per iteration.
        if q_sol is None:
            try:
                _, fk_rot = ik_solver.fk(cur_q6)
                q_sol = ik_solver.ik_with_retries(
                    pos_xyz, fk_rot.as_matrix(), cur_q6, dt=0.05
                )
                if q_sol is not None:
                    print(f"[policy_server] IK pos-only fallback succeeded ({arm})")
            except Exception:
                q_sol = None

        if q_sol is None:
            fk_pos, _ = ik_solver.fk(cur_q6)
            print(
                f"[policy_server] IK failed ({arm})"
                f"  target_xyz={np.round(pos_xyz, 3)}"
                f"  cur_fk_xyz={np.round(fk_pos, 3)}"
                f"  cur_q6={np.round(cur_q6, 3)}"
            )
            q_sol = cur_q6

        result[off : off + 6] = q_sol
        result[off + 6] = gripper  # keep normalized (0=close, 1=open)

    return result


def build_policy(args) -> PolicyRollout:
    print(f"[policy_server] Loading checkpoint: {args.policy_path}")
    policy = PolicyRollout(
        arm=args.arms,
        policy_path=args.policy_path,
        query_frequency=args.query_frequency,
        cartesian=args.cartesian,
        extrinsics_key=args.extrinsics_key,
        resampled_action_len=args.resampled_action_len,
    )
    print("[policy_server] Checkpoint loaded and ready.")
    return policy


def main(args):
    policy = build_policy(args)

    # IK solver — only needed in Cartesian mode
    ik_solver = None
    if args.cartesian:
        model_path = _model_xml_path()
        print(f"[policy_server] Loading IK solver from {model_path}")
        ik_solver = EvaMinkKinematicsSolver(model_path=model_path)
        print("[policy_server] IK solver ready.")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"[policy_server] Listening on tcp://localhost:{args.port}")
    print("[policy_server] Waiting for client...")

    step_i = 0

    try:
        while True:
            raw = socket.recv()
            request = pickle.loads(raw)
            cmd = request.get("cmd")

            # ---- ping: check server is alive ----
            if cmd == "ping":
                socket.send(pickle.dumps({"status": "ok"}))

            # ---- reset: clear policy state, restart step counter ----
            elif cmd == "reset":
                policy.reset()
                step_i = 0
                socket.send(pickle.dumps({"status": "ok"}))
                print("[policy_server] Reset.")

            # ---- set_extrinsics: override camera-to-base transforms ----
            # Call this once after sim loads so the server uses the sim's
            # actual camera poses instead of the hardcoded x5Dec13_2 values.
            elif cmd == "set_extrinsics":
                policy.extrinsics = request["extrinsics"]
                socket.send(pickle.dumps({"status": "ok"}))
                print("[policy_server] Extrinsics updated from sim.")

            # ---- step: run one inference step ----
            elif cmd == "step":
                obs = request["obs"]
                # Client can override step_i (for action chunk replay alignment)
                step_i = request.get("step_i", step_i)

                t0 = time.time()
                # rollout_step returns Cartesian EE poses [14] in base frame
                action = policy.rollout_step(step_i, obs)

                # Convert Cartesian → joint positions via IK (Cartesian mode only)
                if ik_solver is not None:
                    current_joints = np.asarray(
                        obs.get("joint_positions", np.zeros(14)), dtype=np.float64
                    )
                    action = cartesian_to_joints(
                        action, current_joints, ik_solver, args.arms
                    )

                elapsed = time.time() - t0

                step_i += 1

                if elapsed > (1.0 / args.query_frequency) * 0.9:
                    print(f"[policy_server] Inference+IK: {elapsed*1000:.1f}ms")

                socket.send(
                    pickle.dumps(
                        {
                            "action": action.tolist(),  # list[float] — avoids numpy version mismatch
                            "inference_time": elapsed,
                        }
                    )
                )

            else:
                socket.send(pickle.dumps({"error": f"Unknown command: {cmd}"}))

    except KeyboardInterrupt:
        print("\n[policy_server] Shutting down.")
    finally:
        socket.close()
        context.term()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZMQ server: serves HPT policy inference over localhost."
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path to trained EgoVerse checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--arms",
        default="both",
        choices=["left", "right", "both"],
        help="Which arms the policy controls.",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="Policy outputs Cartesian actions (xyz+ypr+gripper).",
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="ZMQ port to listen on (default: 5555)."
    )
    parser.add_argument(
        "--query-frequency",
        type=int,
        default=30,
        help="Run inference every N steps; replay cached chunk in between.",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=45,
        help="Downsample predicted action chunk to this length.",
    )
    parser.add_argument(
        "--extrinsics-key",
        default="x5Dec13_2",
        help="Camera-to-base extrinsics key. Overridden at runtime if the "
        "client sends set_extrinsics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
