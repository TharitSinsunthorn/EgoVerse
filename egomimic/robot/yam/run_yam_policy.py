"""
Run a trained EgoVerse policy on real YAM bimanual hardware.

Pre-flight checklist:
  1.  CAN interfaces up:
        sudo ip link set can0 up type can bitrate 1000000   # left arm
        sudo ip link set can1 up type can bitrate 1000000   # right arm
  2.  Activate venv:
        source emimic/bin/activate
  3.  Find camera serials (if not known):
        python egomimic/robot/yam/robot_interface.py

Usage — Cartesian (default, recommended):
    python egomimic/robot/yam/run_yam_policy.py \\
        --policy-path logs/<run>/checkpoints/last.ckpt \\
        --arms both \\
        --cartesian \\
        --front-cam-serial <serial> \\
        --left-wrist-serial <serial> \\
        --right-wrist-serial <serial>

Usage — Joint space:
    python egomimic/robot/yam/run_yam_policy.py \\
        --policy-path logs/<run>/checkpoints/last.ckpt \\
        --arms both \\
        --front-cam-serial <serial>

Interactive keys during rollout:
  Any key  → open intervention menu
  c        → continue rollout
  r        → restart (set_home + clear policy state)
  a <path> → load new annotation file
  q        → quit
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty

# -----------------------------------------------------------------------
# Path setup: robot_utils lives in egomimic/robot/
# -----------------------------------------------------------------------
_ROBOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROBOT_DIR not in sys.path:
    sys.path.insert(0, _ROBOT_DIR)

from robot_utils import RateLoop  # noqa: E402 (after path setup)

from egomimic.robot.rollout import (  # noqa: E402
    DEFAULT_FREQUENCY,
    DEFAULT_RESAMPLE_LENGTH,
    QUERY_FREQUENCY,
    PolicyRollout,
    ReplayRollout,
    reset_rollout,
)
from egomimic.robot.yam.robot_interface import (  # noqa: E402
    OfflineYAMInterface,
    YAMInterface,
)

# -----------------------------------------------------------------------
# Terminal helper (identical to rollout.py _KeyPoll)
# -----------------------------------------------------------------------


class _KeyPoll:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def _enter_intervention(kp, policy, rollout_type):
    termios.tcsetattr(kp.fd, termios.TCSADRAIN, kp.old)
    print("\n--- INTERVENTION (rollout paused) ---")
    print("  c            : continue rollout")
    print("  a <path>     : load new annotation file")
    print("  r            : restart rollout")
    print("  q            : quit")

    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            tty.setcbreak(kp.fd)
            return "quit"

        if cmd == "c":
            print("Resuming rollout.")
            tty.setcbreak(kp.fd)
            return "continue"
        elif cmd == "q":
            tty.setcbreak(kp.fd)
            return "quit"
        elif cmd == "r":
            tty.setcbreak(kp.fd)
            return "restart"
        elif cmd.startswith("a "):
            ann_path = cmd[2:].strip()
            if not ann_path:
                print("Usage: a <annotation_path>")
                continue
            if rollout_type != "policy" or not isinstance(policy, PolicyRollout):
                print("Annotation loading is only supported for policy rollouts.")
                continue
            policy.load_annotation(ann_path)
        else:
            print(f"Unknown command: '{cmd}'. Use c / a <path> / r / q.")


def _build_robot_interface(args, arms_list):
    if args.offline_debug:
        print("[run_yam_policy] Offline mode — no hardware will be used.")
        return OfflineYAMInterface(
            arms=arms_list,
            dataset_path=args.offline_episode_path,
        )
    print("[run_yam_policy] Connecting to YAM hardware...")
    ri = YAMInterface(
        arms=arms_list,
        left_channel=args.left_channel,
        right_channel=args.right_channel,
        front_cam_serial=args.front_cam_serial or None,
        left_wrist_serial=args.left_wrist_serial or None,
        right_wrist_serial=args.right_wrist_serial or None,
    )
    print("[run_yam_policy] Hardware connected.")
    return ri


def main(args):
    arms_list = ["right", "left"] if args.arms == "both" else [args.arms]

    if args.offline_episode_path is not None and not args.offline_debug:
        raise ValueError("--offline-episode-path requires --offline-debug.")
    if (
        args.policy_path is not None
        and args.offline_debug
        and args.offline_episode_path is None
    ):
        raise ValueError(
            "--policy-path requires --offline-episode-path in --offline-debug mode."
        )

    ri = _build_robot_interface(args, arms_list)

    if args.policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=args.arms,
            policy_path=args.policy_path,
            query_frequency=args.query_frequency,
            cartesian=args.cartesian,
            extrinsics_key=args.extrinsics_key,
            resampled_action_len=args.resampled_action_len,
            annotation_path=args.annotation_path,
        )
    elif args.dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=args.dataset_path, cartesian=args.cartesian)
    else:
        ri.close()
        raise ValueError("Provide --policy-path or --dataset-path.")

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)
            result = _enter_intervention(kp, policy, rollout_type)
            if result == "quit":
                print("Quit requested.")
                return
            if result == "restart":
                reset_rollout(ri, policy)

            while True:
                with RateLoop(frequency=args.frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch is not None:
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                print("Quit requested.")
                                return
                            elif result == "restart":
                                print("Restart requested.")
                                reset_rollout(ri, policy)
                                result = _enter_intervention(kp, policy, rollout_type)
                                if result == "quit":
                                    return
                                if result == "restart":
                                    reset_rollout(ri, policy)
                                break
                            if hasattr(policy, "actions"):
                                policy.actions = None
                            break

                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            actions = policy.rollout_step(step_i, obs)
                        else:
                            actions = policy.rollout_step(step_i)

                        if actions is None:
                            print("Fininh rollout.")
                            reset_rollout(ri, policy)
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                return
                            if result == "restart":
                                reset_rollout(ri, policy)
                            break

                        for arm in arms_list:
                            arm_offset = (
                                7 if (arm == "right" and args.arms == "both") else 0
                            )
                            arm_action = actions[arm_offset : arm_offset + 7]
                            if args.cartesian:
                                ri.set_pose(arm_action, arm)
                            else:
                                ri.set_joints(arm_action, arm)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
    finally:
        print("[run_yam_policy] Closing hardware connections...")
        ri.close()


def build_arg_parser(
    description="Run an EgoVerse policy on real YAM bimanual hardware.",
):
    import argparse

    parser = argparse.ArgumentParser(description=description)

    # Arms
    parser.add_argument(
        "--arms",
        default="both",
        choices=["left", "right", "both"],
        help="Which arm(s) to control",
    )

    # CAN channels
    parser.add_argument(
        "--left-channel",
        default="can0",
        help="CAN interface for left arm",
    )
    parser.add_argument(
        "--right-channel",
        default="can1",
        help="CAN interface for right arm",
    )

    # Camera serials — defaults from Apr-2026 calibration; override if hardware changes
    parser.add_argument(
        "--front-cam-serial",
        default="409122272713",
        help="RealSense D405 serial for front/egocam",
    )
    parser.add_argument(
        "--left-wrist-serial",
        default="323622272555",
        help="RealSense D405 serial for left wrist",
    )
    parser.add_argument(
        "--right-wrist-serial",
        default="352122272502",
        help="RealSense D405 serial for right wrist",
    )

    # Control
    parser.add_argument(
        "--frequency", type=float, default=DEFAULT_FREQUENCY, help="Control loop Hz"
    )
    parser.add_argument(
        "--query-frequency",
        type=int,
        default=QUERY_FREQUENCY,
        help="Steps between policy inference calls",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="Control in cartesian space instead of joint space).",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length (e.g., 100 -> 45). Euler if --cartesian",
    )

    # Policy
    parser.add_argument("--policy-path", type=str, help="Policy checkpoint path")
    parser.add_argument("--dataset-path", type=str, help="Dataset path for replay")
    parser.add_argument(
        "--offline-debug",
        action="store_true",
        help="Use the offline dummy robot inference for rollout debugging",
    )
    parser.add_argument(
        "--offline-episode-path",
        type=str,
        help="Local EVA Zarr episode path used as observation source in offline debug mode",
    )
    parser.add_argument(
        "--extrinsics-key",
        default="yamApr2026",
        help="CameraTransforms extrinsics key (default: yamApr2026, from Apr-2026 calibration)",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        help="Path to the annotation file",
    )

    # Visualization
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug visualization of actions on images",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the visualization frames to an MP4 file (requires --visualize)",
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
