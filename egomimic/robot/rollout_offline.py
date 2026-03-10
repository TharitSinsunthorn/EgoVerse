import argparse
import os
import sys
import time

from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
from egomimic.robot.robot_utils import RateLoop
from egomimic.robot.rollout import (
    DEFAULT_FREQUENCY,
    DEFAULT_RESAMPLE_LENGTH,
    QUERY_FREQUENCY,
    PolicyRollout,
    ReplayRollout,
    _KeyPoll,
    debug_policy,
    reset_rollout,
)
from egomimic.utils.egomimicUtils import CameraTransforms

sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

from robot_interface_offline import DummyARXInterface


def _build_robot_interface(arms_list, offline_episode_path=None):
    return DummyARXInterface(arms=arms_list, dataset_path=offline_episode_path)


def main(
    arms,
    frequency,
    cartesian,
    query_frequency=None,
    policy_path=None,
    dataset_path=None,
    debug=False,
    resampled_action_len=None,
    offline_episode_path=None,
):
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    if policy_path is not None and offline_episode_path is None:
        raise ValueError(
            "--policy-path requires --offline-episode-path to provide EVA Zarr observations."
        )

    ri = _build_robot_interface(
        arms_list=arms_list, offline_episode_path=offline_episode_path
    )

    if policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            extrinsics_key="x5Dec13_2",
            resampled_action_len=resampled_action_len,
            debug=debug,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError("Must provide either --policy-path or --dataset-path.")

    print(f"Cartesian value {cartesian}")

    camera_transforms = CameraTransforms(
        intrinsics_key="base", extrinsics_key="x5Dec13_2"
    )
    kinematics_solver = EvaMinkKinematicsSolver(
        model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
    )

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)

            while True:
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch == "q":
                            print("Quit requested.")
                            return
                        if ch == "r":
                            print("Restart requested.")
                            reset_rollout(ri, policy)
                            time.sleep(10.0)
                            break

                        actions = None
                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            actions = policy.rollout_step(step_i, obs)
                        elif rollout_type == "replay":
                            actions = policy.rollout_step(step_i)
                        elif rollout_type == "replay_lerobot":
                            actions = policy.rollout_step(step_i)
                        else:
                            raise ValueError(f"Invalid rollout type: {rollout_type}")

                        if actions is None:
                            print(
                                "Finish rollout. Press 'r' to restart or 'q' to quit."
                            )
                            while True:
                                ch2 = kp.getch()
                                if ch2 == "q":
                                    return
                                if ch2 == "r":
                                    reset_rollout(ri, policy)
                                    time.sleep(10.0)
                                    break
                                time.sleep(0.01)
                            break

                        if debug and rollout_type == "policy":
                            debug_policy(
                                obs,
                                camera_transforms,
                                policy,
                                step_i,
                                cartesian,
                                arms,
                                kinematics_solver,
                            )

                        for arm in arms_list:
                            arm_offset = 7 if (arm == "right" and arms == "both") else 0
                            arm_action = actions[arm_offset : arm_offset + 7]
                            if cartesian:
                                ri.set_pose(arm_action, arm)
                            else:
                                ri.set_joints(arm_action, arm)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline rollout debug for robot model."
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Which arm(s) to control",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=QUERY_FREQUENCY,
        help="Frames which model does inference",
    )
    parser.add_argument("--policy-path", type=str, help="policy checkpoint path")
    parser.add_argument("--dataset-path", type=str, help="dataset path for replay")
    parser.add_argument(
        "--offline-episode-path",
        type=str,
        help="local EVA Zarr episode path used as observation source for policy rollout",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="control in cartesian space instead of joint space",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length (e.g., 100 -> 45). Euler if --cartesian.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug visualization of actions on images",
    )

    args = parser.parse_args()

    print(f"Resampling actions to {args.resampled_action_len}")
    main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        debug=args.debug,
        resampled_action_len=args.resampled_action_len,
        offline_episode_path=args.offline_episode_path,
    )
