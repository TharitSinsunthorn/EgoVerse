"""
Camera stream test — no arms needed.

Opens all three RealSense D405s and shows a live side-by-side preview.

Controls:
    q  — quit
    s  — save a snapshot (written to ./cam_snapshot_<timestamp>.png)

Usage:
    source emimic/bin/activate
    python egomimic/robot/yam/test_cameras.py

Override serials if your setup changes:
    python egomimic/robot/yam/test_cameras.py \\
        --front  409122272713 \\
        --left   323622272555 \\
        --right  352122272502
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_EVA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "eva", "eva_ws", "src", "eva")
)
if _EVA_DIR not in sys.path:
    sys.path.insert(0, _EVA_DIR)

try:
    from stream_d405 import RealSenseRecorder, list_connected_serials
except ImportError:
    print("ERROR: pyrealsense2 is not installed. Run: pip install pyrealsense2")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed. Run: pip install opencv-python")
    sys.exit(1)

DEFAULT_FRONT = "409122272713"
DEFAULT_LEFT = "323622272555"
DEFAULT_RIGHT = "352122272502"


def stream_cameras(serials: dict[str, str]) -> None:
    print("Connected RealSense devices:")
    connected = [str(s) for s in list_connected_serials()]
    for s in connected:
        print(f"  {s}")

    cameras: dict[str, RealSenseRecorder] = {}
    for name, serial in serials.items():
        if not serial:
            continue
        if serial not in connected:
            print(f"  {name} [{serial}]: not found — skipping")
            continue
        print(f"Opening {name} [{serial}] ...", end=" ", flush=True)
        cameras[name] = RealSenseRecorder(serial)
        print("ok")

    if not cameras:
        print("No cameras opened.")
        return

    time.sleep(0.5)  # let streams stabilise

    # labels = list(cameras.keys())
    print(
        f"\nStreaming {len(cameras)} camera(s). Press  q = quit,  s = save snapshot.\n"
    )

    while True:
        frames = []
        for name, cam in cameras.items():
            img = cam.get_image()
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            # label
            cv2.putText(
                img,
                name,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            frames.append(img)

        combined = np.concatenate(frames, axis=1)
        cv2.imshow("YAM cameras  (q=quit  s=save)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"cam_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Saved snapshot → {fname}")

    cv2.destroyAllWindows()
    for cam in cameras.values():
        cam.stop()
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="YAM camera live stream test")
    p.add_argument(
        "--front", default=DEFAULT_FRONT, help="Serial for front/scene camera"
    )
    p.add_argument("--left", default=DEFAULT_LEFT, help="Serial for left wrist camera")
    p.add_argument(
        "--right", default=DEFAULT_RIGHT, help="Serial for right wrist camera"
    )
    p.add_argument("--no-front", action="store_true", help="Skip front camera")
    p.add_argument("--no-left", action="store_true", help="Skip left wrist camera")
    p.add_argument("--no-right", action="store_true", help="Skip right wrist camera")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    serials: dict[str, str] = {}
    if not args.no_front:
        serials["front_img_1"] = args.front
    if not args.no_left:
        serials["left_wrist_img"] = args.left
    if not args.no_right:
        serials["right_wrist_img"] = args.right
    stream_cameras(serials)
