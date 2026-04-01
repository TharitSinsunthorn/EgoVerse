import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.scripts.data_download.sync_s3 import (
    DEFAULT_FILTERS,
    parse_dataset_filter_key,
)

DEFAULT_DATASET_ROOT = Path("/coc/flash7/scratch/egoverseS3ZarrDataset/")
DEFAULT_OUTPUT_PATH = Path("/coc/flash9/fryan6/data/egoverse_head_meta.json")
DEFAULT_FILTER_KEY = "aria-fold-clothes"


@dataclass(frozen=True)
class MotionConfig:
    window_size_s: float
    stride_fraction: float
    contiguous_gap_s: float
    position_weight: float
    angular_weight: float


def _safe_float(value: float | np.floating) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_mean": _safe_float(np.mean(values)),
        f"{prefix}_median": _safe_float(np.median(values)),
        f"{prefix}_p90": _safe_float(np.percentile(values, 90)),
        f"{prefix}_p95": _safe_float(np.percentile(values, 95)),
        f"{prefix}_max": _safe_float(np.max(values)),
    }


def _window_suffix(window_s: float) -> str:
    return str(window_s).replace(".", "p") + "s"


def _normalize_quaternions_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = quat_wxyz / norms

    contiguous = normalized.copy()
    for i in range(1, len(contiguous)):
        if np.dot(contiguous[i - 1], contiguous[i]) < 0:
            contiguous[i] = -contiguous[i]
    return contiguous


def _rotation_step_degrees(quat_wxyz: np.ndarray) -> np.ndarray:
    if len(quat_wxyz) < 2:
        return np.empty((0,), dtype=np.float64)

    quat = _normalize_quaternions_wxyz(quat_wxyz)
    dots = np.sum(quat[:-1] * quat[1:], axis=1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    angles_rad = 2.0 * np.arccos(dots)
    return np.degrees(angles_rad)


def _valid_pose_mask(head_pose: np.ndarray) -> np.ndarray:
    finite_mask = np.all(np.isfinite(head_pose), axis=1)
    sentinel_mask = np.all(np.abs(head_pose) < 1e8, axis=1)
    quat = head_pose[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1)
    quat_mask = quat_norm > 1e-8
    return finite_mask & sentinel_mask & quat_mask


def _build_segments(
    timestamps_ns: np.ndarray,
    valid_mask: np.ndarray,
    *,
    contiguous_gap_ns: int,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    prev_idx = None

    for idx, is_valid in enumerate(valid_mask):
        if not is_valid:
            if start is not None:
                segments.append((start, idx))
                start = None
            prev_idx = None
            continue

        if start is None:
            start = idx
            prev_idx = idx
            continue

        assert prev_idx is not None
        gap_ns = int(timestamps_ns[idx] - timestamps_ns[prev_idx])
        if gap_ns > contiguous_gap_ns:
            segments.append((start, prev_idx + 1))
            start = idx
        prev_idx = idx

    if start is not None:
        end = (prev_idx + 1) if prev_idx is not None else start + 1
        segments.append((start, end))

    return [segment for segment in segments if segment[1] - segment[0] >= 2]


def _segment_step_stats(
    head_pose: np.ndarray,
    timestamps_ns: np.ndarray,
    segments: list[tuple[int, int]],
) -> dict[str, float]:
    position_steps = []
    angular_steps_deg = []
    segment_durations_s = []

    for start, end in segments:
        pose_segment = head_pose[start:end]
        ts_segment = timestamps_ns[start:end]

        position_steps.append(
            np.linalg.norm(np.diff(pose_segment[:, :3], axis=0), axis=1)
        )
        angular_steps_deg.append(_rotation_step_degrees(pose_segment[:, 3:7]))
        segment_durations_s.append((ts_segment[-1] - ts_segment[0]) / 1e9)

    position_steps_arr = (
        np.concatenate(position_steps) if position_steps else np.empty((0,))
    )
    angular_steps_arr = (
        np.concatenate(angular_steps_deg) if angular_steps_deg else np.empty((0,))
    )
    segment_durations_arr = np.asarray(segment_durations_s, dtype=np.float64)

    stats = {
        "head_num_segments": float(len(segments)),
        "head_total_segment_duration_s": _safe_float(np.sum(segment_durations_arr))
        if segment_durations_arr.size
        else 0.0,
    }
    stats.update(_summarize(segment_durations_arr, "head_segment_duration_s"))
    stats.update(_summarize(position_steps_arr, "head_position_step_m"))
    stats.update(_summarize(angular_steps_arr, "head_angular_step_deg"))
    return stats


def _window_metrics_for_segment(
    pose_segment: np.ndarray,
    ts_segment: np.ndarray,
    *,
    window_ns: int,
    stride_ns: int,
    position_weight: float,
    angular_weight: float,
) -> list[dict[str, float]]:
    metrics = []
    start_idx = 0
    n = len(ts_segment)

    while start_idx < n - 1:
        start_ts = int(ts_segment[start_idx])
        end_idx = int(np.searchsorted(ts_segment, start_ts + window_ns, side="right"))
        if end_idx - start_idx >= 2:
            pose_window = pose_segment[start_idx:end_idx]
            ts_window = ts_segment[start_idx:end_idx]
            duration_s = (ts_window[-1] - ts_window[0]) / 1e9
            if duration_s > 0:
                position_steps = np.linalg.norm(
                    np.diff(pose_window[:, :3], axis=0), axis=1
                )
                angular_steps_deg = _rotation_step_degrees(pose_window[:, 3:7])
                position_path_m = _safe_float(np.sum(position_steps))
                angular_path_deg = _safe_float(np.sum(angular_steps_deg))
                motion_score = (
                    position_weight * position_path_m
                    + angular_weight * angular_path_deg
                )

                metrics.append(
                    {
                        "duration_s": duration_s,
                        "position_path_m": position_path_m,
                        "angular_path_deg": angular_path_deg,
                        "motion_score": motion_score,
                    }
                )

        next_start_ts = start_ts + stride_ns
        next_idx = int(np.searchsorted(ts_segment, next_start_ts, side="left"))
        start_idx = max(next_idx, start_idx + 1)

    return metrics


def _windowed_stats(
    head_pose: np.ndarray,
    timestamps_ns: np.ndarray,
    segments: list[tuple[int, int]],
    config: MotionConfig,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    window_s = config.window_size_s
    suffix = _window_suffix(window_s)
    window_ns = int(window_s * 1e9)
    stride_ns = max(1, int(window_ns * config.stride_fraction))
    window_metrics = []

    for start, end in segments:
        pose_segment = head_pose[start:end]
        ts_segment = timestamps_ns[start:end]
        window_metrics.extend(
            _window_metrics_for_segment(
                pose_segment,
                ts_segment,
                window_ns=window_ns,
                stride_ns=stride_ns,
                position_weight=config.position_weight,
                angular_weight=config.angular_weight,
            )
        )

    stats["head_window_size_s"] = float(window_s)
    stats[f"head_num_windows_w{suffix}"] = float(len(window_metrics))
    if not window_metrics:
        for key in (
            "duration_s",
            "position_path_m",
            "angular_path_deg",
            "motion_score",
        ):
            stats.update(_summarize(np.empty((0,)), f"head_{key}_w{suffix}"))
        return stats

    for key in ("duration_s", "position_path_m", "angular_path_deg", "motion_score"):
        values = np.asarray([entry[key] for entry in window_metrics], dtype=np.float64)
        stats.update(_summarize(values, f"head_{key}_w{suffix}"))
    return stats


def compute_episode_metadata(
    path: Path, config: MotionConfig
) -> tuple[str, dict[str, Any]]:
    episode_hash = path.name[:-5] if path.name.endswith(".zarr") else path.name
    try:
        store = zarr.open_group(str(path), mode="r")
        head_pose = np.asarray(store["obs_head_pose"][:], dtype=np.float64)
        timestamps_ns = np.asarray(store["obs_rgb_timestamps_ns"][:], dtype=np.int64)
        attrs = dict(store.attrs)
    except Exception as exc:
        return episode_hash, {"head_meta_error": str(exc)}

    if head_pose.ndim != 2 or head_pose.shape[1] != 7:
        return episode_hash, {
            "head_meta_error": f"Expected obs_head_pose shape (T, 7), got {head_pose.shape}"
        }
    if timestamps_ns.ndim != 1 or len(timestamps_ns) != len(head_pose):
        return episode_hash, {
            "head_meta_error": (
                f"Expected obs_rgb_timestamps_ns shape ({len(head_pose)},), "
                f"got {timestamps_ns.shape}"
            )
        }

    valid_mask = _valid_pose_mask(head_pose)
    contiguous_gap_ns = int(config.contiguous_gap_s * 1e9)
    segments = _build_segments(
        timestamps_ns,
        valid_mask,
        contiguous_gap_ns=contiguous_gap_ns,
    )

    output: dict[str, Any] = {
        "head_episode_hash": episode_hash,
        "head_total_frames": float(len(head_pose)),
        "head_valid_frames": float(np.sum(valid_mask)),
        "head_valid_fraction": _safe_float(np.mean(valid_mask))
        if len(valid_mask)
        else 0.0,
        "head_contiguous_gap_s": float(config.contiguous_gap_s),
        "head_window_stride_fraction": float(config.stride_fraction),
        "head_position_weight": float(config.position_weight),
        "head_angular_weight": float(config.angular_weight),
        "head_embodiment": attrs.get("embodiment", ""),
    }

    if len(timestamps_ns) >= 2:
        gaps_s = np.diff(timestamps_ns).astype(np.float64) / 1e9
        output.update(_summarize(gaps_s, "head_frame_gap_s"))
    else:
        output.update(_summarize(np.empty((0,)), "head_frame_gap_s"))

    output.update(_segment_step_stats(head_pose, timestamps_ns, segments))
    output.update(_windowed_stats(head_pose, timestamps_ns, segments, config))

    if segments:
        segment_lengths = np.asarray(
            [end - start for start, end in segments], dtype=np.float64
        )
        output.update(_summarize(segment_lengths, "head_segment_frames"))
    else:
        output.update(_summarize(np.empty((0,)), "head_segment_frames"))

    return episode_hash, output


def _episode_paths(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (
            path.name.endswith(".zarr")
            or (path / ".zgroup").exists()
            or (path / "zarr.json").exists()
        ):
            paths.append(path)
    return paths


def _resolve_episode_paths(root: Path, filter_key: str | None) -> list[Path]:
    if filter_key is None:
        return _episode_paths(root)

    filters = parse_dataset_filter_key(filter_key)
    sql_paths = S3EpisodeResolver._get_filtered_paths(filters=filters)
    wanted_hashes = {episode_hash for _, episode_hash in sql_paths}

    local_paths = _episode_paths(root)
    local_paths_by_hash = {
        (path.name[:-5] if path.name.endswith(".zarr") else path.name): path
        for path in local_paths
    }
    matched_paths = [
        local_paths_by_hash[episode_hash]
        for episode_hash in sorted(wanted_hashes)
        if episode_hash in local_paths_by_hash
    ]

    missing_count = len(wanted_hashes) - len(matched_paths)
    print(
        f"SQL filter {filter_key!r} matched {len(wanted_hashes)} episodes; "
        f"found {len(matched_paths)} locally under {root}"
    )
    if missing_count > 0:
        print(f"Skipped {missing_count} filtered SQL episodes not present locally")

    return matched_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-episode Aria head-motion metadata sidecar from local zarr episodes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing per-episode zarr folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output JSON sidecar.",
    )
    parser.add_argument(
        "--window-sizes",
        "--window-size-s",
        type=float,
        dest="window_size_s",
        default=1.0,
        help="Temporal window size in seconds.",
    )
    parser.add_argument(
        "--stride-fraction",
        type=float,
        default=1.0,
        help="Sliding-window stride as a fraction of the window size. Default 1.0 means stride equals window size.",
    )
    parser.add_argument(
        "--contiguous-gap-s",
        type=float,
        default=0.05,
        help="Break segments when timestamp gaps exceed this many seconds.",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=1.0,
        help="Weight for position path length in the combined motion score.",
    )
    parser.add_argument(
        "--angular-weight",
        type=float,
        default=0.01,
        help="Weight for angular path in degrees in the combined motion score.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes. Use 0 to run in-process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, process only the first N episodes.",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=DEFAULT_FILTER_KEY,
        help=(
            "Named DatasetFilter preset key used to select local episodes before "
            "computing metadata. Use 'none' to disable filtering. "
            f"Available keys: {', '.join(sorted(DEFAULT_FILTERS))}"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.stride_fraction <= 0:
        raise ValueError("--stride-fraction must be > 0")
    if args.contiguous_gap_s <= 0:
        raise ValueError("--contiguous-gap-s must be > 0")
    if args.window_size_s <= 0:
        raise ValueError("--window-size-s must be > 0")

    config = MotionConfig(
        window_size_s=float(args.window_size_s),
        stride_fraction=float(args.stride_fraction),
        contiguous_gap_s=float(args.contiguous_gap_s),
        position_weight=float(args.position_weight),
        angular_weight=float(args.angular_weight),
    )

    filter_key = None if args.filters.lower() == "none" else args.filters
    episode_paths = _resolve_episode_paths(args.dataset_root, filter_key)
    if args.limit > 0:
        episode_paths = episode_paths[: args.limit]
    if not episode_paths:
        if filter_key is None:
            raise ValueError(f"No zarr episodes found under {args.dataset_root}")
        raise ValueError(
            f"No zarr episodes found under {args.dataset_root} matching filter {filter_key!r}"
        )

    result: dict[str, dict[str, Any]] = {
        "__meta__": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(args.dataset_root),
            "filters": filter_key,
            "window_size_s": config.window_size_s,
            "stride_fraction": config.stride_fraction,
            "contiguous_gap_s": config.contiguous_gap_s,
            "position_weight": config.position_weight,
            "angular_weight": config.angular_weight,
            "num_episodes": len(episode_paths),
        }
    }

    if args.workers <= 0:
        for idx, path in enumerate(episode_paths, start=1):
            episode_hash, metadata = compute_episode_metadata(path, config)
            result[episode_hash] = metadata
            if idx % 25 == 0 or idx == len(episode_paths):
                print(f"Processed {idx}/{len(episode_paths)} episodes")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(compute_episode_metadata, path, config): path
                for path in episode_paths
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                episode_hash, metadata = future.result()
                result[episode_hash] = metadata
                if idx % 25 == 0 or idx == len(episode_paths):
                    print(f"Processed {idx}/{len(episode_paths)} episodes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote head metadata for {len(episode_paths)} episodes to {args.output}")


if __name__ == "__main__":
    main()
