#!/usr/bin/env python3
"""
Rerank IRASim rollout candidates and export README-style submission folders.

Expected rollout layout (from rollout_minireal.py):
  <rollouts>/<episode>/candidate_<k>/frames.npy   uint8 [50,H,W,3]
                        meta.json
                        action_51x26.npy

Optional baseline:
  <rollouts>/<episode>/baseline_repeat/frames.npy  (repeat-last-frame)

Exports:
  <out>/<episode>/{video.mp4,action.txt,joint.txt,instruction.txt,instructions.txt}

video.mp4 is exactly 50 frames: frames 0–15 are reused from test video.mp4 at native
resolution (conditioning prefix, not overwritten). Frames 16–49 are the first 34
frames of the rollout candidate, resized to test resolution (e.g. 1280×720).
Rollout still stores 50 generated frames; only the leading 34 align with this
50-frame submission timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.minireal_action_util import pick_instruction

PREFIX_FRAMES = 16
EXPORT_VIDEO_FRAMES = 50
SUFFIX_FROM_ROLLOUT = EXPORT_VIDEO_FRAMES - PREFIX_FRAMES  # 34


def load_frames(path: Path) -> np.ndarray:
    return np.load(path)


def visual_score(frames: np.ndarray) -> float:
    """Higher is better: brightness + motion."""
    if frames.size == 0:
        return -1e9
    x = frames.astype(np.float64)
    mu = float(x.mean())
    if x.shape[0] >= 2:
        d = np.abs(x[1:] - x[:-1]).mean()
    else:
        d = 0.0
    # Penalize near-black generations
    if mu < 8.0:
        return -1e6 + mu
    return mu * 0.01 + d


def read_csv_header_and_max_id(test_action_path: Path) -> Tuple[List[str], float]:
    with test_action_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data = np.asarray([r for r in rows[1:] if r], dtype=np.float64)
    max_id = float(data[:, 0].max()) if data.size else -1.0
    return header, max_id


def write_csv_matrix(path: Path, header: Sequence[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for row in data:
            row_out = [int(round(row[0]))]
            row_out.extend([f"{float(v):.10f}" for v in row[1:]])
            w.writerow(row_out)


def build_action_rows(base_id: float, action_body_51x26: np.ndarray) -> np.ndarray:
    """Prepend Unnamed: 0 column with consecutive integers."""
    ids = np.arange(base_id + 1, base_id + 1 + action_body_51x26.shape[0], dtype=np.float64).reshape(
        -1, 1
    )
    return np.concatenate([ids, action_body_51x26.astype(np.float64)], axis=1)


def read_video_prefix_frames_rgb(path: Path, n_frames: int = PREFIX_FRAMES) -> np.ndarray:
    """First n_frames from video as RGB uint8 [n,H,W,3] at native resolution."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while len(frames) < n_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < n_frames:
        raise RuntimeError(f"{path}: need >= {n_frames} frames for conditioning prefix export, got {len(frames)}")
    return np.stack(frames, axis=0).astype(np.uint8)


def read_video_size_wh(path: Path) -> tuple[int, int]:
    """Return (width, height) of the reference video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open reference video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Bad reference video size for {path}: {w}x{h}")
    return w, h


def resize_frames_cv2(frames: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    import cv2

    tw, th = target_wh
    h, w = frames.shape[1], frames.shape[2]
    if (w, h) == (tw, th):
        return frames
    return np.stack(
        [cv2.resize(fr, (tw, th), interpolation=cv2.INTER_LANCZOS4) for fr in frames],
        axis=0,
    ).astype(np.uint8)


def write_video_cv2(
    path: Path, frames: np.ndarray, fps: int, target_wh: tuple[int, int] | None = None
) -> None:
    import cv2

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames shape {frames.shape}")
    if target_wh is not None:
        frames = resize_frames_cv2(frames, target_wh)
    h, w = frames.shape[1], frames.shape[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed for {path}")
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def export_episode(
    episode: str,
    roll_root: Path,
    test_root: Path,
    sub_root: Path,
    fps: int,
    base_id_override: float | None,
) -> None:
    ep_roll = roll_root / episode
    if not ep_roll.is_dir():
        raise FileNotFoundError(ep_roll)

    cand_dirs = sorted(
        [p for p in ep_roll.iterdir() if p.is_dir() and p.name.startswith("candidate_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    baseline_dir = ep_roll / "baseline_repeat"
    candidates: List[Tuple[str, Path, np.ndarray]] = []
    if baseline_dir.is_dir() and (baseline_dir / "frames.npy").is_file():
        candidates.append(
            ("baseline_repeat", baseline_dir, load_frames(baseline_dir / "frames.npy"))
        )
    for cd in cand_dirs:
        fp = cd / "frames.npy"
        if fp.is_file():
            candidates.append((cd.name, cd, load_frames(fp)))

    if not candidates:
        raise RuntimeError(f"No candidates under {ep_roll}")

    best_name, best_dir, best_frames = max(candidates, key=lambda x: visual_score(x[2]))

    test_ep = test_root / episode
    action_ref = test_ep / "action.txt"
    joint_ref = test_ep / "joint.txt"
    if not action_ref.is_file() or not joint_ref.is_file():
        raise FileNotFoundError(f"Need test {episode}/action.txt and joint.txt")

    hdr_a, max_a = read_csv_header_and_max_id(action_ref)
    hdr_j, max_j = read_csv_header_and_max_id(joint_ref)
    base_id = base_id_override
    if base_id is None:
        base_id = max(max_a, max_j)

    act_body = np.load(best_dir / "action_51x26.npy")
    if act_body.shape != (51, 26):
        raise ValueError(f"{best_dir}/action_51x26.npy bad shape {act_body.shape}")

    action_mat = build_action_rows(base_id, act_body)
    joint_mat = (
        build_action_rows(base_id, np.load(best_dir / "joint_51x26.npy"))
        if (best_dir / "joint_51x26.npy").is_file()
        else build_action_rows(base_id, act_body)
    )

    out_ep = sub_root / episode
    out_ep.mkdir(parents=True, exist_ok=True)
    write_csv_matrix(out_ep / "action.txt", hdr_a, action_mat)
    write_csv_matrix(out_ep / "joint.txt", hdr_j, joint_mat)

    instr = pick_instruction(test_ep)
    (out_ep / "instruction.txt").write_text(instr + "\n", encoding="utf-8")
    (out_ep / "instructions.txt").write_text(instr + "\n", encoding="utf-8")

    ref_vid = test_ep / "video.mp4"
    target_wh = read_video_size_wh(ref_vid)
    prefix_native = read_video_prefix_frames_rgb(ref_vid, PREFIX_FRAMES)
    tw, th = target_wh
    if (prefix_native.shape[2], prefix_native.shape[1]) != (tw, th):
        prefix_native = resize_frames_cv2(prefix_native, target_wh)
    if best_frames.shape[0] < SUFFIX_FROM_ROLLOUT:
        raise ValueError(
            f"{episode}: rollout frames need >= {SUFFIX_FROM_ROLLOUT} for 50-frame export, "
            f"got {best_frames.shape[0]}"
        )
    gen_tail = best_frames[:SUFFIX_FROM_ROLLOUT]
    generated_hw = resize_frames_cv2(gen_tail, target_wh)
    full_video = np.concatenate([prefix_native, generated_hw], axis=0)
    assert full_video.shape[0] == EXPORT_VIDEO_FRAMES
    write_video_cv2(out_ep / "video.mp4", full_video, fps=fps, target_wh=None)

    meta = {
        "episode": episode,
        "selected": best_name,
        "score": visual_score(best_frames),
        "fps": fps,
        "base_row_id": base_id,
        "output_resolution": [target_wh[0], target_wh[1]],
        "video_prefix_frames_native": PREFIX_FRAMES,
        "video_from_rollout_frames": SUFFIX_FROM_ROLLOUT,
        "video_total_frames": EXPORT_VIDEO_FRAMES,
        "rollout_candidate_frames": int(best_frames.shape[0]),
    }
    (out_ep / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported {out_ep} (candidate={best_name})")


def maybe_add_baseline(test_root: Path, episode: str, roll_root: Path, prefix_frames: int = 16):
    """Create baseline_repeat from last frame of test video (optional helper)."""
    import cv2

    ep = test_root / episode
    vid = ep / "video.mp4"
    if not vid.is_file():
        return
    cap = cv2.VideoCapture(str(vid))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < prefix_frames:
        return
    last = frames[prefix_frames - 1]
    rep = np.stack([last] * 50, axis=0).astype(np.uint8)
    dest = roll_root / episode / "baseline_repeat"
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "frames.npy", rep)
    print(f"Added baseline_repeat for {episode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True, help="Original test folders for headers + instruction")
    parser.add_argument("--out", type=str, default="submission")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--first-row-id",
        type=float,
        default=None,
        help="Override Unnamed: 0 numbering start (default max id from test action/joint)",
    )
    parser.add_argument(
        "--add-baseline-repeat",
        action="store_true",
        help="Create repeat-last-frame candidate before reranking",
    )
    args = parser.parse_args()

    roll_root = Path(args.rollouts).resolve()
    test_root = Path(args.test_data).resolve()
    sub_root = Path(args.out)
    if not sub_root.is_absolute():
        sub_root = (_ROOT / sub_root).resolve()
    else:
        sub_root = sub_root.resolve()

    episodes = sorted([p.name for p in roll_root.iterdir() if p.is_dir()])
    for ep in episodes:
        if args.add_baseline_repeat:
            maybe_add_baseline(test_root, ep, roll_root)
        export_episode(ep, roll_root, test_root, sub_root, args.fps, args.first_row_id)


if __name__ == "__main__":
    main()
