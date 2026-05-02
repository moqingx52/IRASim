#!/usr/bin/env python3
"""
Convert MiniReal competition folders (train/<episode>/{action,joint,instruction,video}.mp4)
into IRASim robotdata layout under robotdata/opensource_robotdata/minireal/.

Run from IRASim repo root:
  python scripts/convert_minireal_to_irasim.py --src /path/to/release --split train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

# Allow imports when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.minireal_action_util import (
    joint_matrix_to_arm_states,
    parse_csv_numeric,
    pick_instruction,
)


def count_video_frames(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def convert_episode(
    ep_dir: Path,
    out_video_root: Path,
    out_ann_root: Path,
    split_name: str,
) -> dict:
    """Convert one episode directory; returns summary dict."""
    ep = ep_dir.name
    vid_src = ep_dir / "video.mp4"
    if not vid_src.is_file():
        raise FileNotFoundError(f"Missing video.mp4 in {ep_dir}")

    joint_path = ep_dir / "joint.txt"
    action_path = ep_dir / "action.txt"
    if not joint_path.is_file() or not action_path.is_file():
        raise FileNotFoundError(f"Need joint.txt and action.txt in {ep_dir}")

    header_j, joint_mat = parse_csv_numeric(joint_path)
    header_a, action_mat = parse_csv_numeric(action_path)
    if joint_mat.shape[0] != action_mat.shape[0]:
        raise ValueError(
            f"{ep}: joint rows {joint_mat.shape[0]} != action rows {action_mat.shape[0]}"
        )

    n_vid = count_video_frames(vid_src)
    T = joint_mat.shape[0]
    if n_vid != T:
        # Trim to minimum length (dataset requires consistent T)
        t_min = min(T, n_vid)
        print(
            f"[WARN] {ep}: video frames {n_vid} != table rows {T}; trimming to {t_min}"
        )
        joint_mat = joint_mat[:t_min]
        action_mat = action_mat[:t_min]
        T = t_min

    arm_states, gripper = joint_matrix_to_arm_states(joint_mat)
    state_json = arm_states.tolist()
    grip_json = gripper.tolist()

    # out_video_root already ends with the split (e.g. .../videos/train); do not repeat it.
    dest_dir = out_video_root / ep
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_rgb = dest_dir / "rgb.mp4"
    shutil.copy2(vid_src, dest_rgb)

    # action_gt: same numeric columns as CSV without index column
    action_gt = action_mat[:, 1:].astype(np.float32)
    np.save(dest_dir / "action_gt.npy", action_gt)

    instruction = pick_instruction(ep_dir)

    rel_video = f"videos/{split_name}/{ep}/rgb.mp4"
    ann = {
        "episode_id": ep,
        "instruction": instruction,
        "state": state_json,
        "continuous_gripper_state": grip_json,
        "videos": [{"video_path": rel_video}],
        "num_frames": int(T),
        "fps": 30,
    }

    out_ann_root.mkdir(parents=True, exist_ok=True)
    ann_path = out_ann_root / f"{ep}.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)

    meta = {
        "episode": ep,
        "split": split_name,
        "T": T,
        "action_dim": action_gt.shape[1],
        "header_action": header_a,
        "header_joint": header_j,
    }
    return meta


def maybe_copy_rdt(ep_dir: Path, dest_ep_dir: Path, rdt_root: Path | None) -> None:
    if rdt_root is None:
        return
    cand = rdt_root / ep_dir.name / "action_rdt.npy"
    if cand.is_file():
        shutil.copy2(cand, dest_ep_dir / "action_rdt.npy")
        print(f"  copied RDT actions from {cand}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniReal -> IRASim robotdata converter")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Release root containing train/ (and optionally test/) subfolders",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="robotdata/opensource_robotdata/minireal",
        help="Destination under IRASim root",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "test_source"),
        default="train",
        help="Which source folder under src to read (test_source maps folder name test/)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of train episodes to place in annotation/val",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rdt-actions",
        type=str,
        default=None,
        help="Optional root with <episode>/action_rdt.npy copied beside rgb.mp4",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most N episodes (debug)",
    )
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    split_dir_name = "test" if args.split == "test_source" else "train"
    split_src = src_root / split_dir_name
    if not split_src.is_dir():
        raise FileNotFoundError(f"Missing {split_src}")

    dst = Path(args.dst)
    if not dst.is_absolute():
        dst = (_ROOT / dst).resolve()

    videos_train = dst / "videos" / "train"
    videos_val = dst / "videos" / "val"
    ann_train = dst / "annotation" / "train"
    ann_val = dst / "annotation" / "val"

    episodes = sorted([p for p in split_src.iterdir() if p.is_dir()])
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise RuntimeError(f"No episode dirs under {split_src}")

    random.seed(args.seed)
    rdt_root = Path(args.rdt_actions).resolve() if args.rdt_actions else None

    summaries = []

    if split_dir_name == "train":
        if len(episodes) == 1:
            val_list, train_list = [], list(episodes)
        else:
            n_val = max(1, int(len(episodes) * args.val_ratio))
            n_val = min(n_val, len(episodes) - 1)
            order = list(episodes)
            random.shuffle(order)
            val_list = order[:n_val]
            train_list = order[n_val:]
        print(f"Train episodes: {len(train_list)}, Val episodes: {len(val_list)}")
        for ep_dir in val_list:
            print(f"Converting {ep_dir.name} -> val")
            meta = convert_episode(ep_dir, videos_val, ann_val, "val")
            maybe_copy_rdt(ep_dir, videos_val / ep_dir.name, rdt_root)
            summaries.append(meta)
        for ep_dir in train_list:
            print(f"Converting {ep_dir.name} -> train")
            meta = convert_episode(ep_dir, videos_train, ann_train, "train")
            maybe_copy_rdt(ep_dir, videos_train / ep_dir.name, rdt_root)
            summaries.append(meta)
    elif args.split == "test_source":
        for ep_dir in episodes:
            print(f"Converting test episode {ep_dir.name}")
            meta = convert_episode(
                ep_dir, dst / "videos" / "test", dst / "annotation" / "test", "test"
            )
            maybe_copy_rdt(ep_dir, dst / "videos" / "test" / ep_dir.name, rdt_root)
            summaries.append(meta)
    else:
        raise ValueError(f"Unhandled split {args.split}")

    summary_path = dst / "convert_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {len(summaries)} episodes under {dst}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
