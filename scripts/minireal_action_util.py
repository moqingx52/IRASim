# Copyright (2024) ByteDance-style utilities for MiniReal <-> IRASim action alignment.
"""Helpers to map MiniReal CSV rows to IRASim Dataset_3D state/actions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def parse_csv_numeric(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read CSV with header; return header row and float matrix (skips empty rows)."""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    header = rows[0]
    data_rows = [r for r in rows[1:] if r]
    if not data_rows:
        raise ValueError(f"No data rows: {path}")
    data = np.asarray(data_rows, dtype=np.float64)
    return header, data


def joint_rows_to_state6_gripper(joint_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    joint_data: [T, C] with first column index (Unnamed: 0), then joint columns.
    Uses left arm joints 1-6 as IRASim state[:, :6] (matches Dataset_3D slicing).
    Gripper: mean of left finger columns (last 5 before right hand in MiniReal export), scaled.
    Column layout from train/1_1/joint.txt:
      0: index
      1-14: arm joints (7 left + 7 right)
      15-26: finger/aux positions
    """
    if joint_data.ndim != 2 or joint_data.shape[1] < 27:
        raise ValueError(f"joint matrix expected >=27 cols, got {joint_data.shape}")
    q_left = joint_data[:, 1:7].astype(np.float64)
    # fingers: columns 15..19 left (5), column 20 right thumb0 etc. — use left 5 as gripper proxy
    fingers = joint_data[:, 15:20].astype(np.float64)
    g = np.clip(fingers.mean(axis=1) / 2000.0, 0.0, 1.0)
    return q_left, g


def states_to_delta_actions(
    arm_states: np.ndarray,
    gripper_states: np.ndarray,
    accumulate_action: bool = False,
) -> np.ndarray:
    """
    MiniReal 左臂状态为**关节角** [T,6]，不能用 Dataset_3D 的末端 xyz+rpy 相对运动。
    返回 [T-1, 7]：关节差分（或相对首帧的关节差）+ 当前步 gripper；训练时再乘 c_act_scaler。
    """
    assert arm_states.shape[0] == gripper_states.shape[0]
    t = arm_states.shape[0]
    action = np.zeros((t - 1, 7), dtype=np.float32)
    if accumulate_action:
        action[:, :6] = (arm_states[1:, :6] - arm_states[0:1, :6]).astype(np.float32)
    else:
        action[:, :6] = arm_states[1:, :6] - arm_states[:-1, :6]
    action[:, 6] = gripper_states[1:]
    return action


def joint_matrix_to_arm_states(joint_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Full joint CSV numeric matrix -> arm_states [T,6], continuous_gripper [T]."""
    q_left, g = joint_rows_to_state6_gripper(joint_data)
    return q_left.astype(np.float32), g.astype(np.float32)


def rdt_action_row_to_arm_gripper(row26: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    One row of shape [26] aligned with action_gt columns (no index column).
    Columns 0:6 = left arm joints 1-6; finger slice matches joint CSV layout at indices 14:19.
    """
    row = np.asarray(row26, dtype=np.float64).reshape(-1)
    if row.shape[0] < 26:
        raise ValueError(f"Expected 26-dim row, got {row.shape}")
    arm = row[0:6].astype(np.float64)
    fingers = row[14:19]
    g = float(np.clip(fingers.mean() / 2000.0, 0.0, 1.0))
    return arm.astype(np.float32), g


def build_arm_trajectory_from_test_and_rdt(
    joint_row_t15: np.ndarray,
    rdt_51x26: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    joint_row_t15: full CSV row vector including index column for frame 15.
    rdt_51x26: [51, 26] future commanded configs (absolute).
    Returns arm_states [52,6], gripper [52].
    """
    jmat = joint_row_t15.reshape(1, -1)
    a0, g0 = joint_rows_to_state6_gripper(jmat)
    arm_list = [a0[0]]
    grip_list = [float(g0[0])]
    for i in range(rdt_51x26.shape[0]):
        arm, g = rdt_action_row_to_arm_gripper(rdt_51x26[i])
        arm_list.append(arm)
        grip_list.append(g)
    arm_states = np.stack(arm_list, axis=0).astype(np.float32)
    gripper = np.asarray(grip_list, dtype=np.float32)
    return arm_states, gripper


def pick_instruction(episode_dir: Path) -> str:
    for name in ("instruction.txt", "instructions.txt"):
        p = episode_dir / name
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"No instruction file in {episode_dir}")
