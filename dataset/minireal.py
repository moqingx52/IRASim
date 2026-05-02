# Copyright (2024) Bytedance Ltd. and/or its affiliates
"""MiniReal dataset: same JSON/video layout as RT-1 / Bridge (Dataset_3D), joint-space actions."""

import numpy as np
import torch

from dataset.dataset_3D import Dataset_3D
from scripts.minireal_action_util import states_to_delta_actions


class Dataset_MiniReal(Dataset_3D):
    """Episode annotations from scripts/convert_minireal_to_irasim.py; actions are joint deltas."""

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = states_to_delta_actions(
            np.asarray(arm_states, dtype=np.float32),
            np.asarray(gripper_states, dtype=np.float32),
            accumulate_action,
        )
        return torch.from_numpy(action)
