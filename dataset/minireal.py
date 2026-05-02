# Copyright (2024) Bytedance Ltd. and/or its affiliates
"""MiniReal dataset: same JSON/video layout as RT-1 / Bridge (Dataset_3D)."""

from dataset.dataset_3D import Dataset_3D


class Dataset_MiniReal(Dataset_3D):
    """Episode annotations produced by scripts/convert_minireal_to_irasim.py."""

    pass
