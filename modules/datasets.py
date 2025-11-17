# -*- coding: utf-8 -*-
"""
modules/datasets.py — загрузка подготовленных тензоров и датасет для обучения.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset

class WindowsDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, target_ids: torch.Tensor, sec_ids: torch.Tensor):
        assert input_ids.shape[0] == target_ids.shape[0] == sec_ids.shape[0]
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.sec_ids = sec_ids
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx], self.sec_ids[idx]

@dataclass
class PreparedSet:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    sec_ids: torch.Tensor

def load_prepared_tensors(path: Path) -> PreparedSet:
    data = torch.load(path)
    return PreparedSet(
        input_ids=data['input_ids'],
        target_ids=data['target_ids'],
        sec_ids=data['sec_ids'],
    )
