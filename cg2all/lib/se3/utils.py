"""Small helpers used across the SE(3)-Transformer port. DGL-free."""

from typing import Dict, List

import torch
from torch import Tensor


def aggregate_residual(feats1, feats2, method: str):
    """Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2."""
    if method in ("add", "sum"):
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ("cat", "concat"):
        return {
            k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v
            for k, v in feats2.items()
        }
    raise ValueError("method must be add/sum or cat/concat")


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(
        zip(
            map(str, degrees),
            features.split([degree_to_dim(deg) for deg in degrees], dim=-1),
        )
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
    raise ValueError("Boolean value expected.")
