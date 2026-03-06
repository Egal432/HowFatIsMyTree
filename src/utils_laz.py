from __future__ import annotations

from pathlib import Path
from typing import Tuple

import laspy
import numpy as np


def read_laz_xyz(path: Path) -> np.ndarray:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    return xyz


def estimate_ground_z_quantile(z: np.ndarray, percentile: float = 2.0) -> float:
    return float(np.percentile(z, percentile))


def compute_bh_mask(
    z_rel: np.ndarray,
    bh_min_z: float,
    bh_max_z: float,
) -> np.ndarray:
    return (z_rel >= bh_min_z) & (z_rel <= bh_max_z)


def compute_origin_xy(
    xyz_abs: np.ndarray,
    bh_mask: np.ndarray,
    min_bh_points_for_origin: int = 20,
) -> Tuple[float, float, str]:
    if int(bh_mask.sum()) >= min_bh_points_for_origin:
        xy = xyz_abs[bh_mask, :2]
        mode = "bh_centroid"
    else:
        xy = xyz_abs[:, :2]
        mode = "tree_centroid"

    origin_x = float(np.mean(xy[:, 0]))
    origin_y = float(np.mean(xy[:, 1]))
    return origin_x, origin_y, mode


def build_point_features(
    xyz_abs: np.ndarray,
    origin_x: float,
    origin_y: float,
    z_ground: float,
    bh_mask: np.ndarray,
    use_is_bh_window: bool = True,
) -> np.ndarray:
    x_local = xyz_abs[:, 0] - origin_x
    y_local = xyz_abs[:, 1] - origin_y
    z_rel = xyz_abs[:, 2] - z_ground

    feats = [x_local[:, None], y_local[:, None], z_rel[:, None]]

    if use_is_bh_window:
        feats.append(bh_mask.astype(np.float32)[:, None])

    return np.concatenate(feats, axis=1).astype(np.float32)


def sample_points_bh_aware(
    features: np.ndarray,
    bh_mask: np.ndarray,
    max_points: int,
    bh_fraction_cap: float = 0.5,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    n_points = features.shape[0]
    all_idx = np.arange(n_points)
    bh_idx = all_idx[bh_mask]
    non_bh_idx = all_idx[~bh_mask]

    bh_cap = int(round(max_points * bh_fraction_cap))

    if len(bh_idx) <= bh_cap:
        chosen_bh = bh_idx
    else:
        chosen_bh = rng.choice(bh_idx, size=bh_cap, replace=False)

    remaining = max_points - len(chosen_bh)

    if remaining > 0:
        if len(non_bh_idx) >= remaining:
            chosen_non_bh = rng.choice(non_bh_idx, size=remaining, replace=False)
        elif len(non_bh_idx) > 0:
            chosen_non_bh = rng.choice(non_bh_idx, size=remaining, replace=True)
        else:
            chosen_non_bh = np.array([], dtype=np.int64)
    else:
        chosen_non_bh = np.array([], dtype=np.int64)

    chosen = np.concatenate([chosen_bh, chosen_non_bh])

    if len(chosen) < max_points:
        pad = rng.choice(all_idx, size=max_points - len(chosen), replace=True)
        chosen = np.concatenate([chosen, pad])

    if len(chosen) > max_points:
        chosen = rng.choice(chosen, size=max_points, replace=False)

    rng.shuffle(chosen)
    return features[chosen], chosen.astype(np.int64)


def apply_xy_flips(
    features: np.ndarray,
    target: np.ndarray,
    flip_x: bool,
    flip_y: bool,
):
    features = features.copy()
    target = target.copy()

    if flip_x:
        features[:, 0] *= -1.0
        target[0] *= -1.0

    if flip_y:
        features[:, 1] *= -1.0
        target[1] *= -1.0

    return features, target


def recover_absolute_xy(
    x_local_pred: float,
    y_local_pred: float,
    origin_x: float,
    origin_y: float,
):
    return x_local_pred + origin_x, y_local_pred + origin_y


def extract_predinstance_from_filename(path: Path) -> str:
    return path.stem.split("_")[-1]