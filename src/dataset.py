from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DataConfig
from utils_laz import (
    apply_xy_flips,
    build_point_features,
    compute_bh_mask,
    compute_origin_xy,
    estimate_ground_z_quantile,
    read_laz_xyz,
    sample_points_bh_aware,
)


class TreeDbhDataset(Dataset):
    """
    Generic dataset built from a provided sample list.
    """

    def __init__(
        self,
        samples: List[Dict],
        cfg: DataConfig,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.cfg = cfg
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        if len(self.samples) == 0:
            raise RuntimeError("Dataset received zero samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        tree_path = sample["tree_path"]

        xyz_abs = read_laz_xyz(tree_path)
        if xyz_abs.shape[0] == 0:
            raise ValueError(f"Empty point cloud in file: {tree_path}")

        z_ground = estimate_ground_z_quantile(
            xyz_abs[:, 2],
            percentile=self.cfg.ground_percentile,
        )

        z_rel = xyz_abs[:, 2] - z_ground

        bh_mask = compute_bh_mask(
            z_rel=z_rel,
            bh_min_z=self.cfg.bh_min_z,
            bh_max_z=self.cfg.bh_max_z,
        )

        origin_x, origin_y, origin_mode = compute_origin_xy(
            xyz_abs=xyz_abs,
            bh_mask=bh_mask,
            min_bh_points_for_origin=self.cfg.min_bh_points_for_origin,
        )

        x_local_target = sample["x_abs"] - origin_x
        y_local_target = sample["y_abs"] - origin_y
        target = np.array(
            [x_local_target, y_local_target, sample["dbh_m"]],
            dtype=np.float32,
        )

        features = build_point_features(
            xyz_abs=xyz_abs,
            origin_x=origin_x,
            origin_y=origin_y,
            z_ground=z_ground,
            bh_mask=bh_mask,
            use_is_bh_window=self.cfg.use_is_bh_window,
        )

        sampled_features, sampled_idx = sample_points_bh_aware(
            features=features,
            bh_mask=bh_mask,
            max_points=self.cfg.max_points,
            bh_fraction_cap=self.cfg.bh_fraction_cap,
            rng=self.rng,
        )

        sampled_bh_mask = bh_mask[sampled_idx]

        flip_x = False
        flip_y = False

        if self.augment:
            flip_x = self.cfg.enable_flip_x and (self.rng.random() < self.cfg.p_flip_x)
            flip_y = self.cfg.enable_flip_y and (self.rng.random() < self.cfg.p_flip_y)

            sampled_features, target = apply_xy_flips(
                features=sampled_features,
                target=target,
                flip_x=flip_x,
                flip_y=flip_y,
            )

        meta = {
            "tree_path": str(tree_path),
            "pred_instance": sample["pred_instance"],
            "x_abs": sample["x_abs"],
            "y_abs": sample["y_abs"],
            "dbh_m": sample["dbh_m"],
            "origin_x": origin_x,
            "origin_y": origin_y,
            "origin_mode": origin_mode,
            "z_ground": z_ground,
            "n_points_raw": int(xyz_abs.shape[0]),
            "n_points_sampled": int(sampled_features.shape[0]),
            "n_bh_points_raw": int(bh_mask.sum()),
            "n_bh_points_sampled": int(sampled_bh_mask.sum()),
            "flip_x": flip_x,
            "flip_y": flip_y,
        }

        return {
            "points": torch.from_numpy(sampled_features),  # [N, F]
            "target": torch.from_numpy(target),            # [3]
            "meta": meta,
        }