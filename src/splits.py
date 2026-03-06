from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import DataConfig
from utils_laz import extract_predinstance_from_filename


def load_labels_table(
    labels_csv: Path,
    id_col: str,
    x_col: str,
    y_col: str,
    dbh_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    required = [id_col, x_col, y_col, dbh_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {missing}")

    df = df[required].copy()
    df[id_col] = df[id_col].astype(str)
    return df


def build_sample_index(
    trees_dir: Path,
    labels_csv: Path,
    cfg: DataConfig,
) -> List[Dict]:
    labels_df = load_labels_table(
        labels_csv=labels_csv,
        id_col=cfg.id_col,
        x_col=cfg.x_col,
        y_col=cfg.y_col,
        dbh_col=cfg.dbh_col,
    )

    label_map = {
        str(row[cfg.id_col]).split(".")[0]: row
        for _, row in labels_df.iterrows()
    }

    tree_files = sorted(list(trees_dir.glob("*.laz")) + list(trees_dir.glob("*.las")))
    samples: List[Dict] = []


    for tree_path in tree_files:
        pred_instance = extract_predinstance_from_filename(tree_path)



        if pred_instance not in label_map:
            if cfg.ignore_unmatched_files:
                continue
            raise KeyError(f"No label match for tree file: {tree_path.name}")

        row = label_map[pred_instance]
        samples.append(
            {
                "tree_path": tree_path,
                "pred_instance": pred_instance,
                "x_abs": float(row[cfg.x_col]),
                "y_abs": float(row[cfg.y_col]),
                "dbh_m": float(row[cfg.dbh_col]),
            }
        )

    return samples


def make_cv_folds(
    samples: List[Dict],
    n_splits: int,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (train_idx, val_idx) tuples over the given sample list.
    """
    n = len(samples)
    if n < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from only {n} samples."
        )

    indices = np.arange(n)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    folds = []
    for train_idx, val_idx in kf.split(indices):
        folds.append((train_idx, val_idx))
    return folds


def build_all_splits(cfg: DataConfig) -> Dict:
    """
    Build:
    - full train sample index
    - CV folds over train
    - optional external test sample index
    """
    train_samples = build_sample_index(
        trees_dir=cfg.train_trees_dir,
        labels_csv=cfg.train_labels_csv,
        cfg=cfg,
    )

    folds = make_cv_folds(
        samples=train_samples,
        n_splits=cfg.n_splits,
        seed=cfg.cv_seed,
        shuffle=cfg.shuffle_folds,
    )

    test_samples: Optional[List[Dict]] = None
    if cfg.test_trees_dir is not None and cfg.test_labels_csv is not None:
        test_samples = build_sample_index(
            trees_dir=cfg.test_trees_dir,
            labels_csv=cfg.test_labels_csv,
            cfg=cfg,
        )

    return {
        "train_samples": train_samples,
        "cv_folds": folds,
        "test_samples": test_samples,
    }