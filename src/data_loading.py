# data_loading.py

from typing import Dict, Tuple
from dataset import TreeDbhDataset


def get_fold_datasets(split_bundle: Dict, fold_id: int, cfg, seed: int = 42):
    train_samples = split_bundle["train_samples"]
    train_idx, val_idx = split_bundle["cv_folds"][fold_id]

    fold_train_samples = [train_samples[i] for i in train_idx]
    fold_val_samples = [train_samples[i] for i in val_idx]

    ds_train = TreeDbhDataset(
        samples=fold_train_samples,
        cfg=cfg,
        augment=True,
        seed=seed + fold_id,
    )
    ds_val = TreeDbhDataset(
        samples=fold_val_samples,
        cfg=cfg,
        augment=False,
        seed=seed + fold_id,
    )
    return ds_train, ds_val


def get_external_test_dataset(split_bundle: Dict, cfg, seed: int = 999):
    test_samples = split_bundle["test_samples"]
    if test_samples is None:
        return None

    return TreeDbhDataset(
        samples=test_samples,
        cfg=cfg,
        augment=False,
        seed=seed,
    )