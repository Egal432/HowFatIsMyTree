from pathlib import Path

from config import DataConfig
from splits import build_all_splits
from dataset import TreeDbhDataset


def main():
    cfg = DataConfig(
        n_splits=2,
        max_points=1024,
    )

    split_bundle = build_all_splits(cfg)

    print("n_train_samples:", len(split_bundle["train_samples"]))
    print("n_folds:", len(split_bundle["cv_folds"]))

    if split_bundle["test_samples"] is not None:
        print("n_test_samples:", len(split_bundle["test_samples"]))

    train_samples = split_bundle["train_samples"]
    train_idx, val_idx = split_bundle["cv_folds"][0]

    fold_train_samples = [train_samples[i] for i in train_idx]
    fold_val_samples = [train_samples[i] for i in val_idx]

    ds_train = TreeDbhDataset(fold_train_samples, cfg=cfg, augment=True)
    ds_val = TreeDbhDataset(fold_val_samples, cfg=cfg, augment=False)

    print("fold0 train size:", len(ds_train))
    print("fold0 val size:", len(ds_val))

    sample = ds_train[0]
    print("points:", sample["points"].shape)
    print("target:", sample["target"])
    print("meta:", sample["meta"])


if __name__ == "__main__":
    main()