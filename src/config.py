from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    # -------------------------
    # Training dataset
    # -------------------------
    train_trees_dir: Path = Path("TestTrees/Trees")
    train_labels_csv: Path = Path("TestTrees/labels.csv")

    # -------------------------
    # Optional external test dataset
    # -------------------------
    test_trees_dir: Optional[Path] = None
    test_labels_csv: Optional[Path] = None

    # -------------------------
    # Label columns
    # -------------------------
    id_col: str = "PredInstance"
    x_col: str = "x"
    y_col: str = "y"
    dbh_col: str = "diameter_m"

    # -------------------------
    # Point features
    # -------------------------
    use_is_bh_window: bool = True

    # -------------------------
    # Ground normalization
    # -------------------------
    ground_percentile: float = 2.0
    terrain_mode: str = "quantile"   # later: "dtm"
    terrain_path: Optional[Path] = None

    # -------------------------
    # Breast-height window
    # -------------------------
    bh_min_z: float = 1.20
    bh_max_z: float = 1.40
    min_bh_points_for_origin: int = 20

    # -------------------------
    # Sampling
    # -------------------------
    max_points: int = 1024
    bh_fraction_cap: float = 0.5

    # -------------------------
    # Augmentation
    # -------------------------
    enable_flip_x: bool = True
    enable_flip_y: bool = True
    p_flip_x: float = 0.5
    p_flip_y: float = 0.5

    # -------------------------
    # Matching / filtering
    # -------------------------
    ignore_unmatched_labels: bool = True
    ignore_unmatched_files: bool = True

    # -------------------------
    # Cross-validation
    # -------------------------
    n_splits: int = 5
    cv_seed: int = 42
    shuffle_folds: bool = True



@dataclass
class ModelConfig:
    model_name: str = "pointnet_original_light"
    in_channels: int = 4
    out_channels: int = 3

    pointnet_feat_dims: list[int] = field(default_factory=lambda: [64, 64, 64, 128, 1024])
    pointnet_head_dims: list[int] = field(default_factory=lambda: [512, 256])
    pointnet_dropout: float = 0.3
    pointnet_use_bn: bool = True
    pointnet_use_input_transform: bool = False
    pointnet_use_feature_transform: bool = False

    pointconv_hidden_dim: int = 64
    pointconv_layers: int = 2
    pointconv_k: int = 16
    pointconv_dropout: float = 0.3
    pointconv_use_bn: bool = True
    pointconv_head_dims: list[int] = field(default_factory=lambda: [128, 64])


@dataclass
class TrainConfig:
    output_dir: Path = Path("outputs/default_run")
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"

    # loss
    xy_weight: float = 1.0
    dbh_weight: float = 1.0
    smooth_l1_beta: float = 1.0

    # logging
    use_wandb: bool = True
    wandb_project: str = "tree_dbh"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "online"   # "online", "offline", "disabled"
    wandb_watch_model: bool = False

    save_history: bool = True
    save_best_model: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)