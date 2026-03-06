from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from data_loading import get_fold_datasets
from logger_utils import LocalLogger, WandbLogger
from losses import CoupledXYDbhLoss
from models import PointNetConvRegressor, PointNetRegressorOriginal
from splits import build_all_splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def move_batch_to_device(batch: dict, device: torch.device):
    points = batch["points"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    return points, target, batch["meta"]


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    dx = pred[:, 0] - target[:, 0]
    dy = pred[:, 1] - target[:, 1]
    dxy = torch.sqrt(dx.square() + dy.square() + 1e-12)

    dbh_abs = torch.abs(pred[:, 2] - target[:, 2])

    rmse_x = torch.sqrt(torch.mean(dx.square()))
    rmse_y = torch.sqrt(torch.mean(dy.square()))
    rmse_dbh = torch.sqrt(torch.mean((pred[:, 2] - target[:, 2]).square()))

    return {
        "mae_xy_radial": float(dxy.mean().item()),
        "mae_dbh": float(dbh_abs.mean().item()),
        "rmse_x": float(rmse_x.item()),
        "rmse_y": float(rmse_y.item()),
        "rmse_dbh": float(rmse_dbh.item()),
    }


def build_model(exp_cfg: ExperimentConfig):
    in_channels = 4 if exp_cfg.data.use_is_bh_window else 3
    model_name = exp_cfg.model.model_name

    if model_name == "pointnet_original_light":
        return PointNetRegressorOriginal(
            in_channels=in_channels,
            out_channels=exp_cfg.model.out_channels,
            feat_dims=exp_cfg.model.pointnet_feat_dims,
            head_dims=exp_cfg.model.pointnet_head_dims,
            use_bn=exp_cfg.model.pointnet_use_bn,
            dropout=exp_cfg.model.pointnet_dropout,
            use_input_transform=False,
            use_feature_transform=False,
        )

    if model_name == "pointnet_original_full":
        return PointNetRegressorOriginal(
            in_channels=in_channels,
            out_channels=exp_cfg.model.out_channels,
            feat_dims=exp_cfg.model.pointnet_feat_dims,
            head_dims=exp_cfg.model.pointnet_head_dims,
            use_bn=exp_cfg.model.pointnet_use_bn,
            dropout=exp_cfg.model.pointnet_dropout,
            use_input_transform=exp_cfg.model.pointnet_use_input_transform,
            use_feature_transform=exp_cfg.model.pointnet_use_feature_transform,
        )

    if model_name == "pointnet_conv":
        return PointNetConvRegressor(
            in_channels=in_channels,
            out_channels=exp_cfg.model.out_channels,
            hidden_dim=exp_cfg.model.pointconv_hidden_dim,
            layers=exp_cfg.model.pointconv_layers,
            k=exp_cfg.model.pointconv_k,
            head_dims=exp_cfg.model.pointconv_head_dims,
            use_bn=exp_cfg.model.pointconv_use_bn,
            dropout=exp_cfg.model.pointconv_dropout,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        points, target, _ = move_batch_to_device(batch, device)

        if training:
            optimizer.zero_grad()

        pred = model(points)
        loss = criterion(pred, target)

        if training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * points.size(0)
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)
    components = criterion.components(all_preds, all_targets)

    metrics.update(components)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def run_cross_validation(exp_cfg: ExperimentConfig) -> None:
    set_seed(exp_cfg.train.seed)
    device = get_device(exp_cfg.train.device)
    output_dir = exp_cfg.train.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_bundle = build_all_splits(exp_cfg.data)
    cv_summary = []

    print(f"Using device: {device}")

    for fold_id in range(exp_cfg.data.n_splits):
        print(f"\n=== Fold {fold_id + 1}/{exp_cfg.data.n_splits} ===")

        fold_dir = output_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        local_logger = LocalLogger(fold_dir)
        local_logger.log_config(exp_cfg)

        wandb_logger = WandbLogger(exp_cfg, fold_id=fold_id)
        wandb_logger.init()

        ds_train, ds_val = get_fold_datasets(
            split_bundle,
            fold_id=fold_id,
            cfg=exp_cfg.data,
            seed=exp_cfg.train.seed,
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=exp_cfg.train.batch_size,
            shuffle=True,
            num_workers=exp_cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=exp_cfg.train.batch_size,
            shuffle=False,
            num_workers=exp_cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(exp_cfg).to(device)
        wandb_logger.watch_model(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=exp_cfg.train.lr,
            weight_decay=exp_cfg.train.weight_decay,
        )

        criterion = CoupledXYDbhLoss(
            xy_weight=exp_cfg.train.xy_weight,
            dbh_weight=exp_cfg.train.dbh_weight,
            smooth_l1_beta=exp_cfg.train.smooth_l1_beta,
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = -1

        for epoch in range(1, exp_cfg.train.epochs + 1):
            train_metrics = run_one_epoch(model, dl_train, criterion, device, optimizer=optimizer)
            val_metrics = run_one_epoch(model, dl_val, criterion, device, optimizer=None)

            row = {
                "fold": fold_id,
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            local_logger.log_metrics(row)

            wandb_row = {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_xy_loss_unweighted": train_metrics["xy_loss_unweighted"],
                "train_xy_loss_weighted": train_metrics["xy_loss_weighted"],
                "train_xy_mean_euclidean_error": train_metrics["xy_mean_euclidean_error"],
                "train_dbh_loss_unweighted": train_metrics["dbh_loss_unweighted"],
                "train_dbh_loss_weighted": train_metrics["dbh_loss_weighted"],
                "train_mae_xy_radial": train_metrics["mae_xy_radial"],
                "train_mae_dbh": train_metrics["mae_dbh"],
                "val_loss": val_metrics["loss"],
                "val_xy_loss_unweighted": val_metrics["xy_loss_unweighted"],
                "val_xy_loss_weighted": val_metrics["xy_loss_weighted"],
                "val_xy_mean_euclidean_error": val_metrics["xy_mean_euclidean_error"],
                "val_dbh_loss_unweighted": val_metrics["dbh_loss_unweighted"],
                "val_dbh_loss_weighted": val_metrics["dbh_loss_weighted"],
                "val_mae_xy_radial": val_metrics["mae_xy_radial"],
                "val_mae_dbh": val_metrics["mae_dbh"],
                "val_rmse_dbh": val_metrics["rmse_dbh"],
            }
            wandb_logger.log(wandb_row, step=epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.5f} | "
                f"val_loss={val_metrics['loss']:.5f} | "
                f"val_xy={val_metrics['mae_xy_radial']:.4f} m | "
                f"val_dbh={val_metrics['mae_dbh']:.4f} m"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        if exp_cfg.train.save_best_model and best_state is not None:
            torch.save(best_state, fold_dir / "best_model.pt")

        best_summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }

        if best_state is not None:
            best_model = build_model(exp_cfg).to(device)
            best_model.load_state_dict(best_state)
            best_val_metrics = run_one_epoch(best_model, dl_val, criterion, device, optimizer=None)
            best_summary.update(best_val_metrics)

        with open(fold_dir / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        wandb_logger.summary(best_summary)
        wandb_logger.finish()

        cv_summary.append({"fold": fold_id, **best_summary})

    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n=== CV summary ===")
    for row in cv_summary:
        print(
            f"Fold {row['fold']:02d} | "
            f"best_epoch={row['best_epoch']} | "
            f"val_xy={row.get('mae_xy_radial', float('nan')):.4f} m | "
            f"val_dbh={row.get('mae_dbh', float('nan')):.4f} m"
        )


if __name__ == "__main__":
    exp_cfg = ExperimentConfig(
        data=DataConfig(
            train_trees_dir=Path("TestTrees/Trees"),
            train_labels_csv=Path("TestTrees/labels.csv"),
            test_trees_dir=None,
            test_labels_csv=None,
            max_points=512,
            n_splits=2,
        ),
        model=ModelConfig(
            model_name="pointnet_original_light",
        ),
        train=TrainConfig(
            output_dir=Path("outputs/debug_run"),
            batch_size=2,
            epochs=2,
            lr=1e-3,
            weight_decay=1e-4,
            seed=42,
            use_wandb=False,
            xy_weight=1.0,
            dbh_weight=1.0,
        ),
    )

    run_cross_validation(exp_cfg)