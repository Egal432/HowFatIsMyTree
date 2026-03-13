from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import laspy
import pandas as pd

from config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from data_loading import get_fold_datasets, get_external_test_dataset
from logger_utils import LocalLogger, WandbLogger
from losses import CoupledXYDbhLoss
from models import PointNetConvRegressor, PointNetRegressorOriginal
from splits import build_all_splits
from utils_laz import recover_absolute_xy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


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


def run_inference_and_export(
    exp_cfg: ExperimentConfig,
    best_model_checkpoint: Path
) -> None:
    """
    Loads the best model, runs inference on test data, and exports .laz files
    with prediction scalar fields for CloudCompare.
    """
    device = get_device(exp_cfg.train.device)
    print(f"\n=== Running Test Inference ===")
    print(f"Using checkpoint: {best_model_checkpoint}")

    # 1. Prepare Test Data
    # Re-use the split builder to ensure we process files exactly like training
    split_bundle = build_all_splits(exp_cfg.data)
    ds_test = get_external_test_dataset(split_bundle, exp_cfg.data, seed=999)

    if ds_test is None:
        print("No test data found (test_trees_dir is empty or config is None). Skipping inference.")
        return

    dl_test = DataLoader(
        ds_test,
        batch_size=32,  # Higher batch size usually fine for inference (fast)
        shuffle=False,
        num_workers=exp_cfg.train.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Load Model
    model = build_model(exp_cfg).to(device)
    state_dict = torch.load(best_model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = exp_cfg.train.output_dir / "predictions_las"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Inference Loop
    print(f"Found {len(ds_test)} test samples. Predicting...")
    with torch.no_grad():
        for batch in dl_test:
            # Dataset returns sampled features
            points = batch["points"].to(device)
            meta = batch["meta"]

            # Predict
            pred = model(points)

            batch_size = pred.shape[0]
            for i in range(batch_size):
                m = meta[i]
                pred_val = pred[i]

                pred_x_local = pred_val[0].item()
                pred_y_local = pred_val[1].item()
                pred_dbh = pred_val[2].item()

                # Recover Absolute Coords
                pred_x_abs, pred_y_abs = recover_absolute_xy(
                    pred_x_local, pred_y_local, m["origin_x"], m["origin_y"]
                )

                # Load ORIGINAL full-resolution LAS for visualization
                laz_path = Path(m["tree_path"])
                try:
                    las = laspy.read(laz_path)

                    # Add Extra Dimensions
                    las.add_extra_dim(laspy.ExtraDimsParams("pred_dbh", "f8"))
                    las.add_extra_dim(laspy.ExtraDimsParams("pred_x", "f8"))
                    las.add_extra_dim(laspy.ExtraDimsParams("pred_y", "f8"))

                    # Fill with predicted value
                    las.pred_dbh = np.full(las.points.shape, pred_dbh)
                    las.pred_x = np.full(las.points.shape, pred_x_abs)
                    las.pred_y = np.full(las.points.shape, pred_y_abs)

                    # Save
                    out_path = out_dir / laz_path.name
                    las.write(out_path)
                except Exception as e:
                    print(f"Error writing {laz_path.name}: {e}")

    print(f"\nDone. Predictions exported to: {out_dir}")
    print("You can now open files in CloudCompare and color by 'pred_dbh'.")


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


def run_one_epoch(model, loader, criterion, device, optimizer=None, accumulation_steps=1):
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    all_preds = []
    all_targets = []

    if training:
        optimizer.zero_grad()

    for i, batch in enumerate(loader):
        points, target, _ = move_batch_to_device(batch, device)

        pred = model(points)

        # Calculate loss
        loss = criterion(pred, target)

        if training:
            # Normalize loss to account for accumulation to get accurate average gradients
            loss = loss / accumulation_steps
            loss.backward()

            # Detach preds for metrics
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

            # Update running loss (Scale back up to true loss)
            # loss.item() is divided by accumulation_steps, so we multiply back
            running_loss += (loss.item() * points.size(0) * accumulation_steps)

            # STEP THE OPTIMIZER every 'accumulation_steps' batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
               # torch.cuda.empty_cache()
        else:
            # Validation mode
            running_loss += loss.item() * points.size(0)
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

    # Combine all preds/targets for final epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)

    # Calculate final loss components for logging
    # (Use a fresh criterion instance to ensure clean calculations on the whole epoch batch)
    final_loss_criterion = type(criterion)(
        xy_weight=criterion.xy_weight,
        dbh_weight=criterion.dbh_weight,
        smooth_l1_beta=criterion.smooth_l1_beta
    )
    final_loss_items = final_loss_criterion.components(all_preds, all_targets)

    # Average loss over the dataset
    metrics["loss"] = running_loss / len(loader.dataset)
    metrics.update(final_loss_items)

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

        # 1. Get Datasets
        ds_train, ds_val = get_fold_datasets(
            split_bundle,
            fold_id=fold_id,
            cfg=exp_cfg.data,
            seed=exp_cfg.train.seed,
        )

        # 2. Create Dataloaders
        dl_train = DataLoader(
            ds_train,
            batch_size=exp_cfg.train.batch_size,
            shuffle=True,
            num_workers=exp_cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(exp_cfg.train.num_workers > 0),
            prefetch_factor=4 if exp_cfg.train.num_workers > 0 else None,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=exp_cfg.train.batch_size,
            shuffle=False,
            num_workers=exp_cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(exp_cfg.train.num_workers > 0),
            prefetch_factor=4 if exp_cfg.train.num_workers > 0 else None,
        )

        # 3. Setup Model
        model = build_model(exp_cfg).to(device)
        wandb_logger.watch_model(model)

        # 4. Setup Optimizer and Criterion
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
        patience = getattr(exp_cfg.train, "early_stopping_patience", 30)
        epochs_without_improvement = 0

        # ACCUMULATION STEPS
        accumulation_steps = 4

        # 5. Training Loop
        for epoch in range(1, exp_cfg.train.epochs + 1):
            train_metrics = run_one_epoch(
                model, dl_train, criterion, device, optimizer=optimizer,
                accumulation_steps=accumulation_steps
            )
            val_metrics = run_one_epoch(
                model, dl_val, criterion, device, optimizer=None
            )

            row = {
                "fold": fold_id,
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            local_logger.log_metrics(row)

            wandb_logger.log({
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_mae_xy_radial": val_metrics["mae_xy_radial"],
                "val_mae_dbh": val_metrics["mae_dbh"],
            }, step=epoch)

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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience {patience})")
                break

        if exp_cfg.train.save_best_model and best_state is not None:
            torch.save(best_state, fold_dir / "best_model.pt")

        # 6. Save Fold Summary
        best_summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }

        if best_state is not None:
            best_model = build_model(exp_cfg).to(device)
            best_model.load_state_dict(best_state)
            best_val_metrics = run_one_epoch(
                best_model, dl_val, criterion, device, optimizer=None)
            best_summary.update(best_val_metrics)

        with open(fold_dir / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        wandb_logger.summary(best_summary)
        wandb_logger.finish()
        cv_summary.append({"fold": fold_id, **best_summary})

    # 7. Global CV Summary
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

    # 8. Run Inference on Best Model
    best_fold_row = min(
        cv_summary, key=lambda x: x.get('mae_dbh', float('inf')))
    best_fold_id = best_fold_row['fold']
    best_checkpoint_path = output_dir / \
        f"fold_{best_fold_id:02d}" / "best_model.pt"

    print(f"Selected best checkpoint: {best_checkpoint_path}")
    run_inference_and_export(exp_cfg, best_checkpoint_path)


if __name__ == "__main__":
    # Make sure Set 'test_trees_dir' and 'test_labels_csv' to None if you have no DBH for test data.
    # If you have test data, point them to correct paths.
    exp_cfg = ExperimentConfig(
        data=DataConfig(
            # Corrected: Folder with LAZ
            train_trees_dir=Path("out/ecosense/final_trees"),
            # Corrected: CSV file
            train_labels_csv=Path("out/ecosense/final_trees.csv"),
            # Set paths here if you want inference at the end
            test_trees_dir=Path(
                "/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees_valid"),
            test_labels_csv=Path(
                "/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees_valid/dbh_valid_fixed.csv"),
            max_points=2048,
            n_splits=5,            # Reduced to 5 for robustness and speed
            bh_fraction_cap=0.5,
            use_is_bh_window=False,
        ),
        model=ModelConfig(
            model_name="pointnet_conv",
            pointconv_k=32,
            pointconv_layers=3,
            pointconv_hidden_dim=128,
            pointconv_head_dims=[256, 128],
            pointconv_dropout=0.1,
            pointconv_use_bn=True,
        ),
        train=TrainConfig(
            output_dir=Path("outputs/Last_Run"),
            batch_size=8,
            num_workers=8,
            epochs=300,
            early_stopping_patience=30,  # Requires this field in TrainConfig!
            lr=1e-3,
            weight_decay=1e-4,
            seed=42,
            use_wandb=True,
            wandb_project="tree_dbh_test",
            wandb_entity="skreidl",
            wandb_run_name="Last_Try",
            xy_weight=1.0,
            dbh_weight=1.0,
            smooth_l1_beta=1.0,
            save_best_model=True,
            save_history=True,
        ),
    )

    run_cross_validation(exp_cfg)
