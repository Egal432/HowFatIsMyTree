import argparse
import traceback
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import DataConfig, ExperimentConfig, ModelConfig
from dataset import TreeDbhDataset
from models import PointNetConvRegressor, PointNetRegressorOriginal
from splits import build_all_splits
from utils_laz import recover_absolute_xy


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
    elif model_name == "pointnet_conv":
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
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict DBH and export enriched LAZ files")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--trees_dir", required=True,
                        help="Folder with .laz files")
    parser.add_argument("--labels_csv", required=True,
                        help="CSV with full_id, x, y, diameter_m")
    parser.add_argument("--output_dir", default="out/predictions_las")
    parser.add_argument("--max_points", type=int, default=1024)
    parser.add_argument("--ground_percentile", type=float, default=2.0)
    parser.add_argument("--use_is_bh_window",
                        action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg_for_model = DataConfig(use_is_bh_window=args.use_is_bh_window)
    model_cfg = ModelConfig(
        model_name="pointnet_conv",
        out_channels=3,
        pointconv_hidden_dim=128,
        pointconv_layers=3,
        pointconv_k=32,
        pointconv_head_dims=[256, 128],
        pointconv_use_bn=True,
        pointconv_dropout=0.1,
    )
    model = build_model(ExperimentConfig(
        data=data_cfg_for_model, model=model_cfg)).to(device)

    state_dict = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    data_cfg = DataConfig(
        train_trees_dir=Path(args.trees_dir),
        train_labels_csv=Path(args.labels_csv),
        test_trees_dir=None,
        test_labels_csv=None,
        max_points=args.max_points,
        ground_percentile=args.ground_percentile,
        use_is_bh_window=args.use_is_bh_window,
        n_splits=2,
    )

    split_bundle = build_all_splits(data_cfg)
    ds_test = TreeDbhDataset(
        samples=split_bundle["train_samples"],
        cfg=data_cfg,
        augment=False,
        seed=42,
    )

    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(ds_test)} samples. Predicting...")

    errors = 0
    results = []

    with torch.no_grad():
        for batch in dl_test:
            points = batch["points"].to(device)
            meta = batch["meta"]

            pred = model(points)

            pred_x_local = float(pred[0, 0].cpu().item())
            pred_y_local = float(pred[0, 1].cpu().item())
            pred_dbh = float(pred[0, 2].cpu().item())

            origin_x = float(meta["origin_x"][0])
            origin_y = float(meta["origin_y"][0])
            z_ground = float(meta["z_ground"][0])
            z_bh = z_ground + 1.3  # absolute Z at breast height

            pred_instance = meta["pred_instance"][0]
            tree_path = str(meta["tree_path"][0])
            gt_dbh = float(meta["dbh_m"][0])
            gt_x = float(meta["x_abs"][0])
            gt_y = float(meta["y_abs"][0])

            pred_x_abs, pred_y_abs = recover_absolute_xy(
                pred_x_local, pred_y_local, origin_x, origin_y)

            err_dbh_cm = abs(pred_dbh - gt_dbh) * 100
            err_xy_m = float(np.sqrt((pred_x_abs - gt_x) **
                             2 + (pred_y_abs - gt_y)**2))

            results.append({
                "pred_instance": pred_instance,
                "gt_dbh_cm": round(gt_dbh * 100, 2),
                "pred_dbh_cm": round(pred_dbh * 100, 2),
                "err_dbh_cm": round(err_dbh_cm, 2),
                "gt_x": round(gt_x, 3),
                "gt_y": round(gt_y, 3),
                "pred_x_abs": round(pred_x_abs, 3),
                "pred_y_abs": round(pred_y_abs, 3),
                "err_xy_m": round(err_xy_m, 3),
                "z_ground": round(z_ground, 3),
                "z_bh": round(z_bh, 3),
            })

            print(f"{pred_instance}: Pred={pred_dbh*100:.1f}cm | GT={gt_dbh*100:.1f}cm | err_dbh={err_dbh_cm:.1f}cm | err_xy={err_xy_m*100:.1f}cm")

            try:
                las = laspy.read(tree_path)
                n_pts = len(las.points)

                las.add_extra_dim(laspy.ExtraBytesParams(
                    name="pred_dbh", type=np.float64))
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name="pred_x_abs", type=np.float64))
                las.add_extra_dim(laspy.ExtraBytesParams(
                    name="pred_y_abs", type=np.float64))

                las.pred_dbh = np.full(n_pts, pred_dbh, dtype=np.float64)
                las.pred_x_abs = np.full(n_pts, pred_x_abs, dtype=np.float64)
                las.pred_y_abs = np.full(n_pts, pred_y_abs, dtype=np.float64)

                out_path = out_dir / f"{pred_instance}_pred.laz"
                las.write(str(out_path))

            except Exception as e:
                errors += 1
                print(f"Error saving {tree_path}: {e}")
                traceback.print_exc()

    # Save per-tree results CSV
    df = pd.DataFrame(results)
    csv_path = out_dir / "predictions.csv"
    df.to_csv(str(csv_path), index=False)

    # Save CloudCompare-compatible CSV (one point per tree at breast height)
    cc_rows = []
    for r in results:
        cc_rows.append({
            "X": r["pred_x_abs"],
            "Y": r["pred_y_abs"],
            "Z": r["z_bh"],
            "pred_dbh_cm": r["pred_dbh_cm"],
            "gt_dbh_cm": r["gt_dbh_cm"],
            "err_dbh_cm": r["err_dbh_cm"],
            "err_xy_m": r["err_xy_m"],
        })

    cc_df = pd.DataFrame(cc_rows)
    cc_path = out_dir / "predictions_cloudcompare.csv"
    cc_df.to_csv(str(cc_path), index=False)

    # Print summary
    print(f"\n=== Summary ({len(df)} trees) ===")
    print(f"MAE DBH:     {df['err_dbh_cm'].mean():.2f} cm")
    print(f"RMSE DBH:    {np.sqrt((df['err_dbh_cm']**2).mean()):.2f} cm")
    print(f"MAE XY:      {df['err_xy_m'].mean()*100:.2f} cm")
    print(
        f"Max err DBH: {df['err_dbh_cm'].max():.2f} cm  ({df.loc[df['err_dbh_cm'].idxmax(), 'pred_instance']})")
    print(f"\nPer-tree results:     {csv_path}")
    print(f"CloudCompare file:    {cc_path}")
    print(
        f"LAZ files:            {out_dir}  ({len(df) - errors}/{len(df)} successful)")


if __name__ == "__main__":
    main()
