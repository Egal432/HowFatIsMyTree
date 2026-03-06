from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class LocalLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "history.jsonl"
        self.csv_path = self.output_dir / "history.csv"
        self._csv_initialized = False
        self._csv_fields: list[str] | None = None

    def log_config(self, config_obj: Any) -> None:
        config_dict = _to_serializable(config_obj)
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    def log_metrics(self, row: dict) -> None:
        row = _to_serializable(row)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        flat = flatten_dict(row)
        if not self._csv_initialized:
            self._csv_fields = list(flat.keys())
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields)
                writer.writeheader()
                writer.writerow(flat)
            self._csv_initialized = True
        else:
            assert self._csv_fields is not None
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields)
                writer.writerow({k: flat.get(k, "") for k in self._csv_fields})


class WandbLogger:
    def __init__(self, exp_cfg, fold_id: Optional[int] = None) -> None:
        self.exp_cfg = exp_cfg
        self.fold_id = fold_id
        self.run = None

    def init(self) -> None:
        if not self.exp_cfg.train.use_wandb or self.exp_cfg.train.wandb_mode == "disabled":
            return

        import wandb

        os.environ["WANDB_MODE"] = self.exp_cfg.train.wandb_mode

        config_dict = _to_serializable(self.exp_cfg)
        run_name = self.exp_cfg.train.wandb_run_name
        if run_name is None:
            base = self.exp_cfg.model.model_name
            run_name = f"{base}_fold_{self.fold_id:02d}" if self.fold_id is not None else base

        self.run = wandb.init(
            project=self.exp_cfg.train.wandb_project,
            entity=self.exp_cfg.train.wandb_entity,
            name=run_name,
            tags=self.exp_cfg.train.wandb_tags,
            config=config_dict,
        )

    def watch_model(self, model) -> None:
        if self.run is None:
            return
        if self.exp_cfg.train.wandb_watch_model:
            self.run.watch(model)

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def summary(self, summary_dict: dict) -> None:
        if self.run is None:
            return
        for k, v in summary_dict.items():
            self.run.summary[k] = v

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()