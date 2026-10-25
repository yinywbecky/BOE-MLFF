from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_center import VEFeaturesEnergyDataset
from model_two import DualEnergyModelContainer as BaseDualEnergyModelContainer
from model_two import SubNet, plot_force_comparisons


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH1 = BASE_DIR / "low_temp_v2" / "dataset1" / "dataset_normal"
DEFAULT_DATASET_PATH2 = BASE_DIR / "low_temp_v2" / "dataset2" / "dataset_normal"
DEFAULT_RESULT_ROOT = BASE_DIR / "result_dual_restricted_drift6"
DEFAULT_INPUT_DIM = 42
DEFAULT_HIDDEN_DIM = 96
ENERGY_OFFSET_DEFAULT = 428.8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DualEnergyModelContainer(BaseDualEnergyModelContainer):
    """Dual-model trainer with an extra penalty on structure-level total-force drift."""

    def __init__(self, model1, model2, device: str | None = None, non_blocking: bool = False):
        super().__init__(model1, model2, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.non_blocking = non_blocking

    @staticmethod
    def _structure_total_force_components(out_forces: torch.Tensor) -> torch.Tensor:
        total_fx = torch.sum(out_forces[:, :, 0] + out_forces[:, :, 3], dim=1)
        total_fy = torch.sum(out_forces[:, :, 1] + out_forces[:, :, 4], dim=1)
        total_fz = torch.sum(out_forces[:, :, 2] + out_forces[:, :, 5], dim=1)
        return torch.stack((total_fx, total_fy, total_fz), dim=1)

    @classmethod
    def _total_force_drift_loss(
        cls,
        out_forces: torch.Tensor,
        xy_limit: float = 0.1,
        z_limit: float = 0.2,
        normalize_by_ve_count: bool = False,
    ) -> torch.Tensor:
        total_force = cls._structure_total_force_components(out_forces)
        if normalize_by_ve_count:
            total_force = total_force / max(out_forces.shape[1], 1)
        limits = torch.tensor((xy_limit, xy_limit, z_limit), dtype=out_forces.dtype, device=out_forces.device)
        excess = torch.relu(torch.abs(total_force) - limits)
        return torch.mean(excess.square())

    @classmethod
    def _total_force_drift_stats(
        cls,
        out_forces: torch.Tensor,
        xy_limit: float = 0.1,
        z_limit: float = 0.2,
        normalize_by_ve_count: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_force = cls._structure_total_force_components(out_forces)
        if normalize_by_ve_count:
            total_force = total_force / max(out_forces.shape[1], 1)
        limits = torch.tensor((xy_limit, xy_limit, z_limit), dtype=out_forces.dtype, device=out_forces.device)
        excess = torch.relu(torch.abs(total_force) - limits)
        exceed_rate = torch.mean(torch.any(excess > 0.0, dim=1).to(out_forces.dtype))
        excess_norm = torch.linalg.norm(excess, dim=1)
        p95 = torch.quantile(excess_norm, 0.95)
        return exceed_rate, p95

    @staticmethod
    def _linear_schedule(start_value: float, end_value: float, epoch: int, ramp_epochs: int) -> float:
        if ramp_epochs <= 0:
            return end_value
        progress = min(max((epoch - 1) / float(ramp_epochs), 0.0), 1.0)
        return start_value + (end_value - start_value) * progress

    @classmethod
    def _epoch_drift_weight(
        cls,
        epoch: int,
        final_drift_weight: float,
        drift_warmup_epochs: int,
        drift_ramp_epochs: int,
    ) -> float:
        if epoch <= drift_warmup_epochs:
            return 0.0
        ramp_epoch = epoch - drift_warmup_epochs
        return cls._linear_schedule(0.0, final_drift_weight, ramp_epoch, drift_ramp_epochs)

    @classmethod
    def _epoch_force_limits(
        cls,
        epoch: int,
        target_xy_limit: float,
        target_z_limit: float,
        start_xy_limit: float | None,
        start_z_limit: float | None,
        limit_ramp_epochs: int,
    ) -> tuple[float, float]:
        if start_xy_limit is None or start_z_limit is None or limit_ramp_epochs <= 0:
            return target_xy_limit, target_z_limit
        xy_limit = cls._linear_schedule(start_xy_limit, target_xy_limit, epoch, limit_ramp_epochs)
        z_limit = cls._linear_schedule(start_z_limit, target_z_limit, epoch, limit_ramp_epochs)
        return xy_limit, z_limit

    @staticmethod
    def _val_early_stop_score(
        val_summary: dict[str, float],
        metric: str,
        force_mev_weight: float,
        energy_mev_weight: float,
    ) -> float:
        force_rmse_mev = float(val_summary["pooled_force_rmse"]) * 1000.0
        energy_rmse_mev_atom = float(val_summary["energy_rmse_mev_atom"])
        if metric == "force_rmse":
            return force_rmse_mev
        if metric == "composite":
            return force_mev_weight * force_rmse_mev + energy_mev_weight * energy_rmse_mev_atom
        raise ValueError(f"Unsupported early_stop_metric: {metric}")

    def fit(
        self,
        loader1: DataLoader,
        loader2: DataLoader,
        ep: int,
        learning_rate: float = 1e-3,
        force_weight: float = 1.0,
        energy_weight: float = 0.1,
        weight_decay: float = 0.0,
        drift_weight: float = 1.0,
        total_force_xy_limit: float = 0.1,
        total_force_z_limit: float = 0.2,
        val_loaders: tuple[DataLoader, DataLoader] | None = None,
        checkpoint_paths: tuple[Path, Path] | None = None,
        details_path: Path | None = None,
        val_frequency: int = 1,
        early_stop_patience: int = 12,
        early_stop_metric: str = "force_rmse",
        early_stop_force_mev_weight: float = 1.0,
        early_stop_energy_mev_weight: float = 0.0,
        drift_weight_mode: str = "fixed",
        adaptive_drift_target_exceed_rate: float = 0.2,
        adaptive_drift_update_lr: float = 0.5,
        adaptive_drift_weight_min: float = 0.0,
        adaptive_drift_weight_max: float = 1.0,
        checkpoint_select_start_epoch: int = 1,
        early_stop_start_epoch: int = 1,
        early_stop_min_delta: float = 0.0,
        drift_warmup_epochs: int = 0,
        drift_ramp_epochs: int = 0,
        normalize_drift_by_ve_count: bool = False,
        total_force_xy_limit_start: float | None = None,
        total_force_z_limit_start: float | None = None,
        limit_ramp_epochs: int = 0,
    ) -> list[dict[str, float]]:
        self.optimizer = torch.optim.Adam(
            list(self.model1.parameters()) + list(self.model2.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        self.model1.train()
        self.model2.train()

        best_val_score = float("inf")
        best_val_force_rmse = float("inf")
        best_val_energy_rmse = float("inf")
        best_epoch = None
        epochs_since_improve = 0
        history: list[dict[str, float]] = []

        if details_path is not None:
            details_path.write_text("Dual-model training details with total-force drift penalty\n", encoding="utf-8")

        adaptive_drift_weight = float(drift_weight)

        for epoch in range(1, ep + 1):
            epoch_start = time.perf_counter()
            if drift_weight_mode == "adaptive":
                epoch_drift_weight = float(adaptive_drift_weight)
            else:
                epoch_drift_weight = self._epoch_drift_weight(
                    epoch=epoch,
                    final_drift_weight=drift_weight,
                    drift_warmup_epochs=drift_warmup_epochs,
                    drift_ramp_epochs=drift_ramp_epochs,
                )
            epoch_xy_limit, epoch_z_limit = self._epoch_force_limits(
                epoch=epoch,
                target_xy_limit=total_force_xy_limit,
                target_z_limit=total_force_z_limit,
                start_xy_limit=total_force_xy_limit_start,
                start_z_limit=total_force_z_limit_start,
                limit_ramp_epochs=limit_ramp_epochs,
            )
            f_loss_sum = 0.0
            e_loss_sum = 0.0
            drift_loss_sum = 0.0
            drift_exceed_rate_sum = 0.0
            drift_excess_p95_sum = 0.0
            loss_sum = 0.0
            n_batches = 0

            for (X1, forces1, energy1), (X2, forces2, _energy2) in zip(loader1, loader2):
                X1 = X1.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                forces1 = forces1.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                energy1 = energy1.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                X2 = X2.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)
                forces2 = forces2.to(self.device, dtype=torch.float32, non_blocking=self.non_blocking)

                self.optimizer.zero_grad(set_to_none=True)

                batch_size, num_ve, feature_dim1 = X1.shape
                _, _, feature_dim2 = X2.shape
                out1 = self.model1(X1.reshape(batch_size * num_ve, feature_dim1)).reshape(batch_size, num_ve, -1)
                out2 = self.model2(X2.reshape(batch_size * num_ve, feature_dim2)).reshape(batch_size, num_ve, -1)
                out_forces1 = out1[:, :, 0:6]
                out_forces2 = out2[:, :, 0:6]
                tot_e1 = out1[:, :, 6].sum(dim=1)
                tot_e2 = out2[:, :, 6].sum(dim=1)

                avg_energy = (tot_e1 + tot_e2) / 2.0

                f_loss1 = torch.nn.functional.mse_loss(out_forces1, forces1)
                f_loss2 = torch.nn.functional.mse_loss(out_forces2, forces2)
                f_loss = f_loss1 + f_loss2
                e_loss = self.criterion(avg_energy.squeeze(), energy1)

                drift_loss1 = self._total_force_drift_loss(
                    out_forces1,
                    xy_limit=epoch_xy_limit,
                    z_limit=epoch_z_limit,
                    normalize_by_ve_count=normalize_drift_by_ve_count,
                )
                drift_loss2 = self._total_force_drift_loss(
                    out_forces2,
                    xy_limit=epoch_xy_limit,
                    z_limit=epoch_z_limit,
                    normalize_by_ve_count=normalize_drift_by_ve_count,
                )
                drift_loss = drift_loss1 + drift_loss2

                exceed_rate1, p95_1 = self._total_force_drift_stats(
                    out_forces1,
                    xy_limit=epoch_xy_limit,
                    z_limit=epoch_z_limit,
                    normalize_by_ve_count=normalize_drift_by_ve_count,
                )
                exceed_rate2, p95_2 = self._total_force_drift_stats(
                    out_forces2,
                    xy_limit=epoch_xy_limit,
                    z_limit=epoch_z_limit,
                    normalize_by_ve_count=normalize_drift_by_ve_count,
                )
                batch_exceed_rate = (exceed_rate1 + exceed_rate2) / 2.0
                batch_p95 = (p95_1 + p95_2) / 2.0

                loss = force_weight * f_loss + energy_weight * e_loss + epoch_drift_weight * drift_loss
                loss.backward()
                self.optimizer.step()

                f_loss_sum += f_loss.item()
                e_loss_sum += e_loss.item()
                drift_loss_sum += drift_loss.item()
                drift_exceed_rate_sum += batch_exceed_rate.item()
                drift_excess_p95_sum += batch_p95.item()
                loss_sum += loss.item()
                n_batches += 1

            self.scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "train_force_loss": f_loss_sum / max(n_batches, 1),
                "train_energy_loss": e_loss_sum / max(n_batches, 1),
                "train_drift_loss": drift_loss_sum / max(n_batches, 1),
                "train_drift_exceed_rate": drift_exceed_rate_sum / max(n_batches, 1),
                "train_drift_excess_p95": drift_excess_p95_sum / max(n_batches, 1),
                "train_total_loss": loss_sum / max(n_batches, 1),
                "learning_rate": current_lr,
                "epoch_seconds": epoch_seconds,
                "epoch_drift_weight": epoch_drift_weight,
                "epoch_total_force_xy_limit": epoch_xy_limit,
                "epoch_total_force_z_limit": epoch_z_limit,
            }

            if drift_weight_mode == "adaptive":
                adaptive_drift_weight = float(
                    np.clip(
                        adaptive_drift_weight
                        + adaptive_drift_update_lr
                        * (record["train_drift_exceed_rate"] - adaptive_drift_target_exceed_rate),
                        adaptive_drift_weight_min,
                        adaptive_drift_weight_max,
                    )
                )
                record["next_epoch_drift_weight"] = adaptive_drift_weight

            compact_line = (
                f"Epoch {epoch:04d} trainE={record['train_energy_loss']:.6f} "
                f"trainF={record['train_force_loss']:.6f} trainDrift={record['train_drift_loss']:.6f} "
                f"drEx={record['train_drift_exceed_rate']:.4f} drP95={record['train_drift_excess_p95']:.4f} "
                f"dw={epoch_drift_weight:.5f} lim=({epoch_xy_limit:.4f},{epoch_z_limit:.4f}) "
                f"lr={current_lr:.2e}"
            )
            detail_lines = [
                f"Epoch {epoch}",
                f"  train_energy_mse={record['train_energy_loss']:.10f}",
                f"  train_force_mse={record['train_force_loss']:.10f}",
                f"  train_total_force_drift_penalty={record['train_drift_loss']:.10f}",
                f"  train_drift_exceed_rate={record['train_drift_exceed_rate']:.10f}",
                f"  train_drift_excess_p95={record['train_drift_excess_p95']:.10f}",
                f"  train_total_loss={record['train_total_loss']:.10f}",
                f"  learning_rate={current_lr:.10e}",
                f"  epoch_seconds={epoch_seconds:.10f}",
                f"  weight_decay={float(weight_decay):.10e}",
                f"  drift_weight_final={float(drift_weight):.10f}",
                f"  drift_weight_epoch={float(epoch_drift_weight):.10f}",
                f"  drift_weight_mode={drift_weight_mode}",
                f"  adaptive_drift_target_exceed_rate={float(adaptive_drift_target_exceed_rate):.10f}",
                f"  adaptive_drift_update_lr={float(adaptive_drift_update_lr):.10f}",
                f"  adaptive_drift_weight_min={float(adaptive_drift_weight_min):.10f}",
                f"  adaptive_drift_weight_max={float(adaptive_drift_weight_max):.10f}",
                f"  total_force_xy_limit_target={float(total_force_xy_limit):.10f}",
                f"  total_force_z_limit_target={float(total_force_z_limit):.10f}",
                f"  total_force_xy_limit_epoch={float(epoch_xy_limit):.10f}",
                f"  total_force_z_limit_epoch={float(epoch_z_limit):.10f}",
                f"  normalize_drift_by_ve_count={int(normalize_drift_by_ve_count)}",
            ]
            if "next_epoch_drift_weight" in record:
                detail_lines.append(f"  next_epoch_drift_weight={float(record['next_epoch_drift_weight']):.10f}")

            if val_loaders is not None and epoch % val_frequency == 0:
                val_results = self.evaluate(val_loaders[0], val_loaders[1], verbose=False)
                val_summary = self._summarize_results(val_results)
                record.update(
                    {
                        "val_pooled_force_rmse": val_summary["pooled_force_rmse"],
                        "val_energy_rmse_mev_atom": val_summary["energy_rmse_mev_atom"],
                        "val_model1_pair1_rmse_mev": val_summary["model1_pair1_rmse_mev"],
                        "val_model1_pair2_rmse_mev": val_summary["model1_pair2_rmse_mev"],
                        "val_model2_pair1_rmse_mev": val_summary["model2_pair1_rmse_mev"],
                        "val_model2_pair2_rmse_mev": val_summary["model2_pair2_rmse_mev"],
                    }
                )
                compact_line += (
                    f" valF={val_summary['pooled_force_rmse'] * 1000:.3f} "
                    f"valE={val_summary['energy_rmse_mev_atom']:.3f}"
                )
                val_score = self._val_early_stop_score(
                    val_summary,
                    metric=early_stop_metric,
                    force_mev_weight=early_stop_force_mev_weight,
                    energy_mev_weight=early_stop_energy_mev_weight,
                )
                record["val_early_stop_score"] = val_score
                compact_line += f" valScore={val_score:.3f}"
                detail_lines.append(f"  val_early_stop_metric={early_stop_metric}")
                detail_lines.append(f"  val_early_stop_score={val_score:.10f}")
                checkpoint_selection_enabled = epoch >= max(checkpoint_select_start_epoch, 1)
                if not checkpoint_selection_enabled:
                    detail_lines.append(
                        f"  checkpoint_selection_skipped=1 (start_epoch={int(checkpoint_select_start_epoch)})"
                    )

                if checkpoint_selection_enabled and val_score < (best_val_score - max(early_stop_min_delta, 0.0)):
                    best_val_score = val_score
                    best_val_force_rmse = val_summary["pooled_force_rmse"]
                    best_val_energy_rmse = val_summary["energy_rmse_mev_atom"]
                    best_epoch = epoch
                    epochs_since_improve = 0
                    record["best_checkpoint"] = 1.0
                    compact_line += " best"
                    detail_lines.append("  checkpoint=updated")
                    if checkpoint_paths is not None:
                        self.save(checkpoint_paths[0], checkpoint_paths[1])
                else:
                    epochs_since_improve += 1
                    record["best_checkpoint"] = 0.0

            history.append(record)
            print(compact_line)
            self._append_details(details_path, "\n".join(detail_lines) + "\n")

            if (
                val_loaders is not None
                and early_stop_patience > 0
                and epoch >= max(early_stop_start_epoch, 1)
                and epochs_since_improve >= early_stop_patience
            ):
                stop_msg = (
                    f"Early stopping triggered at epoch {epoch}: "
                    f"no validation improvement for {early_stop_patience} validation checks "
                    f"on metric={early_stop_metric}."
                )
                print(stop_msg)
                self._append_details(details_path, stop_msg)
                break

        if checkpoint_paths is not None:
            if best_epoch is not None and checkpoint_paths[0].exists() and checkpoint_paths[1].exists():
                self.load(checkpoint_paths[0], checkpoint_paths[1])
                print(
                    f"Best validation checkpoint: epoch {best_epoch} "
                    f"score={best_val_score:.3f} ({early_stop_metric}), "
                    f"pooled_force_rmse={best_val_force_rmse * 1000:.3f} meV/A, "
                    f"energy_rmse={best_val_energy_rmse:.3f} meV/atom"
                )
                self._append_details(
                    details_path,
                    f"Best validation checkpoint: epoch {best_epoch}, "
                    f"early_stop_score={best_val_score:.10f}, "
                    f"pooled_force_rmse_mevA={best_val_force_rmse * 1000:.10f}, "
                    f"energy_rmse_mev_atom={best_val_energy_rmse:.10f}",
                )
            else:
                self.save(checkpoint_paths[0], checkpoint_paths[1])

        return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the dual-force model with an extra total-force-drift penalty."
    )
    parser.add_argument("--dataset-path1", type=Path, default=DEFAULT_DATASET_PATH1, help="Path to mode-1 dataset_normal directory.")
    parser.add_argument("--dataset-path2", type=Path, default=DEFAULT_DATASET_PATH2, help="Path to mode-2 dataset_normal directory.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Directory to store checkpoints, history, and evaluation artifacts.")
    parser.add_argument("--n-col", type=int, default=DEFAULT_INPUT_DIM, help="Number of VE feature columns to use from each CSV.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension c used for both SubNet models.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for both modes.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (L2 regularization).")
    parser.add_argument("--force-weight", type=float, default=1.0, help="Weight for the force MSE term.")
    parser.add_argument("--energy-weight", type=float, default=0.1, help="Weight for the energy MSE term.")
    parser.add_argument("--drift-weight", type=float, default=0.1, help="Weight for the total-force drift penalty.")
    parser.add_argument(
        "--drift-warmup-epochs",
        type=int,
        default=0,
        help="Number of initial epochs with zero drift weight before ramping.",
    )
    parser.add_argument(
        "--drift-ramp-epochs",
        type=int,
        default=0,
        help="Number of epochs to linearly ramp drift weight from 0 to --drift-weight after warmup.",
    )
    parser.add_argument("--total-force-xy-limit", type=float, default=0.2, help="Allowed x/y total-force drift per structure before penalty applies.")
    parser.add_argument("--total-force-z-limit", type=float, default=0.4, help="Allowed z total-force drift per structure before penalty applies.")
    parser.add_argument(
        "--total-force-xy-limit-start",
        type=float,
        default=None,
        help="Optional starting x/y drift limit for coarse-to-tight scheduling; if omitted, fixed target limit is used.",
    )
    parser.add_argument(
        "--total-force-z-limit-start",
        type=float,
        default=None,
        help="Optional starting z drift limit for coarse-to-tight scheduling; if omitted, fixed target limit is used.",
    )
    parser.add_argument(
        "--limit-ramp-epochs",
        type=int,
        default=0,
        help="Epochs to linearly ramp force limits from start values to target values.",
    )
    parser.add_argument(
        "--normalize-drift-by-ve-count",
        action="store_true",
        help="Normalize structure total-force components by VE count before drift penalty.",
    )
    parser.add_argument("--offset-energy", type=float, default=ENERGY_OFFSET_DEFAULT, help="Offset added to energy labels before training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count. Default 0 is usually best on Windows for already-loaded CSV data.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=max((os.cpu_count() or 1) - 1, 1),
        help="Number of PyTorch intra-op CPU threads to use.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="Number of PyTorch inter-op CPU threads to use.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned host memory for faster CPU-to-GPU transfer. Useful only on CUDA.",
    )
    parser.add_argument("--val-frequency", type=int, default=1, help="How often to run validation in epochs.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=12,
        help="Early stop if validation pooled force RMSE does not improve for this many validation checks; use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="force_rmse",
        choices=("force_rmse", "composite"),
        help="Validation metric used for early stopping: force_rmse or weighted composite.",
    )
    parser.add_argument(
        "--early-stop-force-mev-weight",
        type=float,
        default=1.0,
        help="Force RMSE weight (meV/A) in early-stop composite score.",
    )
    parser.add_argument(
        "--early-stop-energy-mev-weight",
        type=float,
        default=0.0,
        help="Energy RMSE weight (meV/atom) in early-stop composite score.",
    )
    parser.add_argument(
        "--drift-weight-mode",
        type=str,
        default="fixed",
        choices=("fixed", "adaptive"),
        help="How drift weight is controlled: fixed schedule or adaptive constrained update.",
    )
    parser.add_argument(
        "--adaptive-drift-target-exceed-rate",
        type=float,
        default=0.2,
        help="Target structure exceed-rate used by adaptive drift control.",
    )
    parser.add_argument(
        "--adaptive-drift-update-lr",
        type=float,
        default=0.5,
        help="Update gain for adaptive drift weight control.",
    )
    parser.add_argument(
        "--adaptive-drift-weight-min",
        type=float,
        default=0.0,
        help="Lower clamp for adaptive drift weight.",
    )
    parser.add_argument(
        "--adaptive-drift-weight-max",
        type=float,
        default=1.0,
        help="Upper clamp for adaptive drift weight.",
    )
    parser.add_argument(
        "--checkpoint-select-start-epoch",
        type=int,
        default=1,
        help="Do not update best checkpoint before this epoch.",
    )
    parser.add_argument(
        "--early-stop-start-epoch",
        type=int,
        default=1,
        help="Do not trigger early stop before this epoch.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Required minimum decrease in early-stop score to count as an improvement.",
    )
    parser.add_argument(
        "--init-model1",
        type=Path,
        default=None,
        help="Optional checkpoint path to initialize model1 before training.",
    )
    parser.add_argument(
        "--init-model2",
        type=Path,
        default=None,
        help="Optional checkpoint path to initialize model2 before training.",
    )
    return parser.parse_args()


def read_split_frames(dataset_path: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ve_df = pd.read_csv(dataset_path / f"{split}_ve.csv")
    label_df = pd.read_csv(dataset_path / f"{split}_label.csv")
    return ve_df, label_df


def build_dataset(
    ve_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_col: int,
    offset_energy: float,
    pre_mean: np.ndarray | None = None,
    pre_std: np.ndarray | None = None,
) -> VEFeaturesEnergyDataset:
    dataset = VEFeaturesEnergyDataset(ve_df, label_df, n_col)
    dataset.normalize_y(offset_energy=offset_energy)
    dataset.normalize_X(mode="standard", pre_mean=pre_mean, pre_std=pre_std)
    return dataset


def build_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int, pin_memory: bool) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs["generator"] = generator
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def prepare_mode_loaders(
    dataset_path: Path,
    batch_size: int,
    n_col: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    offset_energy: float,
) -> dict[str, DataLoader]:
    train_ve, train_label = read_split_frames(dataset_path, "train")
    val_ve, val_label = read_split_frames(dataset_path, "val")
    test_ve, test_label = read_split_frames(dataset_path, "test")

    train_dataset = build_dataset(train_ve, train_label, n_col, offset_energy)
    val_dataset = build_dataset(val_ve, val_label, n_col, offset_energy, train_dataset.pca_mean, train_dataset.pca_std)
    test_dataset = build_dataset(test_ve, test_label, n_col, offset_energy, train_dataset.pca_mean, train_dataset.pca_std)

    return {
        "train_loader": build_loader(train_dataset, batch_size, True, seed, num_workers, pin_memory),
        "val_loader": build_loader(val_dataset, batch_size, False, seed, num_workers, pin_memory),
        "test_loader": build_loader(test_dataset, batch_size, False, seed, num_workers, pin_memory),
    }


def save_summary(summary: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(args.torch_threads, 1))
    torch.set_num_interop_threads(max(args.torch_interop_threads, 1))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args.result_root.mkdir(parents=True, exist_ok=True)

    mode1_data = prepare_mode_loaders(
        args.dataset_path1,
        args.batch_size,
        args.n_col,
        args.seed,
        args.num_workers,
        args.pin_memory,
        args.offset_energy,
    )
    mode2_data = prepare_mode_loaders(
        args.dataset_path2,
        args.batch_size,
        args.n_col,
        args.seed,
        args.num_workers,
        args.pin_memory,
        args.offset_energy,
    )

    model = DualEnergyModelContainer(
        SubNet(input_size=args.n_col, output_size=7, c=args.hidden_dim),
        SubNet(input_size=args.n_col, output_size=7, c=args.hidden_dim),
        non_blocking=args.pin_memory and torch.cuda.is_available(),
    )

    if args.init_model1 is not None or args.init_model2 is not None:
        if args.init_model1 is None or args.init_model2 is None:
            raise ValueError("Both --init-model1 and --init-model2 must be provided together.")
        if not args.init_model1.is_file() or not args.init_model2.is_file():
            raise FileNotFoundError(
                f"Warm-start checkpoints not found: model1={args.init_model1}, model2={args.init_model2}"
            )
        print(f"Warm-starting from checkpoints: {args.init_model1} | {args.init_model2}")
        model.load(args.init_model1, args.init_model2)

    checkpoint_paths = (args.result_root / "model1.pth", args.result_root / "model2.pth")
    details_path = args.result_root / "training_details.txt"
    history = model.fit(
        mode1_data["train_loader"],
        mode2_data["train_loader"],
        ep=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        force_weight=args.force_weight,
        energy_weight=args.energy_weight,
        drift_weight=args.drift_weight,
        total_force_xy_limit=args.total_force_xy_limit,
        total_force_z_limit=args.total_force_z_limit,
        val_loaders=(mode1_data["val_loader"], mode2_data["val_loader"]),
        checkpoint_paths=checkpoint_paths,
        details_path=details_path,
        val_frequency=args.val_frequency,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_force_mev_weight=args.early_stop_force_mev_weight,
        early_stop_energy_mev_weight=args.early_stop_energy_mev_weight,
        drift_weight_mode=args.drift_weight_mode,
        adaptive_drift_target_exceed_rate=args.adaptive_drift_target_exceed_rate,
        adaptive_drift_update_lr=args.adaptive_drift_update_lr,
        adaptive_drift_weight_min=args.adaptive_drift_weight_min,
        adaptive_drift_weight_max=args.adaptive_drift_weight_max,
        checkpoint_select_start_epoch=args.checkpoint_select_start_epoch,
        early_stop_start_epoch=args.early_stop_start_epoch,
        early_stop_min_delta=args.early_stop_min_delta,
        drift_warmup_epochs=args.drift_warmup_epochs,
        drift_ramp_epochs=args.drift_ramp_epochs,
        normalize_drift_by_ve_count=args.normalize_drift_by_ve_count,
        total_force_xy_limit_start=args.total_force_xy_limit_start,
        total_force_z_limit_start=args.total_force_z_limit_start,
        limit_ramp_epochs=args.limit_ramp_epochs,
    )

    pd.DataFrame(history).to_csv(args.result_root / "history.csv", index=False)

    train_results = model.evaluate(mode1_data["train_loader"], mode2_data["train_loader"], verbose=False)
    val_results = model.evaluate(mode1_data["val_loader"], mode2_data["val_loader"], verbose=False)
    test_results = model.evaluate(mode1_data["test_loader"], mode2_data["test_loader"], verbose=False)

    summary = {
        "dataset_path1": str(args.dataset_path1),
        "dataset_path2": str(args.dataset_path2),
        "result_root": str(args.result_root),
        "n_col": args.n_col,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "force_weight": args.force_weight,
        "energy_weight": args.energy_weight,
        "drift_weight": args.drift_weight,
        "drift_warmup_epochs": args.drift_warmup_epochs,
        "drift_ramp_epochs": args.drift_ramp_epochs,
        "normalize_drift_by_ve_count": args.normalize_drift_by_ve_count,
        "total_force_xy_limit": args.total_force_xy_limit,
        "total_force_z_limit": args.total_force_z_limit,
        "total_force_xy_limit_start": args.total_force_xy_limit_start,
        "total_force_z_limit_start": args.total_force_z_limit_start,
        "limit_ramp_epochs": args.limit_ramp_epochs,
        "offset_energy": args.offset_energy,
        "seed": args.seed,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_force_mev_weight": args.early_stop_force_mev_weight,
        "early_stop_energy_mev_weight": args.early_stop_energy_mev_weight,
        "drift_weight_mode": args.drift_weight_mode,
        "adaptive_drift_target_exceed_rate": args.adaptive_drift_target_exceed_rate,
        "adaptive_drift_update_lr": args.adaptive_drift_update_lr,
        "adaptive_drift_weight_min": args.adaptive_drift_weight_min,
        "adaptive_drift_weight_max": args.adaptive_drift_weight_max,
        "checkpoint_select_start_epoch": args.checkpoint_select_start_epoch,
        "early_stop_start_epoch": args.early_stop_start_epoch,
        "early_stop_min_delta": args.early_stop_min_delta,
        "init_model1": str(args.init_model1) if args.init_model1 is not None else None,
        "init_model2": str(args.init_model2) if args.init_model2 is not None else None,
        "train_summary": model._summarize_results(train_results),
        "val_summary": model._summarize_results(val_results),
        "test_summary": model._summarize_results(test_results),
    }
    save_summary(summary, args.result_root / "summary.json")

    m1_p1, m1_p2 = test_results["model1"]
    m2_p1, m2_p2 = test_results["model2"]
    energy = test_results["energy"]
    plot_force_comparisons(m1_p1, m1_p2, energy, save_path=str(args.result_root), fname="/test_rmse_m1.png")
    plot_force_comparisons(m2_p1, m2_p2, energy, save_path=str(args.result_root), fname="/test_rmse_m2.png")

    m1_p1_tr, m1_p2_tr = train_results["model1"]
    m2_p1_tr, m2_p2_tr = train_results["model2"]
    energy_tr = train_results["energy"]
    plot_force_comparisons(m1_p1_tr, m1_p2_tr, energy_tr, save_path=str(args.result_root), fname="/train_rmse_m1.png")
    plot_force_comparisons(m2_p1_tr, m2_p2_tr, energy_tr, save_path=str(args.result_root), fname="/train_rmse_m2.png")

    print(f"Saved training history to {args.result_root / 'history.csv'}")
    print(f"Saved summary to {args.result_root / 'summary.json'}")


if __name__ == "__main__":
    main()
