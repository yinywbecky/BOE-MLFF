import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local model_two_force.py drift-weight sweep without warmup or ramp."
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run training.")
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path("model_two_force.py"),
        help="Path to model_two_force.py.",
    )
    parser.add_argument(
        "--dataset-path1",
        type=Path,
        default=Path("low_temp_v2") / "dataset1" / "dataset_normal",
        help="Mode-1 dataset path.",
    )
    parser.add_argument(
        "--dataset-path2",
        type=Path,
        default=Path("low_temp_v2") / "dataset2" / "dataset_normal",
        help="Mode-2 dataset path.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("resultsweep_low_temp_v2"),
        help="Base output directory for all runs.",
    )
    parser.add_argument(
        "--drift-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.05],
        help="Drift weights to sweep.",
    )
    parser.add_argument("--n-col", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--total-force-xy-limit", type=float, default=0.2)
    parser.add_argument("--total-force-z-limit", type=float, default=0.4)
    parser.add_argument("--offset-energy", type=float, default=429.825)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=max((__import__("os").cpu_count() or 1) - 1, 1))
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--val-frequency", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep even if one run fails.",
    )
    return parser.parse_args()


def run_one(args: argparse.Namespace, drift_weight: float) -> int:
    run_dir = args.result_root / f"dw_{drift_weight:.3f}".replace(".", "p")
    cmd = [
        args.python,
        str(args.trainer),
        "--dataset-path1",
        str(args.dataset_path1),
        "--dataset-path2",
        str(args.dataset_path2),
        "--result-root",
        str(run_dir),
        "--n-col",
        str(args.n_col),
        "--hidden-dim",
        str(args.hidden_dim),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--force-weight",
        str(args.force_weight),
        "--energy-weight",
        str(args.energy_weight),
        "--weight-decay",
        str(args.weight_decay),
        "--drift-weight",
        str(drift_weight),
        "--drift-warmup-epochs",
        "0",
        "--drift-ramp-epochs",
        "0",
        "--total-force-xy-limit",
        str(args.total_force_xy_limit),
        "--total-force-z-limit",
        str(args.total_force_z_limit),
        "--limit-ramp-epochs",
        "0",
        "--val-frequency",
        str(args.val_frequency),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--offset-energy",
        str(args.offset_energy),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--torch-threads",
        str(args.torch_threads),
        "--torch-interop-threads",
        str(args.torch_interop_threads),
    ]

    print("\n=== Running drift weight:", drift_weight, "===")
    print("Output:", run_dir)
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "sweep_stdout.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_handle.write(line)
        return_code = proc.wait()

    print(f"Finished drift_weight={drift_weight}, exit_code={return_code}, log={log_path}")
    return int(return_code)


def main() -> None:
    args = parse_args()

    if not args.trainer.is_file():
        raise FileNotFoundError(f"Trainer script not found: {args.trainer}")

    args.result_root.mkdir(parents=True, exist_ok=True)
    failures = []

    for drift_weight in args.drift_weights:
        return_code = run_one(args, drift_weight)
        if return_code != 0:
            failures.append((drift_weight, return_code))
            print(f"Run failed for drift_weight={drift_weight}, exit_code={return_code}")
            if not args.continue_on_error:
                break

    print("\n=== Sweep finished ===")
    if failures:
        print("Failed runs:")
        for drift_weight, return_code in failures:
            print(f"  drift_weight={drift_weight}, exit_code={return_code}")
        raise SystemExit(1)

    print("All runs completed successfully.")


if __name__ == "__main__":
    main()