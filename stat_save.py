import argparse
import os
import numpy as np
import pandas as pd


def save_train_stats(dataset_path: str, n_col: int, output: str):
    ve_train = pd.read_csv(os.path.join(dataset_path, 'train_ve.csv'))
    features = ve_train.iloc[:, :n_col]
    values = features.to_numpy(dtype=np.float32)
    if values.shape[1] != n_col or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{dataset_path}: expected finite training features with {n_col} columns.")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    stat_df = pd.DataFrame({
        'mean': mean,
        'std': std,
    }, index=features.columns).T

    stat_df.to_csv(output, index=True)
    print(f"Saved training set statistics ({n_col} features) to {output}")
    print(stat_df)


def save_all_temp_stats(all_temp_root: str, n_col: int):
    """Save mode-1 and mode-2 training-set statistics under the all_temp directory."""
    mode_configs = {
        "1": {
            "dataset_path": os.path.join(all_temp_root, "dataset1", "dataset_normal"),
            "output": os.path.join(all_temp_root, "ds_train_stat1.csv"),
        },
        "2": {
            "dataset_path": os.path.join(all_temp_root, "dataset2", "dataset_normal"),
            "output": os.path.join(all_temp_root, "ds_train_stat2.csv"),
        },
    }

    for mode, config in mode_configs.items():
        dataset_path = config["dataset_path"]
        output = config["output"]
        if not os.path.isfile(os.path.join(dataset_path, "train_ve.csv")):
            raise FileNotFoundError(
                f"Training feature file not found for mode {mode}: "
                f"{os.path.join(dataset_path, 'train_ve.csv')}"
            )

        save_train_stats(dataset_path=dataset_path, n_col=n_col, output=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export training set mean/std to CSV')
    parser.add_argument('--dataset_path', type=str,
                        default=None,
                        help='Path to the folder containing train_ve.csv')
    parser.add_argument('--n_col', type=int, default=42,
                        help='Number of feature columns to include')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--all_temp_root', type=str, default='./low_temp_v2',
                        help='Root directory containing dataset1 and dataset2 for the combined all-temperature dataset')
    parser.add_argument('--single_mode', action='store_true',
                        help='Use --dataset_path and --output to save stats for a single dataset only')
    args = parser.parse_args()

    if args.single_mode:
        if args.dataset_path is None or args.output is None:
            raise ValueError('--single_mode requires both --dataset_path and --output.')
        save_train_stats(args.dataset_path, args.n_col, args.output)
    else:
        save_all_temp_stats(args.all_temp_root, args.n_col)
