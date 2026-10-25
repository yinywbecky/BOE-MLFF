import os
import numpy as np
import pandas as pd

from gen_force import gen_pca_features
from gen_center_all_data import load_energy_series


BASE_DIR = os.path.dirname(__file__)
TEMPERATURES = ["100","200","300", "400", "500", "600"]
ALL_TEMP_ROOT = os.path.join(BASE_DIR, "low_temp_v2")
METADATA_PATH = os.path.join(ALL_TEMP_ROOT, "selected_structure_metadata.csv")
FORCE_FILE_CANDIDATES = ["force_only.dat", "forces_only.dat"]
MODE_CONFIGS = {
    "1": {
        "center_dir": os.path.join(ALL_TEMP_ROOT, "center1"),
        "dataset_dir": os.path.join(ALL_TEMP_ROOT, "dataset1"),
        "ref_dir": os.path.join(BASE_DIR, "data1"),
    },
    "2": {
        "center_dir": os.path.join(ALL_TEMP_ROOT, "center2"),
        "dataset_dir": os.path.join(ALL_TEMP_ROOT, "dataset2"),
        "ref_dir": os.path.join(BASE_DIR, "data2"),
    },
}


def resolve_force_file(temp: str) -> str:
    """Return the first available force file for a temperature directory."""
    for candidate in FORCE_FILE_CANDIDATES:
        path = os.path.join(BASE_DIR, temp, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No force file found in {temp}. Expected one of: {FORCE_FILE_CANDIDATES}"
    )


def load_structure_metadata() -> pd.DataFrame:
    """Load the structure table produced by gen_center_all_data.py."""
    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}. Run gen_center_all_data.py first."
        )

    metadata_df = pd.read_csv(METADATA_PATH)
    required = {"temperature_K", "frame_index", "energy_eV", "global_structure_index"}
    missing = required.difference(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing)}")

    return metadata_df.sort_values("global_structure_index").reset_index(drop=True)


def load_center_pairs(ref_dir: str) -> np.ndarray:
    """Load the two-atom pair indices used to align forces with BOEs."""
    standard_pair_list = np.load(os.path.join(ref_dir, "standar_pair_list.npy"))
    bond_center_indices = np.load(os.path.join(ref_dir, "bond_center_indices.npy"))
    return standard_pair_list[bond_center_indices]


def build_force_dataframe(metadata_df: pd.DataFrame, ref_dir: str, output_dir: str) -> pd.DataFrame:
    """Collect selected forces across all temperatures and align them to BOE pair order."""
    center_pairs = load_center_pairs(ref_dir)
    selected_force_blocks = []

    for temp in TEMPERATURES:
        temp_rows = metadata_df.loc[metadata_df["temperature_K"] == int(temp)].copy()
        if temp_rows.empty:
            continue

        force_file = resolve_force_file(temp)
        raw_forces = np.loadtxt(force_file, ndmin=2)
        if (
            raw_forces.shape[1] != 3
            or len(raw_forces) % 80
            or not np.isfinite(raw_forces).all()
        ):
            raise ValueError(f"{force_file}: expected complete 80-atom blocks of finite XYZ forces.")
        n_steps = len(raw_forces) // 80
        energies = load_energy_series(temp)
        if n_steps != len(energies):
            raise ValueError(
                f"{force_file}: {n_steps} force frames but {len(energies)} energies."
            )
        forces = raw_forces.reshape(n_steps, 80, 3)
        raw_indices = temp_rows["frame_index"].to_numpy()
        if (
            not np.isfinite(raw_indices).all()
            or (raw_indices != np.floor(raw_indices)).any()
            or (raw_indices < 0).any()
        ):
            raise ValueError(f"{METADATA_PATH}: invalid raw frame indices for {temp} K.")
        frame_indices = temp_rows["frame_index"].astype(int).to_numpy()

        if frame_indices.max(initial=-1) >= len(forces):
            raise ValueError(
                f"Selected frame index exceeds available force steps for {temp}: "
                f"max selected={frame_indices.max()}, available={len(forces)}"
            )
        if not np.allclose(
            temp_rows["energy_eV"].to_numpy(), energies[frame_indices], rtol=0, atol=1e-8
        ):
            raise ValueError(f"{METADATA_PATH}: energy/frame mismatch for {temp} K.")

        temp_selected = np.zeros((len(frame_indices) * len(center_pairs), 6))
        chosen_forces = forces[frame_indices]
        for i, frame_force in enumerate(chosen_forces):
            for j, pair in enumerate(center_pairs):
                temp_selected[i * len(center_pairs) + j] = frame_force[pair].reshape(-1)
        selected_force_blocks.append(temp_selected)

    if not selected_force_blocks:
        raise RuntimeError("No selected force blocks were generated.")

    force_columns = [f"f{axis}{i}" for i in range(2) for axis in ["x", "y", "z"]]
    df_forces = pd.DataFrame(np.vstack(selected_force_blocks), columns=force_columns)
    df_forces.to_csv(os.path.join(output_dir, "df_forces.csv"), index=False)
    print(f"Aligned force dataframe saved to {os.path.join(output_dir, 'df_forces.csv')}")
    return df_forces


def split_dataset_with_length(
    df_boe: pd.DataFrame,
    df_forces: pd.DataFrame,
    df_label: pd.DataFrame,
    dataset_root: str,
    data_dir: str,
    mode: str = "normal",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    n_ve: int = 20,
):
    """Split the all-temperature dataset into train/val/test using its real structure count."""
    outpath = os.path.join(dataset_root, f"dataset_{mode}")
    os.makedirs(outpath, exist_ok=True)

    if mode == "pca":
        df_features = gen_pca_features(df_boe, output_dir=data_dir)
    elif mode == "normal":
        df_features = df_boe
    elif mode == "reorganize":
        original_columns = df_boe.columns
        groups = [original_columns[i * 3 : (i + 1) * 3] for i in range(10)]
        new_order = [8, 9, 2, 3, 6, 7, 8, 9, 0, 1, 4, 5]
        reordered_groups = [groups[i] for i in new_order]
        new_columns = [col for group in reordered_groups for col in group]
        df_features = df_boe[new_columns]
    else:
        raise ValueError("Invalid mode. Choose from 'pca', 'normal', or 'reorganize'.")

    merged_data = pd.concat([df_features, df_forces], axis=1)
    print(f"{mode} BOE data shape: {df_features.shape}")
    print(f"Merged data shape: {merged_data.shape}")

    length = len(df_label)
    indices = np.arange(length)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(length * train_ratio)
    n_val = int(length * val_ratio)
    n_test = length - n_train - n_val

    split_indices = {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train:n_train + n_val]),
        "test": np.sort(indices[n_train + n_val:n_train + n_val + n_test]),
    }

    for split_name, structure_idx in split_indices.items():
        ve_idx = np.array(
            [np.arange(i * n_ve, (i + 1) * n_ve) for i in structure_idx],
            dtype=int,
        ).reshape(-1)
        merged_data.iloc[ve_idx].to_csv(
            os.path.join(outpath, f"{split_name}_ve.csv"), index=False
        )
        df_label.iloc[structure_idx].to_csv(
            os.path.join(outpath, f"{split_name}_label.csv"), index=False
        )

    print(
        f"Saved splits to {outpath}: train={len(split_indices['train'])}, "
        f"val={len(split_indices['val'])}, test={len(split_indices['test'])}"
    )
    return split_indices


def save_split_metadata(metadata_df: pd.DataFrame, split_indices: dict[str, np.ndarray], dataset_root: str):
    """Save train/val/test structure metadata for reproducibility."""
    dataset_dir = os.path.join(dataset_root, "dataset_normal")
    for split_name, structure_idx in split_indices.items():
        metadata_df.iloc[structure_idx].to_csv(
            os.path.join(dataset_dir, f"{split_name}_structure_metadata.csv"),
            index=False,
        )
    print(f"Saved split metadata to {dataset_dir}")


def generate_all_temp_force_datasets():
    """Build and split the combined all-temperature datasets for both BOE modes."""
    metadata_df = load_structure_metadata()
    label_df = metadata_df[["energy_eV"]].rename(columns={"energy_eV": "energy"})

    for mode, config in MODE_CONFIGS.items():
        center_dir = config["center_dir"]
        dataset_dir = config["dataset_dir"]
        ref_dir = config["ref_dir"]
        os.makedirs(dataset_dir, exist_ok=True)

        boe_file = os.path.join(center_dir, "all_BOE_features_mode.csv")
        if not os.path.isfile(boe_file):
            raise FileNotFoundError(
                f"BOE feature file not found: {boe_file}. Run gen_center_all_data.py first."
            )

        print(f"\n=== Processing all_temp mode {mode} ===")
        df_boe = pd.read_csv(boe_file)
        df_forces = build_force_dataframe(metadata_df, ref_dir=ref_dir, output_dir=center_dir)
        split_indices = split_dataset_with_length(
            df_boe=df_boe,
            df_forces=df_forces,
            df_label=label_df,
            dataset_root=dataset_dir,
            data_dir=center_dir,
            mode="normal",
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42,
        )
        save_split_metadata(metadata_df, split_indices, dataset_dir)


if __name__ == "__main__":
    generate_all_temp_force_datasets()