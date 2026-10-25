import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice

from boe_cashe import BPProcessor, get_centers


BASE_DIR = os.path.dirname(__file__)
TEMPERATURES = ["100","200","300", "400", "500", "600"]

# SELECTION_CSV = None  # Set to None to use all frames without selection
REF_STRUCTURE_PATH = os.path.join(BASE_DIR, "POSCAR_REV.vasp")
OUTPUT_ROOT = os.path.join(BASE_DIR, "low_temp_v2")
SELECTION_CSV = os.path.join(OUTPUT_ROOT, "selected_frames_proportional_scheme.csv")
OUTPUT_METADATA = os.path.join(OUTPUT_ROOT, "selected_structure_metadata.csv")
ENERGY_FILE_CANDIDATES = ["all_energy.txt", "energy.txt"]
ADD_CENTER = True


def resolve_energy_file(temp: str) -> str:
    """Return the first available energy file for a temperature directory."""
    for candidate in ENERGY_FILE_CANDIDATES:
        path = os.path.join(BASE_DIR, temp, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No energy file found in {temp}. Expected one of: {ENERGY_FILE_CANDIDATES}"
    )


def load_energy_series(temp: str) -> np.ndarray:
    """Load total energies for one temperature directory."""
    path = resolve_energy_file(temp)
    data = np.loadtxt(path).reshape(-1)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite energies.")
    return data


def _extract_temperature_key(temp: str) -> int:
    """Extract a numeric temperature key from strings that may contain control chars."""
    cleaned = "".join(ch for ch in str(temp) if ch.isdigit() or ch == "-")
    if cleaned in {"", "-"}:
        raise ValueError(f"Invalid temperature token: {repr(temp)}")
    return int(cleaned)


def load_frame_table(selection_csv: str | None = SELECTION_CSV) -> dict[str, pd.DataFrame]:
    """Load absolute raw-frame indices and reject misaligned energy labels."""
    use_all_frames = selection_csv is None
    if use_all_frames:
        print("Selection explicitly disabled; using all raw frames.")

    grouped_indices = {}
    if use_all_frames:
        for temp in TEMPERATURES:
            temp_key = _extract_temperature_key(temp)
            energies = load_energy_series(str(temp_key))
            grouped_indices[temp] = pd.DataFrame(
                {
                    "frame_index": np.arange(len(energies), dtype=int),
                    "energy_eV": energies.astype(float),
                }
            )
        return grouped_indices

    selection_df = pd.read_csv(selection_csv)
    required_columns = {"temperature_K", "frame_index", "energy_eV"}
    missing_columns = required_columns.difference(selection_df.columns)
    if missing_columns:
        raise ValueError(
            f"Selection file is missing required columns: {sorted(missing_columns)}"
        )

    if selection_df.empty:
        raise ValueError(f"{selection_csv}: empty selection.")
    for column in required_columns:
        selection_df[column] = pd.to_numeric(selection_df[column], errors="raise")
        values = selection_df[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{selection_csv}: non-finite {column}.")
        if column != "energy_eV" and (values != np.floor(values)).any():
            raise ValueError(f"{selection_csv}: non-integer {column}.")
    if set(selection_df["temperature_K"]) != {int(temp) for temp in TEMPERATURES}:
        raise ValueError(f"{selection_csv}: expected temperatures {TEMPERATURES}.")
    if selection_df.duplicated(["temperature_K", "frame_index"]).any():
        raise ValueError(f"{selection_csv}: duplicate selected frames.")
    selection_df = selection_df[["temperature_K", "frame_index", "energy_eV"]].copy()
    selection_df["temperature_K"] = selection_df["temperature_K"].astype(int)
    selection_df["frame_index"] = selection_df["frame_index"].astype(int)
    selection_df["energy_eV"] = selection_df["energy_eV"].astype(float)

    for temp in TEMPERATURES:
        temp_key = _extract_temperature_key(temp)
        temp_rows = selection_df.loc[
            selection_df["temperature_K"] == temp_key,
            ["frame_index", "energy_eV"],
        ].sort_values("frame_index")
        energies = load_energy_series(str(temp_key))
        indices = temp_rows["frame_index"].to_numpy(dtype=int)
        if (indices < 0).any() or (indices >= len(energies)).any():
            raise ValueError(f"{selection_csv}: raw frame indices out of range for {temp} K.")
        if not np.allclose(
            temp_rows["energy_eV"].to_numpy(), energies[indices], rtol=0, atol=1e-8
        ):
            raise ValueError(
                f"{selection_csv}: energy/frame mismatch for {temp} K. "
                "frame_index must index the full raw trajectory, not the sampling window. "
                "Regenerate selections and datasets with the release scripts."
            )
        grouped_indices[temp] = temp_rows.reset_index(drop=True)

    return grouped_indices


def read_selected_fractional_coords(
    file_name: str,
    config_list: list[int] | None,
    expected_frame_count: int | None = None,
) -> tuple[list[np.ndarray], Lattice]:
    """Read selected XDATCAR frames and return fractional coordinates."""
    with open(file_name, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 8:
        raise ValueError(f"Invalid XDATCAR file: {file_name}")

    scale = float(lines[1].split()[0])
    lattice_matrix = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    ) * scale
    lattice = Lattice(lattice_matrix)

    possible_symbols = lines[5].split()
    possible_counts = lines[6].split()
    has_symbols = all(token.isalpha() for token in possible_symbols)

    if has_symbols:
        atom_counts = np.array([int(x) for x in possible_counts], dtype=int)
        start_line = 7
    else:
        atom_counts = np.array([int(x) for x in possible_symbols], dtype=int)
        start_line = 6

    atom_num = int(np.sum(atom_counts))
    if config_list is None:
        targets = None
        selected_frames = []
    else:
        targets = {int(index): order for order, index in enumerate(config_list)}
        selected_frames = [None] * len(config_list)
    frame_index = -1
    i = start_line

    while i < len(lines):
        header = lines[i].lower()
        if "configuration" in header or header.startswith("direct") or header.startswith("cart"):
            use_cartesian = header.startswith("cart")
            frame_index += 1
            i += 1
            if i + atom_num > len(lines):
                raise ValueError(f"{file_name}: incomplete structure frame {frame_index}.")

            if targets is None or frame_index in targets:
                frame = np.array(
                    [[float(x) for x in lines[j].split()[:3]] for j in range(i, i + atom_num)],
                    dtype=float,
                )
                if use_cartesian:
                    frame = lattice.get_fractional_coords(frame)
                if targets is None:
                    selected_frames.append(frame)
                else:
                    selected_frames[targets[frame_index]] = frame
            i += atom_num
        else:
            i += 1

    if expected_frame_count is not None and frame_index + 1 != expected_frame_count:
        raise ValueError(
            f"{file_name}: {frame_index + 1} structure frames but "
            f"{expected_frame_count} energies."
        )
    if targets is not None:
        missing = [config_list[i] for i, frame in enumerate(selected_frames) if frame is None]
        if missing:
            raise ValueError(f"Could not locate selected frames in {file_name}: {missing[:10]}")

    return selected_frames, lattice


def process_selected_frames_to_csv(
    grouped_indices: dict[str, pd.DataFrame],
    mode: str,
    filename: str,
    add_center: bool = ADD_CENTER,
) -> pd.DataFrame:
    """Generate BOE features for all selected frames using BPProcessor.fast_BOE."""
    all_combined_features = []
    metadata_rows = []
    global_structure_index = 0
    atom_feature_dim = None
    center_feature_dim = None

    for temp in TEMPERATURES:
        temp_rows = grouped_indices.get(temp)
        if temp_rows is None or temp_rows.empty:
            print(f"No selected frames for {temp} K, skipping.")
            continue

        frame_indices = temp_rows["frame_index"].astype(int).tolist()
        xdatcar_path = os.path.join(BASE_DIR, temp, "XDATCAR")
        if not os.path.isfile(xdatcar_path):
            raise FileNotFoundError(xdatcar_path)

        print(f"Generating mode {mode} BOEs from {len(frame_indices)} selected frames at {temp} K...")
        selected_frames, lattice = read_selected_fractional_coords(
            xdatcar_path, frame_indices, expected_frame_count=len(load_energy_series(temp))
        )

        for local_order, frac_coords in enumerate(selected_frames):
            boe_atoms = BPProcessor.fast_BOE(frac_coords, lattice.matrix, mode)
            n_boe_per_structure = boe_atoms.shape[0]
            current_atom_feature_dim = boe_atoms.shape[1] * boe_atoms.shape[2]
            if add_center:
                boe_centers = get_centers(boe_atoms)
                current_center_feature_dim = boe_centers.shape[1] * boe_centers.shape[2]
            else:
                boe_centers = None
                current_center_feature_dim = 0

            if atom_feature_dim is None:
                atom_feature_dim = current_atom_feature_dim
                center_feature_dim = current_center_feature_dim
            elif (
                current_atom_feature_dim != atom_feature_dim
                or current_center_feature_dim != center_feature_dim
            ):
                raise ValueError(
                    "Inconsistent BOE feature dimensions across frames. "
                    f"Expected atom/center dims {atom_feature_dim}/{center_feature_dim}, "
                    f"got {current_atom_feature_dim}/{current_center_feature_dim}."
                )

            feature_blocks = [
                boe_atoms.reshape(n_boe_per_structure, current_atom_feature_dim)
            ]
            if add_center and boe_centers is not None:
                feature_blocks.append(
                    boe_centers.reshape(n_boe_per_structure, current_center_feature_dim)
                )
            combined_features = np.hstack(feature_blocks)
            all_combined_features.append(combined_features)
            metadata_rows.append(
                {
                    "temperature_K": int(temp),
                    "frame_index": int(frame_indices[local_order]),
                    "energy_eV": float(temp_rows.iloc[local_order]["energy_eV"]),
                    "global_structure_index": global_structure_index,
                }
            )
            global_structure_index += 1

    if not all_combined_features:
        raise RuntimeError(f"No BOE features were generated for mode {mode}.")

    final_features_np = np.vstack(all_combined_features)
    if atom_feature_dim is None or center_feature_dim is None:
        raise RuntimeError(f"Failed to infer BOE feature dimensions for mode {mode}.")

    n_atoms_per_boe = atom_feature_dim // 3
    n_centers_per_boe = center_feature_dim // 3
    atom_cols = [f"{coord}{i}" for i in range(n_atoms_per_boe) for coord in ["x", "y", "z"]]
    center_cols = [f"c{coord}{i}" for i in range(n_centers_per_boe) for coord in ["x", "y", "z"]] if add_center else []
    final_df = pd.DataFrame(final_features_np, columns=atom_cols + center_cols)
    final_df.to_csv(filename, index=False)
    print(f"Successfully saved mode {mode} combined BOE features to {filename}")

    metadata_df = pd.DataFrame(metadata_rows)
    return metadata_df


def initialize_fast_boe_cache(ref_struct: Structure):
    """Populate BPProcessor caches required by the fast_BOE path."""
    for mode in ("1", "2"):
        cache_key = BPProcessor._cache_key(mode)
        if cache_key not in BPProcessor._ref_cache:
            BPProcessor(primitive=ref_struct.copy(), mode=mode, ref_struct=ref_struct, use_cache=True)


def generate_all_temp_boes(add_center: bool = ADD_CENTER):
    """Generate mode-1 and mode-2 BOE feature tables for the selected or full all-temperature dataset."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    center1_dir = os.path.join(OUTPUT_ROOT, "center1")
    center2_dir = os.path.join(OUTPUT_ROOT, "center2")
    os.makedirs(center1_dir, exist_ok=True)
    os.makedirs(center2_dir, exist_ok=True)

    grouped_indices = load_frame_table(SELECTION_CSV)
    ref_struct = Structure.from_file(REF_STRUCTURE_PATH).copy()
    initialize_fast_boe_cache(ref_struct)

    filename1 = os.path.join(center1_dir, "all_BOE_features_mode.csv")
    filename2 = os.path.join(center2_dir, "all_BOE_features_mode.csv")

    metadata_df = process_selected_frames_to_csv(grouped_indices, mode="1", filename=filename1, add_center=add_center)
    process_selected_frames_to_csv(grouped_indices, mode="2", filename=filename2, add_center=add_center)

    metadata_df.to_csv(OUTPUT_METADATA, index=False)
    print(f"Structure metadata saved to {OUTPUT_METADATA}")

    print(f"Combined BOE features saved to {center1_dir} and {center2_dir}")


if __name__ == "__main__":
    generate_all_temp_boes(add_center=ADD_CENTER)