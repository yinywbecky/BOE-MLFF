import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from boe_cashe import BPProcessor, get_centers
from model_two_force import DualEnergyModelContainer, SubNet


# Constants
kB = 8.617333262e-5  # Boltzmann constant (eV/K)
fs_to_ps = 1e-3      # Convert fs to ps
c = 2997924.58       # Speed of light in A/ps
p_mass = 30.973762 * 931494103 / (c * c)  # P mass in eV*ps^2/A^2


class CPUDualMLFF:
    """CPU-only dual-model MLFF wrapper matching the md_cpu.py force API."""

    def __init__(
        self,
        dual_model: DualEnergyModelContainer,
        stat_df1: pd.DataFrame,
        stat_df2: pd.DataFrame,
        stat_n_col: int | None = None,
    ):
        self.dual_model = dual_model
        model_feature_dim = int(self.dual_model.model1.FC1.in_features)
        self.feature_dim = model_feature_dim if stat_n_col is None else int(stat_n_col)
        if self.feature_dim not in (30, 42):
            raise ValueError(f"Unsupported dual-model input size: {self.feature_dim}. Expected 30 or 42.")
        if self.feature_dim != model_feature_dim:
            raise ValueError(
                f"Requested stat_n_col={self.feature_dim}, but model input dim is {model_feature_dim}. "
                "Please use matching dimensions."
            )

        self.df1 = stat_df1.iloc[:, :self.feature_dim].copy()
        self.df2 = stat_df2.iloc[:, :self.feature_dim].copy()
        self.ref_indices_cache = None

        self.df1_mean = torch.tensor(self.df1.loc[["mean"]].to_numpy(), dtype=torch.float32)
        self.df1_std = torch.tensor(self.df1.loc[["std"]].to_numpy(), dtype=torch.float32)
        self.df2_mean = torch.tensor(self.df2.loc[["mean"]].to_numpy(), dtype=torch.float32)
        self.df2_std = torch.tensor(self.df2.loc[["std"]].to_numpy(), dtype=torch.float32)
        for mean, std in ((self.df1_mean, self.df1_std), (self.df2_mean, self.df2_std)):
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                raise ValueError("Normalization statistics must be finite with nonnegative standard deviations.")
            std[std == 0] = 1.0

        self._ensure_model_float32()
        self._set_models_eval()

    def _ensure_model_float32(self):
        self.dual_model.model1.to(torch.float32)
        self.dual_model.model2.to(torch.float32)

    def _set_models_eval(self):
        self.dual_model.model1.eval()
        self.dual_model.model2.eval()

    def _predict_boe_batch(self, boes_1: torch.Tensor, boes_2: torch.Tensor):
        out_1 = self.dual_model.model1(boes_1)
        out_2 = self.dual_model.model2(boes_2)

        forcespair1_1 = out_1[:, 0:3].detach().cpu().numpy()
        forcespair2_1 = out_1[:, 3:6].detach().cpu().numpy()
        forcespair1_2 = out_2[:, 0:3].detach().cpu().numpy()
        forcespair2_2 = out_2[:, 3:6].detach().cpu().numpy()
        energy = ((out_1[:, 6].sum() + out_2[:, 6].sum()) / 2.0).detach().cpu().numpy()

        return np.float64(energy), forcespair1_1, forcespair2_1, forcespair1_2, forcespair2_2

    def _build_model_features(self, boes_np: np.ndarray) -> torch.Tensor:
        features = boes_np.reshape(len(boes_np), -1)
        # BOE generator returns 30 base features; 42-D models need +12 center features.
        if self.feature_dim == 42 and features.shape[1] == 30:
            centers = get_centers(boes_np).reshape(len(boes_np), -1)
            features = np.hstack((features, centers))
        if features.shape[1] < self.feature_dim:
            raise ValueError(
                f"BOE feature width {features.shape[1]} is smaller than required {self.feature_dim}."
            )
        if features.shape[1] > self.feature_dim:
            # Keep only the requested/model-compatible feature columns.
            features = features[:, :self.feature_dim]
        return torch.tensor(features, dtype=torch.float32)

    def getForceEnergy(self, structure: Structure = None, ref_struct: Structure = None, frac_coords: np.ndarray = None):
        """
        First call: pass structure + ref_struct to populate cache.
        Subsequent calls: pass frac_coords only for maximum speed.
        """
        with torch.no_grad():
            if self.ref_indices_cache is None:
                bp_1 = BPProcessor(
                    primitive=structure,
                    mode="1",
                    ref_struct=ref_struct if ref_struct is not None else None,
                )
                bp_2 = BPProcessor(
                    primitive=structure,
                    mode="2",
                    ref_struct=ref_struct if ref_struct is not None else None,
                )
                boes_1_np = bp_1.BOEs
                boes_2_np = bp_2.BOEs
                self.ref_indices_cache = bp_1.ref_indices
                self._lat_matrix = structure.lattice.matrix.copy()
                self._inv_lat_matrix = np.linalg.inv(structure.lattice.matrix)
            else:
                if frac_coords is None:
                    frac_coords = structure.frac_coords
                boes_1_np = BPProcessor.fast_BOE(frac_coords, self._lat_matrix, "1")
                boes_2_np = BPProcessor.fast_BOE(frac_coords, self._lat_matrix, "2")

            boes_1 = self._build_model_features(boes_1_np.astype(np.float32))
            boes_2 = self._build_model_features(boes_2_np.astype(np.float32))

            boes_1 = (boes_1 - self.df1_mean) / self.df1_std
            boes_2 = (boes_2 - self.df2_mean) / self.df2_std

            energy, forcespair1_1, forcespair2_1, forcespair1_2, forcespair2_2 = self._predict_boe_batch(boes_1, boes_2)

            forces = np.vstack([forcespair1_1, forcespair2_1, forcespair1_2, forcespair2_2]).astype(np.float64)
            restored_forces = np.zeros_like(forces)
            restored_forces[self.ref_indices_cache] = forces
            return energy, restored_forces


def initialize_velocities(masses: np.ndarray, temperature: float) -> np.ndarray:
    """Initialize Maxwell-Boltzmann velocities (A/fs)."""
    velocities = np.random.normal(
        scale=np.sqrt(kB * temperature / masses[:, None]),
        size=(len(masses), 3),
    )
    velocities -= np.mean(velocities, axis=0)
    return velocities


def calculate_temperature(velocities: np.ndarray, masses: np.ndarray, freedom: int) -> float:
    """Calculate instantaneous temperature from velocities."""
    kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))
    return 2 * kinetic_energy / (freedom * kB)


def calculate_com_velocity(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return the mass-weighted center-of-mass velocity."""
    total_mass = np.sum(masses)
    return np.sum(masses[:, None] * velocities, axis=0) / total_mass


def remove_com_velocity(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Project out center-of-mass drift from the velocity field."""
    return velocities - calculate_com_velocity(velocities, masses)


def remove_net_force(forces: np.ndarray) -> np.ndarray:
    """Project forces onto the zero-total-force subspace."""
    return forces - np.mean(forces, axis=0, keepdims=True)


def write_velocity_snapshot(path: Path, velocities: np.ndarray, append: bool) -> None:
    """Write one velocity matrix, separated from adjacent snapshots by a blank line."""
    with open(path, "a" if append else "w") as handle:
        for vx, vy, vz in velocities:
            handle.write(f"{vx:.10f} {vy:.10f} {vz:.10f}\n")
        handle.write("\n")


def calculate_min_distance(frac_coords: np.ndarray, lat_matrix: np.ndarray) -> float:
    """Return the minimum pair distance using the minimum-image convention."""
    if len(frac_coords) < 2:
        return np.inf

    frac_diffs = frac_coords[np.newaxis, :, :] - frac_coords[:, np.newaxis, :]
    frac_diffs -= np.round(frac_diffs)
    cart_diffs = frac_diffs @ lat_matrix
    distances = np.linalg.norm(cart_diffs, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def run_md_simulation(
    poscar: str,
    ref_poscar: str | None,
    mlff: CPUDualMLFF,
    temperature: float = 300.0,
    dt: float = 0.0001,
    num_steps: int = 20000,
    checkpoint_interval: int = 1000,
    traj_interval: int = 10,
    output_dir: str = "./md/md_with_constraint0/",
    restart: bool = False,
    seed: int = None,
    max_velocity: float = 50.0,
    remove_com_drift: bool = False,
    remove_net_force_drift: bool = False,
    energy_offset_per_20: float = -428.8,
    ensemble: str = "nvt",
    init_velocities_path: str | None = None,
):
    """Run NVT/NVE MD with the same loop and diagnostics as md_cpu.py."""
    ensemble = ensemble.lower().strip()
    if ensemble not in ("nvt", "nve"):
        raise ValueError(f"Unsupported ensemble={ensemble}. Expected one of: nvt, nve.")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ref_struct = Structure.from_file(ref_poscar) if ref_poscar is not None else Structure.from_file(poscar)

    logfile = output_path / "md_log.csv"
    trajfile = output_path / "traj.xyz"
    velocity_traj_file = output_path / "velocity.dat"
    checkpoint_file = output_path / "checkpoint.npz"
    final_struct_file = output_path / "final_struct.vasp"
    velocities_file = output_path / "final_velocities.csv"

    if restart and checkpoint_file.exists():
        print(f"Restarting from checkpoint: {checkpoint_file}")
        checkpoint = np.load(checkpoint_file, allow_pickle=True)

        struct = Structure.from_dict(checkpoint["struct_dict"].item())
        velocities = checkpoint["velocities"]
        if remove_com_drift:
            velocities = remove_com_velocity(velocities, masses=np.array([p_mass for _ in range(len(velocities))]))
        friction = float(checkpoint["friction"]) if ensemble == "nvt" else 0.0
        step_start = int(checkpoint["step"]) + 1
        log_data = pd.read_csv(logfile).to_dict("list")

        num_atoms = len(struct)
        boe_num = int(num_atoms / 4)
        energy_offset = (energy_offset_per_20 / 20.0) * boe_num
        masses = np.array([p_mass for _ in range(num_atoms)])
        freedom = 3 * num_atoms - 3
        tau = 0.005
        Q = freedom * kB * temperature * (tau ** 2) if ensemble == "nvt" else None

        energy, forces = mlff.getForceEnergy(structure=struct, ref_struct=ref_struct)
        if remove_net_force_drift:
            forces = remove_net_force(forces)

        lat_matrix = struct.lattice.matrix
        inv_lat_matrix = mlff._inv_lat_matrix
        cart_coords = struct.cart_coords.copy()
        species = struct.species
        lattice = struct.lattice

        print(f"Restarting from step {step_start}")
    else:
        struct = Structure.from_file(poscar)
        num_atoms = len(struct)
        boe_num = int(num_atoms / 4)
        energy_offset = (energy_offset_per_20 / 20.0) * boe_num
        masses = np.array([p_mass for _ in range(num_atoms)])

        if restart:
            print("No checkpoint found, starting new simulation")

        if init_velocities_path is not None:
            vel_df = pd.read_csv(init_velocities_path)
            required_cols = ("Vx", "Vy", "Vz")
            for col in required_cols:
                if col not in vel_df.columns:
                    raise ValueError(
                        f"Initial velocity file missing column '{col}': {init_velocities_path}"
                    )
            velocities = vel_df.loc[:, required_cols].to_numpy(dtype=np.float64)
            if velocities.shape != (num_atoms, 3):
                raise ValueError(
                    f"Initial velocity shape mismatch: expected {(num_atoms, 3)}, got {velocities.shape}"
                )
        else:
            velocities = initialize_velocities(masses, temperature)
        if remove_com_drift:
            velocities = remove_com_velocity(velocities, masses)
        energy, forces = mlff.getForceEnergy(structure=struct, ref_struct=ref_struct)
        if remove_net_force_drift:
            forces = remove_net_force(forces)

        lat_matrix = struct.lattice.matrix
        inv_lat_matrix = mlff._inv_lat_matrix
        cart_coords = struct.cart_coords.copy()
        species = struct.species
        lattice = struct.lattice

        friction = 0.0
        freedom = 3 * num_atoms - 3
        tau = 0.005
        Q = freedom * kB * temperature * (tau ** 2) if ensemble == "nvt" else None
        step_start = 0

        log_data = {
            "Time_ps": [],
            "Temperature_K": [],
            "Kinetic_Energy_eV": [],
            "Potential_Energy_eV": [],
            "Total_Energy_eV": [],
            "Friction": [],
            "Max_Force_eVA": [],
            "Max_Speed_Aps": [],
            "Min_Distance_A": [],
            "Total_Force_X_eVA": [],
            "Total_Force_Y_eVA": [],
            "Total_Force_Z_eVA": [],
            "Total_Force_Norm_eVA": [],
            "COM_Velocity_X_Aps": [],
            "COM_Velocity_Y_Aps": [],
            "COM_Velocity_Z_Aps": [],
            "COM_Velocity_Norm_Aps": [],
        }

    for key in (
        "Total_Force_X_eVA",
        "Total_Force_Y_eVA",
        "Total_Force_Z_eVA",
        "Total_Force_Norm_eVA",
        "COM_Velocity_X_Aps",
        "COM_Velocity_Y_Aps",
        "COM_Velocity_Z_Aps",
        "COM_Velocity_Norm_Aps",
    ):
        log_data.setdefault(key, [])

    recent_temperatures = []
    start_time = time.time()

    # Report step-0 energy decomposition to avoid confusion between potential and total energy.
    kinetic_energy0 = 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))
    potential_energy0 = energy + energy_offset
    total_force0 = np.sum(forces, axis=0)
    print(
        "Initial state | "
        f"E_pot = {potential_energy0:.3f} eV | "
        f"E_kin = {kinetic_energy0:.3f} eV | "
        f"E_tot = {potential_energy0 + kinetic_energy0:.3f} eV | "
        f"|F_tot| = {np.linalg.norm(total_force0):.3f} eV/A"
    )

    try:
        for step in range(step_start, num_steps):
            kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))
            if ensemble == "nvt":
                friction += (0.5 * dt / Q) * (kinetic_energy - 0.5 * freedom * kB * temperature)

            thermostat_term = friction * velocities if ensemble == "nvt" else 0.0
            velocities = velocities + 0.5 * (forces / masses[:, None] - thermostat_term) * dt

            cart_coords = cart_coords + velocities * dt

            frac_coords = cart_coords @ inv_lat_matrix
            frac_coords -= np.floor(frac_coords)
            cart_coords = frac_coords @ lat_matrix
            energy, forces = mlff.getForceEnergy(frac_coords=frac_coords)
            if remove_net_force_drift:
                forces = remove_net_force(forces)

            total_force = np.sum(forces, axis=0)
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
            min_distance = calculate_min_distance(frac_coords, lat_matrix)

            thermostat_term = friction * velocities if ensemble == "nvt" else 0.0
            velocities = velocities + 0.5 * (forces / masses[:, None] - thermostat_term) * dt
            if remove_com_drift:
                velocities = remove_com_velocity(velocities, masses)

            com_velocity = calculate_com_velocity(velocities, masses)

            max_speed = np.max(np.linalg.norm(velocities, axis=1))
            if not np.isfinite(max_speed) or max_speed > max_velocity:
                raise RuntimeError(
                    f"Velocity blow-up detected at step {step}: max speed = {max_speed:.6g} A/ps "
                    f"(threshold {max_velocity:.6g} A/ps), max force = {max_force:.6g} eV/A, "
                    f"min distance = {min_distance:.6g} A, total force z = {total_force[2]:.6g} eV/A, "
                    f"COM vz = {com_velocity[2]:.6g} A/ps"
                )

            kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1))
            if ensemble == "nvt":
                friction += (0.5 * dt / Q) * (kinetic_energy - 0.5 * freedom * kB * temperature)

            inst_temp = calculate_temperature(velocities, masses, freedom)
            recent_temperatures.append(inst_temp)
            if len(recent_temperatures) > 100:
                recent_temperatures.pop(0)

            moving_avg_temp = np.mean(recent_temperatures)
            potential_energy = energy + energy_offset

            if step % traj_interval == 0:
                log_data["Time_ps"].append(step * dt)
                log_data["Temperature_K"].append(moving_avg_temp)
                log_data["Kinetic_Energy_eV"].append(kinetic_energy)
                log_data["Potential_Energy_eV"].append(potential_energy)
                log_data["Total_Energy_eV"].append(kinetic_energy + potential_energy)
                log_data["Friction"].append(friction)
                log_data["Max_Force_eVA"].append(max_force)
                log_data["Max_Speed_Aps"].append(float(max_speed))
                log_data["Min_Distance_A"].append(min_distance)
                log_data["Total_Force_X_eVA"].append(float(total_force[0]))
                log_data["Total_Force_Y_eVA"].append(float(total_force[1]))
                log_data["Total_Force_Z_eVA"].append(float(total_force[2]))
                log_data["Total_Force_Norm_eVA"].append(float(np.linalg.norm(total_force)))
                log_data["COM_Velocity_X_Aps"].append(float(com_velocity[0]))
                log_data["COM_Velocity_Y_Aps"].append(float(com_velocity[1]))
                log_data["COM_Velocity_Z_Aps"].append(float(com_velocity[2]))
                log_data["COM_Velocity_Norm_Aps"].append(float(np.linalg.norm(com_velocity)))
                pd.DataFrame(log_data).to_csv(logfile, index=False)

                with open(trajfile, "a" if step > 0 else "w") as handle:
                    handle.write(f"{num_atoms}\n")
                    handle.write(f"Step {step}, Time = {step * dt:.4f} ps, T = {moving_avg_temp:.1f} K\n")
                    for atom_i in range(num_atoms):
                        element = species[atom_i].symbol
                        x, y, z = cart_coords[atom_i]
                        handle.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")
                write_velocity_snapshot(velocity_traj_file, velocities, append=step > 0)

            if step % checkpoint_interval == 0 and step > step_start:
                struct = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
                np.savez(
                    checkpoint_file,
                    struct_dict=struct.as_dict(),
                    velocities=velocities,
                    friction=friction,
                    step=step,
                )

                elapsed = time.time() - start_time
                steps_done = step - step_start
                sec_per_step = elapsed / steps_done if steps_done > 0 else 0
                eta_h = sec_per_step * (num_steps - step) / 3600

                print(
                    f"Step {step}/{num_steps} | "
                    f"T = {moving_avg_temp:.1f} K | "
                    f"E_pot = {potential_energy:.3f} eV | "
                    f"E_kin = {kinetic_energy:.3f} eV | "
                    f"E_tot = {kinetic_energy + potential_energy:.3f} eV | "
                    f"max|F| = {max_force:.3f} eV/A | "
                    f"|F_tot| = {np.linalg.norm(total_force):.3f} eV/A | "
                    f"v_com,z = {com_velocity[2]:.4f} A/ps | "
                    f"min r = {min_distance:.3f} A | "
                    f"Elapsed: {elapsed / 3600:.2f}h | "
                    f"{sec_per_step:.3f} s/step | "
                    f"ETA: {eta_h:.2f}h"
                )

        print(f"Simulation completed successfully in {(time.time() - start_time) / 3600:.2f} hours")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as error:
        print(f"\nError during simulation: {error}")
        import traceback
        traceback.print_exc()
    finally:
        print("Saving final state...")

        struct = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
        Poscar(struct).write_file(str(final_struct_file))

        pd.DataFrame(
            {
                "Vx": velocities[:, 0],
                "Vy": velocities[:, 1],
                "Vz": velocities[:, 2],
            }
        ).to_csv(velocities_file, index=False)

        pd.DataFrame(log_data).to_csv(logfile, index=False)

        if checkpoint_file.exists():
            checkpoint_file.unlink()

        print(f"Results saved to {output_dir}")
        print(f"Final structure: {final_struct_file}")
        print(f"Final velocities: {velocities_file}")
        print(f"Log file: {logfile}")
        print(f"Trajectory: {trajfile}")


def main():
    parser = argparse.ArgumentParser(description="Run dual-model NVT/NVE MD simulation on CPU")
    parser.add_argument("--poscar", type=str, default="POSCAR_REV.vasp", help="Input/init POSCAR file for current MD state")
    parser.add_argument(
        "--ref-poscar",
        type=str,
        default=None,
        help="Reference POSCAR used for BOE reference mapping; defaults to --poscar when omitted.",
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Target temperature (K)")
    parser.add_argument("--dt", type=float, default=0.0001, help="Timestep (ps)")
    parser.add_argument("--steps", type=int, default=50000, help="Total number of MD steps")
    parser.add_argument("--checkpoint", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--traj", type=int, default=100, help="Trajectory save interval")
    parser.add_argument("--output", type=str, default="./md/md_with_constraint1/", help="Output directory")
    parser.add_argument("--restart", action="store_true", help="Restart from checkpoint")
    parser.add_argument(
        "--ensemble",
        type=str,
        default="nvt",
        choices=("nvt", "nve"),
        help="Integrator ensemble: nvt (Nose-Hoover thermostat) or nve (microcanonical).",
    )
    parser.add_argument(
        "--init-velocities",
        type=str,
        default=None,
        help="Optional CSV file with columns Vx,Vy,Vz to initialize/continue velocities when not using --restart.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-velocity", type=float, default=50.0, help="Stop if any atomic speed exceeds this value (A/ps)")
    parser.add_argument("--model1", type=str, default="./result/result_sweep_000-050/dw_0p050/model1.pth", help="Path to dual model-1 weights")
    parser.add_argument("--model2", type=str, default="./result/result_sweep_000-050/dw_0p050/model2.pth", help="Path to dual model-2 weights")
    parser.add_argument("--stat1", type=str, default="./all_temp/ds_train_stat1.csv", help="Path to mode-1 normalization stats")
    parser.add_argument("--stat2", type=str, default="./all_temp/ds_train_stat2.csv", help="Path to mode-2 normalization stats")
    parser.add_argument("--width", type=int, default=96, help="Hidden width c for the dual subnet models")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        choices=(30, 42),
        help="Dual subnet input feature dimension. Use 42 for center-aware models. Defaults to --stat-n-col.",
    )
    parser.add_argument(
        "--energy-offset-per-20",
        "--energy-offset",
        dest="energy_offset_per_20",
        type=float,
        default=-428.8,
        help="Base energy offset for 20 BOEs; MD uses energy_offset = (value/20)*BOE_num and potential = raw_pred_energy + energy_offset.",
    )
    parser.add_argument(
        "--stat-n-col",
        type=int,
        default=30,
        help="Use only the first N columns from stat CSVs for normalization (set to model input dim, e.g. 42).",
    )
    parser.add_argument("--remove-com-velocity", action="store_true", help="Remove center-of-mass velocity after each velocity update")
    parser.add_argument("--remove-net-force", action="store_true", help="Subtract the mean force from all atoms after each force evaluation")
    args = parser.parse_args()

    input_dim = args.input_dim if args.input_dim is not None else args.stat_n_col
    if input_dim not in (30, 42):
        raise ValueError(f"Unsupported input dim: {input_dim}. Expected 30 or 42.")

    print("Loading dual MLFF models...")
    dual_model = DualEnergyModelContainer(
        SubNet(input_size=input_dim, output_size=7, c=args.width),
        SubNet(input_size=input_dim, output_size=7, c=args.width),
    )
    dual_model.load(args.model1, args.model2)

    stat_df1 = pd.read_csv(args.stat1, index_col=0)
    stat_df2 = pd.read_csv(args.stat2, index_col=0)

    model_input_dim = int(dual_model.model1.FC1.in_features)
    expected_stat_cols = args.stat_n_col
    if expected_stat_cols != model_input_dim:
        raise ValueError(
            f"Dimension mismatch: --stat-n-col={expected_stat_cols}, model input dim={model_input_dim}. "
            "They must match."
        )
    if stat_df1.shape[1] < expected_stat_cols or stat_df2.shape[1] < expected_stat_cols:
        raise ValueError(
            f"Stat CSV columns insufficient: stat1 has {stat_df1.shape[1]}, stat2 has {stat_df2.shape[1]}, "
            f"but --stat-n-col={expected_stat_cols}."
        )
    for stat_name, stat_df in (("stat1", stat_df1), ("stat2", stat_df2)):
        for row_name in ("mean", "std"):
            if row_name not in stat_df.index:
                raise ValueError(f"{stat_name} is missing required row '{row_name}'.")

    print("MD startup config checks:")
    print(f"  requested_input_dim = {input_dim}")
    print(f"  model_input_dim = {model_input_dim}")
    print(f"  stat_n_col = {expected_stat_cols}")
    print(f"  stat1_columns = {stat_df1.shape[1]}")
    print(f"  stat2_columns = {stat_df2.shape[1]}")
    startup_struct = Structure.from_file(args.poscar)
    startup_boe_num = int(len(startup_struct) / 4)
    ref_poscar = args.ref_poscar if args.ref_poscar is not None else args.poscar
    startup_energy_offset = (args.energy_offset_per_20 / 20.0) * startup_boe_num
    print(f"  BOE_num_from_poscar = {startup_boe_num}")
    print(f"  init_poscar = {args.poscar}")
    print(f"  ref_poscar = {ref_poscar}")
    print(f"  energy_offset_per_20 = {args.energy_offset_per_20:.6f} eV")
    print(f"  scaled_energy_offset = {startup_energy_offset:.6f} eV")
    print(f"  ensemble = {args.ensemble}")
    if args.init_velocities is not None:
        print(f"  init_velocities = {args.init_velocities}")
    print("  potential reconstruction = raw_pred_energy + scaled_energy_offset")

    mlff = CPUDualMLFF(
        dual_model=dual_model,
        stat_df1=stat_df1,
        stat_df2=stat_df2,
        stat_n_col=args.stat_n_col,
    )

    run_md_simulation(
        poscar=args.poscar,
        ref_poscar=ref_poscar,
        mlff=mlff,
        temperature=args.temperature,
        dt=args.dt,
        num_steps=args.steps,
        checkpoint_interval=args.checkpoint,
        traj_interval=args.traj,
        output_dir=args.output,
        restart=args.restart,
        seed=args.seed,
        max_velocity=args.max_velocity,
        remove_com_drift=args.remove_com_velocity,
        remove_net_force_drift=args.remove_net_force,
        energy_offset_per_20=args.energy_offset_per_20,
        ensemble=args.ensemble,
        init_velocities_path=args.init_velocities,
    )


if __name__ == "__main__":
    main()