import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import gen_center_all_data as centers
import gen_force_all_data as forces
import plot_hist_v2 as sampling
from dataset_center import VEFeaturesEnergyDataset
from md_cpu_dual import CPUDualMLFF
from model_two_force import DualEnergyModelContainer, SubNet
from stat_save import save_train_stats


class FrameAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "low_temp_v2"
        self.output.mkdir()
        for temperature in sampling.TEMPERATURES:
            directory = self.root / temperature
            directory.mkdir()
            energies = np.arange(4500, dtype=float) / 100 + int(temperature)
            np.savetxt(directory / "all_energy.txt", energies)
        self.enterContext(patch.object(sampling, "BASE_DIR", str(self.root)))
        self.enterContext(patch.object(sampling, "OUTPUT_DIR", str(self.output)))
        self.enterContext(patch.object(centers, "BASE_DIR", str(self.root)))

    def sample(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return sampling.build_sampling_table()[0]

    def test_absolute_indices_counts_and_energy_roundtrip(self):
        selection = self.sample()
        self.assertEqual(len(selection), 4800)
        for temperature in sampling.TEMPERATURES:
            saved = selection.loc[selection.temperature_K == int(temperature)]
            self.assertEqual(len(saved), 800)
            indices = saved.frame_index.to_numpy()
            self.assertTrue(((indices >= 500) & (indices < 4000)).all())
            self.assertTrue(((indices - 500) % 2 == 0).all())
            raw = np.loadtxt(self.root / temperature / "all_energy.txt")
            np.testing.assert_allclose(saved.energy_eV, raw[indices], rtol=0, atol=1e-8)
        grouped = centers.load_frame_table(str(self.output / sampling.OUTPUT_INDEX_CSV))
        self.assertEqual(sum(len(rows) for rows in grouped.values()), 4800)

    def test_legacy_window_relative_indices_are_rejected(self):
        selection = self.sample()
        selection["frame_index"] -= 500
        path = self.output / sampling.OUTPUT_INDEX_CSV
        selection.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "energy/frame mismatch"):
            centers.load_frame_table(str(path))

    def test_missing_selection_does_not_silently_use_all_frames(self):
        with self.assertRaises(FileNotFoundError):
            centers.load_frame_table(str(self.output / "missing.csv"))

    def test_structure_frame_count_and_coordinates(self):
        path = self.root / "XDATCAR"
        path.write_text(
            "example\n1\n1 0 0\n0 1 0\n0 0 1\nP\n2\n"
            "Direct configuration= 1\n0 0 0\n0.1 0.2 0.3\n"
            "Direct configuration= 2\n0.4 0.5 0.6\n0.7 0.8 0.9\n",
            encoding="ascii",
        )
        frames, _ = centers.read_selected_fractional_coords(str(path), [1], 2)
        np.testing.assert_allclose(frames[0][0], [0.4, 0.5, 0.6])
        with self.assertRaisesRegex(ValueError, "2 structure frames but 3 energies"):
            centers.read_selected_fractional_coords(str(path), [1], 3)

    def test_force_labels_must_match_raw_frame(self):
        metadata = pd.DataFrame(
            {"temperature_K": [100], "frame_index": [0], "energy_eV": [-2.0]}
        )
        with (
            patch.object(forces, "resolve_force_file", return_value="forces_only.dat"),
            patch.object(forces, "load_energy_series", return_value=np.array([-1.0, -2.0])),
            patch.object(forces.np, "loadtxt", return_value=np.zeros((160, 3))),
        ):
            with self.assertRaisesRegex(ValueError, "energy/frame mismatch"):
                forces.build_force_dataframe(metadata, str(Path(__file__).resolve().parents[1] / "data1"), str(self.output))

    def test_incomplete_force_blocks_are_rejected(self):
        metadata = pd.DataFrame(
            {"temperature_K": [100], "frame_index": [0], "energy_eV": [-1.0]}
        )
        with (
            patch.object(forces, "resolve_force_file", return_value="forces_only.dat"),
            patch.object(forces.np, "loadtxt", return_value=np.zeros((81, 3))),
        ):
            with self.assertRaisesRegex(ValueError, "complete 80-atom blocks"):
                forces.build_force_dataframe(metadata, str(Path(__file__).resolve().parents[1] / "data1"), str(self.output))


class NormalizationTests(unittest.TestCase):
    def test_export_and_md_match_training_including_constant_features(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, 42))
        features[:, 0] = 0.25
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(42)])
        for name in ("fx0", "fy0", "fz0", "fx1", "fy1", "fz1"):
            frame[name] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frame.to_csv(path / "train_ve.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = VEFeaturesEnergyDataset(frame, pd.DataFrame({"energy": [0.0, 0.0]}), 42)
                dataset.normalize_X()
                save_train_stats(str(path), 42, str(path / "stats.csv"))
            stats = pd.read_csv(path / "stats.csv", index_col=0)
            model = DualEnergyModelContainer(
                SubNet(42, 7, 4), SubNet(42, 7, 4), device="cpu"
            )
            wrapper = CPUDualMLFF(model, stats, stats, 42)
            raw = torch.tensor(features, dtype=torch.float32)
            for mean, std in (
                (wrapper.df1_mean, wrapper.df1_std),
                (wrapper.df2_mean, wrapper.df2_std),
            ):
                np.testing.assert_array_equal(mean.numpy()[0], dataset.pca_mean)
                np.testing.assert_array_equal(std.numpy()[0], dataset.pca_std)
                np.testing.assert_array_equal(((raw - mean) / std).numpy(), dataset.features)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            git = shutil.which("git")
            candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe" if git else None
            if candidate is None or not candidate.is_file():
                self.skipTest("Git Bash is required for extraction tests on Windows.")
            bash = str(candidate)
        if bash is None:
            self.skipTest("Bash is required for extraction tests.")
        self.bash = bash
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_script(self, name):
        script = Path(__file__).resolve().parents[1] / name
        return subprocess.run(
            [self.bash, str(script)],
            cwd=self.root,
            env={**os.environ, "NATOMS": "2"},
            text=True,
            capture_output=True,
            check=False,
        )

    def test_extracts_all_frames_without_tail(self):
        (self.root / "OUTCAR").write_text(
            " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
            " ---------------------------------------------------------\n"
            " 0 0 0 1 2 3\n 0 0 0 4 5 6\n"
            " energy  without entropy= -10.5  energy(sigma->0) = -10.6\n"
            " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
            " ---------------------------------------------------------\n"
            " 0 0 0 7 8 9\n 0 0 0 10 11 12\n"
            " energy  without entropy= -11.5  energy(sigma->0) = -11.6\n",
            encoding="ascii",
        )
        for script in ("ext_force.sh", "ext_energy.sh"):
            result = self.run_script(script)
            self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_array_equal(
            np.loadtxt(self.root / "forces_only.dat"),
            np.arange(1, 13).reshape(4, 3),
        )
        np.testing.assert_array_equal(
            np.loadtxt(self.root / "all_energy.txt"), [-10.5, -11.5]
        )

    def test_truncated_force_output_preserves_previous_file(self):
        (self.root / "OUTCAR").write_text(
            " POSITION TOTAL-FORCE\n ---\n 0 0 0 1 2 3\n", encoding="ascii"
        )
        output = self.root / "forces_only.dat"
        output.write_text("previous\n", encoding="ascii")
        result = self.run_script("ext_force.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "previous\n")
        self.assertEqual(list(self.root.glob("forces_only.dat.*")), [])

    def test_malformed_energy_preserves_previous_file(self):
        (self.root / "OUTCAR").write_text(
            " energy  without entropy= invalid\n", encoding="ascii"
        )
        output = self.root / "all_energy.txt"
        output.write_text("previous\n", encoding="ascii")
        result = self.run_script("ext_energy.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "previous\n")
        self.assertEqual(list(self.root.glob("all_energy.txt.*")), [])


if __name__ == "__main__":
    unittest.main()
