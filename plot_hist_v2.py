"""Sample [500, 4000) and save zero-based indices into the full raw trajectory."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update(
	{
		"font.family": "sans-serif",
		"font.sans-serif": ["Arial", "DejaVu Sans"],
		"font.size": 11,
		"axes.titlesize": 13,
		"axes.labelsize": 12,
		"axes.linewidth": 1.0,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10,
		"xtick.direction": "in",
		"ytick.direction": "in",
		"xtick.major.size": 4,
		"ytick.major.size": 4,
		"legend.fontsize": 10,
		"legend.frameon": False,
		"figure.dpi": 120,
		"savefig.dpi": 300,
		"savefig.bbox": "tight",
	}
)


BASE_DIR = os.path.dirname(__file__)
TEMPERATURES = ["100", "200", "300", "400", "500", "600"]
COLORS = {
	"100": "tab:cyan",
	"200": "tab:blue",
	"300": "tab:green",
	"400": "tab:red",
	"400": "tab:red",
	"500": "tab:purple",
	"600": "tab:brown",
}
SAMPLE_COUNTS = {
	"100": 800,
	"200": 800,
	"300": 800,
	"400": 800,
	"500": 800,
	"600": 800,
}
START_STEP = 500
MAX_INITIAL_STEPS = 4000
THINNING_STRIDE = 2
ENERGY_BINS = 10
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(BASE_DIR, "low_temp_v2")
OUTPUT_FIGURE = "energy_hist_proportional_scheme.png"
OUTPUT_INDEX_CSV = "selected_frames_proportional_scheme.csv"
OUTPUT_TREND_FIGURE = "energy_time_trend_all_temperatures_v2.png"
OUTPUT_MERGED_SELECTED_FIGURE = "energy_hist_selected_merged_all_temperatures_v2.png"
OUTPUT_GAUSSIAN_TXT = "gaussian_fit_parameters.txt"
OUTPUT_BIN_COUNTS_CSV = "selected_frames_proportional_bin_counts.csv"


def load_energies(base_dir: str, file_name: str = "all_energy.txt") -> np.ndarray:
	"""Load energies from the given temperature directory and keep the first window only."""
	path = os.path.join(base_dir, file_name)
	data = np.loadtxt(path).reshape(-1)
	if len(data) < MAX_INITIAL_STEPS or not np.isfinite(data).all():
		raise ValueError(f"{path}: expected at least {MAX_INITIAL_STEPS} finite energies.")
	return data[START_STEP:MAX_INITIAL_STEPS]


def gaussian_pdf(x_values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
	"""Evaluate a normalized Gaussian probability density function."""
	if sigma <= 0:
		return np.zeros_like(x_values, dtype=float)
	coefficient = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
	exponent = -0.5 * ((x_values - mu) / sigma) ** 2
	return coefficient * np.exp(exponent)


def fit_gaussian(energies: np.ndarray) -> dict:
	"""Fit a single Gaussian by maximum-likelihood estimates for mu and sigma."""
	mu = float(np.mean(energies))
	sigma = float(np.std(energies, ddof=0))
	sample_sigma = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0
	return {
		"n": int(len(energies)),
		"mu": mu,
		"sigma": sigma,
		"sample_sigma": sample_sigma,
		"min": float(np.min(energies)),
		"max": float(np.max(energies)),
	}


def proportional_histogram_sample_indices(
	energies: np.ndarray,
	sample_size: int,
	stride: int,
	n_bins: int,
	rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
	"""Sample the same n/m ratio from every fixed histogram bin."""
	if energies.size == 0:
		return np.array([], dtype=int), pd.DataFrame()

	candidate_indices = np.arange(0, len(energies), stride, dtype=int)
	if sample_size > len(candidate_indices):
		raise ValueError(
			f"Requested {sample_size} samples from only {len(candidate_indices)} candidates. "
			"Reduce sample_size or the thinning stride."
		)

	candidate_energies = energies[candidate_indices]
	bin_edges = np.histogram_bin_edges(candidate_energies, bins=n_bins)
	bin_ids = np.digitize(candidate_energies, bin_edges[1:-1], right=False)
	bin_members = [candidate_indices[bin_ids == bin_idx].copy() for bin_idx in range(n_bins)]
	candidate_counts = np.array([len(members) for members in bin_members], dtype=int)

	ratio = sample_size / len(candidate_indices)
	expected_counts = candidate_counts * ratio
	target_counts = np.floor(expected_counts).astype(int)
	remainder = sample_size - int(target_counts.sum())

	if remainder > 0:
		fractional_order = np.argsort(expected_counts - target_counts, kind="mergesort")[::-1]
		for bin_idx in fractional_order:
			if remainder == 0:
				break
			if target_counts[bin_idx] >= candidate_counts[bin_idx]:
				continue
			target_counts[bin_idx] += 1
			remainder -= 1

	if remainder > 0:
		for bin_idx in np.argsort(candidate_counts - target_counts)[::-1]:
			if remainder == 0:
				break
			capacity = candidate_counts[bin_idx] - target_counts[bin_idx]
			if capacity <= 0:
				continue
			take_extra = min(capacity, remainder)
			target_counts[bin_idx] += take_extra
			remainder -= take_extra

	if remainder != 0:
		raise RuntimeError("Unable to allocate proportional samples across histogram bins.")

	selected = []
	bin_rows = []
	for bin_idx, members in enumerate(bin_members):
		if len(members) > 0:
			rng.shuffle(members)
		take = int(target_counts[bin_idx])
		if take > 0:
			selected.append(members[:take])
		bin_rows.append(
			{
				"bin_index": bin_idx,
				"energy_left_eV": float(bin_edges[bin_idx]),
				"energy_right_eV": float(bin_edges[bin_idx + 1]),
				"candidate_count": int(candidate_counts[bin_idx]),
				"selected_count": take,
				"selected_ratio": float(take / candidate_counts[bin_idx]) if candidate_counts[bin_idx] else 0.0,
			}
		)

	selected_indices = np.sort(np.concatenate(selected)) if selected else np.array([], dtype=int)
	return selected_indices, pd.DataFrame(bin_rows)


def build_sampling_table() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
	"""Apply proportional histogram sampling and store the selected frame list."""
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	rng = np.random.default_rng(RANDOM_SEED)
	rows = []
	bin_count_tables = []
	fit_results = {}

	for temp in TEMPERATURES:
		temp_dir = os.path.join(BASE_DIR, temp)
		energies = load_energies(temp_dir)
		if energies.size == 0:
			continue

		fit_results[temp] = fit_gaussian(energies)
		selected_indices, bin_counts = proportional_histogram_sample_indices(
			energies=energies,
			sample_size=SAMPLE_COUNTS[temp],
			stride=THINNING_STRIDE,
			n_bins=ENERGY_BINS,
			rng=rng,
		)
		bin_counts.insert(0, "temperature_K", int(temp))
		bin_count_tables.append(bin_counts)

		for index in selected_indices:
			rows.append(
				{
					"temperature_K": int(temp),
					"frame_index": int(index + START_STEP),
					"energy_eV": float(energies[index]),
				}
			)

	selection_df = pd.DataFrame(rows)
	bin_count_df = pd.concat(bin_count_tables, ignore_index=True) if bin_count_tables else pd.DataFrame()
	selection_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_INDEX_CSV), index=False)
	bin_count_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_BIN_COUNTS_CSV), index=False)
	write_gaussian_fit_summary(fit_results)
	return selection_df, bin_count_df, fit_results


def write_gaussian_fit_summary(fit_results: dict):
	"""Save Gaussian fit parameters for each temperature as a text report."""
	output_path = os.path.join(OUTPUT_DIR, OUTPUT_GAUSSIAN_TXT)
	with open(output_path, "w", encoding="utf-8") as handle:
		handle.write("Gaussian fit parameters for original energy distributions\n")
		handle.write(f"steps: {START_STEP}-{MAX_INITIAL_STEPS}\n")
		handle.write(f"histogram_bins: {ENERGY_BINS}\n")
		handle.write(f"thinning_stride_for_sampling: {THINNING_STRIDE}\n\n")
		for temp in TEMPERATURES:
			if temp not in fit_results:
				continue
			fit = fit_results[temp]
			handle.write(f"{temp} K\n")
			handle.write(f"  n = {fit['n']}\n")
			handle.write(f"  mu_eV = {fit['mu']:.10f}\n")
			handle.write(f"  sigma_eV = {fit['sigma']:.10f}\n")
			handle.write(f"  sample_sigma_eV = {fit['sample_sigma']:.10f}\n")
			handle.write(f"  min_eV = {fit['min']:.10f}\n")
			handle.write(f"  max_eV = {fit['max']:.10f}\n\n")
	print(f"Gaussian fit parameters saved to {output_path}")


def plot_energy_histograms(
	selection_df: pd.DataFrame,
	fit_results: dict,
	bins: int = 50,
	output: str = OUTPUT_FIGURE,
):
	"""Plot the candidate pool, Gaussian fits, and proportional sampled subset."""
	fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=False)
	all_candidate_energies = []
	all_selected_energies = []

	for temp in TEMPERATURES:
		temp_dir = os.path.join(BASE_DIR, temp)
		energies = load_energies(temp_dir)
		if energies.size == 0:
			continue

		selected = selection_df.loc[selection_df["temperature_K"] == int(temp), "energy_eV"].to_numpy()
		color = COLORS[temp]
		all_candidate_energies.append(energies)
		all_selected_energies.append(selected)

		axes[0].hist(
			energies,
			bins=bins,
			color=color,
			alpha=0.35,
			density=True,
			label=f"{temp} K original (n={len(energies)})",
		)

		fit = fit_results[temp]
		x_values = np.linspace(np.min(energies), np.max(energies), 500)
		axes[0].plot(
			x_values,
			gaussian_pdf(x_values, fit["mu"], fit["sigma"]),
			color=color,
			linewidth=2.0,
			label=f"{temp} K Gaussian fit (mu={fit['mu']:.3f}, sigma={fit['sigma']:.3f})",
		)

		axes[1].hist(
			selected,
			bins=bins,
			histtype="step",
			linewidth=2.2,
			color=color,
			density=True,
			label=f"{temp} K selected (n={len(selected)})",
		)

	axes[0].set_title("Original Energy Distribution")
	axes[1].set_title("Histogram-Bin Training Subset")

	axes[0].text(
		0,
		1.06,
		"(a)",
		transform=axes[0].transAxes,
		fontsize=18,
		fontweight="bold",
		va="bottom",
		ha="left",
	)
	axes[1].text(
		0,
		1.06,
		"(b)",
		transform=axes[1].transAxes,
		fontsize=18,
		fontweight="bold",
		va="bottom",
		ha="left",
	)

	merged_candidate = np.concatenate(all_candidate_energies)
	merged_selected = np.concatenate(all_selected_energies)
	plot_reference = np.concatenate([merged_candidate, merged_selected])
	x_min, x_max = np.percentile(plot_reference, [0.5, 99.5])
	padding = 0.05 * (x_max - x_min)
	x_min -= padding
	x_max += padding

	for ax in axes:
		ax.set_xlim(x_min, x_max)
		ax.set_xlabel("Total energy (eV)")
		ax.set_ylabel("Probability density")
		ax.legend(loc="best")
		ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)

	candidate_clipped = ((merged_candidate < x_min) | (merged_candidate > x_max)).sum()
	selected_clipped = ((merged_selected < x_min) | (merged_selected > x_max)).sum()
	axes[0].text(
		0.02,
		0.92,
		f"clipped outside x-range: {candidate_clipped}",
		transform=axes[0].transAxes,
		fontsize=10,
	)
	axes[1].text(
		0.02,
		0.92,
		f"clipped outside x-range: {selected_clipped}",
		transform=axes[1].transAxes,
		fontsize=10,
	)

	plt.suptitle(
		f"Proportional Histogram-Bin Sampling (steps {START_STEP}-{MAX_INITIAL_STEPS}, "
		f"stride={THINNING_STRIDE}, bins={ENERGY_BINS})",
		y=0.98,
	)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, output), dpi=300)
	print(f"Histogram saved to {os.path.join(OUTPUT_DIR, output)}")
	print(f"Selected frame list saved to {os.path.join(OUTPUT_DIR, OUTPUT_INDEX_CSV)}")
	print(f"Bin counts saved to {os.path.join(OUTPUT_DIR, OUTPUT_BIN_COUNTS_CSV)}")
	print(f"Plot x-range: {x_min:.6f} to {x_max:.6f} eV")
	print(f"Candidate frames clipped from plot: {candidate_clipped}")
	print(f"Selected frames clipped from plot: {selected_clipped}")

	if all_selected_energies:
		print(
			"Merged selected set: "
			f"n={len(merged_selected)}, mean={merged_selected.mean():.6f}, std={merged_selected.std():.6f}, "
			f"min={merged_selected.min():.6f}, max={merged_selected.max():.6f}"
		)


def plot_energy_time_trend(output: str = OUTPUT_TREND_FIGURE):
	"""Plot total energy versus MD step for all temperatures on the same figure."""
	plt.figure(figsize=(10, 6))
	all_energies = []

	for temp in TEMPERATURES:
		temp_dir = os.path.join(BASE_DIR, temp)
		energies = load_energies(temp_dir)
		if energies.size == 0:
			continue

		steps = np.arange(START_STEP, START_STEP + len(energies), dtype=int)
		all_energies.append(energies)
		plt.plot(
			steps,
			energies,
			color=COLORS[temp],
			linewidth=1.2,
			alpha=0.8,
			label=f"{temp} K",
		)

	if not all_energies:
		print("No energy files were found for the time-trend plot.")
		return

	merged_energies = np.concatenate(all_energies)
	y_min, y_max = np.percentile(merged_energies, [0.5, 99.5])
	padding = 0.05 * (y_max - y_min)
	y_min -= padding
	y_max += padding
	clipped = ((merged_energies < y_min) | (merged_energies > y_max)).sum()

	plt.ylim(y_min, y_max)
	plt.xlabel("MD step")
	plt.ylabel("Total energy (eV)")
	plt.title("Energy Trajectory by Temperature")
	plt.legend(loc="best")
	plt.grid(alpha=0.3, linestyle="--", linewidth=0.6)
	plt.text(
		0.02,
		0.95,
		f"seed={RANDOM_SEED}, clipped outside y-range: {clipped}",
		transform=plt.gca().transAxes,
		fontsize=10,
		verticalalignment="top",
	)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, output), dpi=300)
	print(f"Energy-time trend saved to {os.path.join(OUTPUT_DIR, output)}")
	print(f"Trend plot y-range: {y_min:.6f} to {y_max:.6f} eV")
	print(f"Trend frames clipped from plot: {clipped}")


def plot_merged_selected_histogram(
	selection_df: pd.DataFrame,
	bins: int = 50,
	output: str = OUTPUT_MERGED_SELECTED_FIGURE,
):
	"""Plot one histogram for all selected energies merged across temperatures."""
	selected = selection_df["energy_eV"].to_numpy()
	if selected.size == 0:
		print("No selected data found for merged selected histogram.")
		return

	plt.figure(figsize=(9, 6))
	plt.hist(
		selected,
		bins=bins,
		color="tab:blue",
		alpha=0.65,
		edgecolor="black",
		linewidth=0.7,
		density=True,
	)

	x_min, x_max = np.percentile(selected, [0.5, 99.5])
	padding = 0.05 * (x_max - x_min)
	x_min -= padding
	x_max += padding
	clipped = ((selected < x_min) | (selected > x_max)).sum()

	plt.xlim(x_min, x_max)
	plt.xlabel("Total energy (eV)")
	plt.ylabel("Probability density")
	plt.title("Merged Selected Energy Distribution")
	plt.grid(alpha=0.3, linestyle="--", linewidth=0.6)
	plt.text(
		0.62,
		0.95,
		f"n={len(selected)}, seed={RANDOM_SEED}, clipped={clipped}",
		transform=plt.gca().transAxes,
		fontsize=10,
		verticalalignment="top",
	)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, output), dpi=300)
	print(f"Merged selected histogram saved to {os.path.join(OUTPUT_DIR, output)}")
	print(
		"Merged selected summary: "
		f"mean={selected.mean():.6f}, std={selected.std():.6f}, "
		f"min={selected.min():.6f}, max={selected.max():.6f}"
	)


if __name__ == "__main__":
	selection_df, bin_count_df, fit_results = build_sampling_table()
	plot_energy_histograms(selection_df, fit_results, bins=50, output=OUTPUT_FIGURE)
	plot_merged_selected_histogram(selection_df, bins=50, output=OUTPUT_MERGED_SELECTED_FIGURE)
	plot_energy_time_trend(output=OUTPUT_TREND_FIGURE)
