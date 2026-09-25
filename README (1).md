# Conformal Adaptive Martingales (CAM) — simulation code

Code accompanying the paper

> A. Laurenzi, M. Borrotti, D. Demaria, V. Zippo.
> *Conformal Adaptive Martingales: Distribution-Free Sequential Change-Point Detection with Finite-Sample False Alarm Control.*

The repository reproduces every figure and table of Section 6 (Simulation study) of the paper and of its Supplementary Material.

## Contents

| File | Description |
|---|---|
| `cam_grid6.py` | Library: online randomised conformal p-values, betting functions, e-CUSUM detectors (CAM one-sided/two-sided, fixed power, mixture power, global plug-in, oracle and Gaussian CUSUM), calibration routines and all experiments. |
| `cam_simulations_section7.ipynb` | Reproducibility notebook, one section per experiment of Section 6 and of the Supplementary Material (notebook headings follow an earlier numbering; see the table below). |
| `requirements.txt` | Python dependencies (versions used for the paper). |

## Installation

```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
pip install -r requirements.txt
```

Python ≥ 3.10. No compiled extensions are required.

## Reproducing the paper

Open `cam_simulations_section7.ipynb` (locally with Jupyter, or on Google Colab) and run the cells in order. The first code cell detects the environment and writes all outputs to `cam_output/`.

All experiments use fixed seeds derived deterministically from the scenario names (`zlib.crc32`), so re-running the notebook with the package versions in `requirements.txt` reproduces the reported numbers exactly. The second code cell checks the seed signature `[298, 553, 219, 604]`.

| Paper | Notebook section | Function | Output files | Approx. runtime* |
|---|---|---|---|---|
| Figure 1 (Section 6.2) | 7.2 | `make_sim1_paper_figure` | `fig_sim1_fa.{png,pdf}` | tens of minutes |
| Figure 2 (Section 6.2) | 7.2 | `sim1_false_alarm_grid`, `make_sit1_figure` | `table_sit1_arl_nominal_vs_realized.csv`, `fig_sit1_arl.{png,pdf}` | < 1 hour |
| Table 1 (Section 6.3) | 7.3 | `sim2_delay_matched_arl` | `table_sim2_delay_matched_arl.csv`, `table_sim2_calibration.csv` | hours |
| Figure 3 (Section 6.4), Table S3 | 7.4 | `sim_level_shift_grid`, `make_figures` | `table_grid_level_shift.csv`, `table_grid_summary.csv`, `fig_regime.{png,pdf}` | several hours |
| Figure 4 (Section 6.4) | 7.4 | `make_figures` (changepoint sweep) | `table_nu_dilution.csv`, `fig_nu_dilution.{png,pdf}` | ~1 hour |
| Figure S1 | 7.1 | `make_pvalue_hist` | `fig_pvalue_hist.{png,pdf}` | minutes |
| Table S1 | 7.5 | `sim3_phase2_estimators` | `table_sim3_phase2_estimators.csv` | hours |
| Matched-ARL check (Section S5) | 7.5b | `sim3_matched_arl_check` | `table_sim3_matched_arl_check.csv` | hours |
| Table S2 | 7.6 | `sim4_one_sided_vs_two_sided` | `table_sim4_one_sided_vs_two_sided.csv`, `table_sim4_calibration.csv` | ~1 hour |

Every figure is saved both as PNG (for display) and as vector PDF (for the paper). If the CSV tables of a previous run are available in `cam_output/`, the last section of the notebook redraws all figures without re-running the long experiments.

\*Single core, standard Google Colab runtime. Each experiment writes its CSV/PNG outputs as soon as it completes.

## Using CAM on your own data

```python
import numpy as np
import cam_grid6 as m

rng = np.random.default_rng(0)
x_burn = ...          # in-control burn-in sample (1-D array)
x_mon  = ...          # monitoring stream (1-D array)

p = m.online_conformal_pvalues(x_burn, x_mon, rng)   # randomised online conformal p-values
det = m.CAMTwoSidedDetector(200.0)                   # two-sided CAM, design threshold gamma = 200
traj = m.compute_detector_trajectory(det, p)         # e-detector statistic over time
alarm = np.argmax(traj >= 200.0) + 1 if (traj >= 200.0).any() else None
```

Under the in-control hypothesis the detector satisfies E₀[τ] ≥ γ and P₀(τ ≤ m) ≤ 2m/γ for any continuous in-control distribution (validity theorem of Section 5.3 of the paper). The bound on the ARL is conservative; to target a specific ARL, calibrate the threshold once by Monte Carlo on i.i.d. Uniform[0,1] sequences with `calibrate_alarm_threshold` (Section 6.1).

## Citation

If you use this code, please cite the paper (reference to be updated upon publication) and the archived software release:

```
@software{laurenzi_cam_code,
  author = {Laurenzi, Andrea and Borrotti, Matteo and Demaria, Dario and Zippo, Valerio},
  title  = {Conformal Adaptive Martingales: simulation code},
  year   = {2026},
  doi    = {<ZENODO DOI>},
  url    = {https://github.com/<USER>/<REPO>}
}
```

## License

MIT (see `LICENSE`).
