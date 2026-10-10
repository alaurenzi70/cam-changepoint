"""
cam_mixture.py -- Section 6.4 experiments with MixturePower as a fourth method.

Extends, without modifying, the experiments of ``cam_grid6``:

* ``sim_level_shift_grid_mix``  -- the 12-cell level-shift grid (Figure 3, Table G.1)
  with CAM, Power(fixed), GlobalPlugin, MixturePower and the oracle CUSUM;
* ``nu_sweep_mix``              -- the changepoint sweep (Figure 4) with CAM,
  GlobalPlugin and MixturePower;
* ``make_regime_figure_mix`` and ``plot_nu_dilution_mix`` -- the two figures.

Seeds, horizons, calibration paths and target ARL are those of
``cam_grid6.sim_level_shift_grid`` and ``cam_grid6.make_figures`` with the settings
used in the notebook, so CAM, Power(fixed), GlobalPlugin and the oracle are reproduced
exactly and MixturePower is evaluated on the same paths (paired comparison).

MixturePower is the average of fixed power-betting e-CUSUMs over
eps in {0.1, 0.3, 0.5, 0.7, 0.9}: one-sided small-p on the right-skewed
distributions (``cam_grid6.MixturePowerCUSUM``) and, on the normal distribution,
the average over eps of the two-sided power e-CUSUMs (``MixturePowerBilateral``),
mirroring the configuration of CAM and Power(fixed) in each cell.
"""
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cam_grid6 as m

EPS_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
METHODS = ("CAM", "Power(fixed)", "GlobalPlugin", "MixturePower")
COLORS = {"CAM": "C0", "Power(fixed)": "C3", "GlobalPlugin": "C2", "MixturePower": "C1"}


class MixturePowerBilateral:
    """Two-sided MixturePower: for each eps, average of the small-p and large-p
    power e-CUSUMs; then average over the eps grid (a mixture of e-detectors)."""
    def __init__(self, eps_grid=EPS_GRID):
        self.eps = np.asarray(eps_grid, float)
        self.Ms = np.zeros(len(self.eps))
        self.Ml = np.zeros(len(self.eps))

    def step(self, p: float) -> float:
        self.Ms = self.eps * p ** (self.eps - 1.0) * np.maximum(self.Ms, 1.0)
        self.Ml = self.eps * (1.0 - p) ** (self.eps - 1.0) * np.maximum(self.Ml, 1.0)
        return float((0.5 * (self.Ms + self.Ml)).mean())


def _variants(gamma_design, phase2="beta"):
    return {
        ("two_sided", "CAM"): lambda: m.CAMTwoSidedDetector(gamma_design, phase2=phase2),
        ("two_sided", "Power(fixed)"): lambda: m.PowerCUSUMBilateral(0.5),
        ("two_sided", "GlobalPlugin"): lambda: m.GlobalPluginCUSUM(),
        ("two_sided", "MixturePower"): lambda: MixturePowerBilateral(EPS_GRID),
        ("one_sided_small_p", "CAM"): lambda: m.CAMSmallPCUSUM(gamma_design, phase2=phase2),
        ("one_sided_small_p", "Power(fixed)"): lambda: m.PowerCUSUM(0.5),
        ("one_sided_small_p", "GlobalPlugin"): lambda: m.GlobalPluginCUSUM(),
        ("one_sided_small_p", "MixturePower"): lambda: m.MixturePowerCUSUM(EPS_GRID),
    }


def _h0_pvalue_paths(base_seed, reps_cal, horizon_cal):
    seeds = m.make_seed_sequence(base_seed, reps_cal)
    p_h0, _ = m.build_paths(m.Scenario(lambda r, s: r.normal(0, 1, s),
                                       lambda r, s: r.normal(0, 1, s), nu=np.inf),
                            horizon_cal, seeds)
    return seeds, p_h0


def _grid_dist(args):
    """All shift magnitudes of one in-control distribution (one worker)."""
    (dist, deltas, nu, target_arl, thr, gamma_design, phase2,
     horizon_cal, reps_cal, horizon_del, reps_del, base_seed) = args
    spec = m._dist_spec(dist)
    direction = spec["direction"]
    variants = _variants(gamma_design, phase2)
    h0_seeds = m.make_seed_sequence(base_seed, reps_cal)
    raw_h0 = m.build_paths(m.Scenario(spec["sample"], spec["sample"], nu=np.inf),
                           horizon_cal, h0_seeds)[1]
    rows = []
    for delta in deltas:
        sc, _, _ = m._scenario_for(dist, delta, nu=nu)
        del_seeds = m.make_seed_sequence(
            base_seed + 991 * int(100 * delta) + zlib.crc32(dist.encode()) % 1000, reps_del)
        p_h1, raw_h1 = m.build_paths(sc, horizon_del, del_seeds)
        otr = m.compute_raw_trajectories(lambda llr=sc.llr: m.OracleCUSUM(llr), raw_h0)
        h_or = m.calibrate_alarm_threshold(otr, target_arl, horizon_cal)
        for name in METHODS:
            h = thr[(direction, name)]
            taus = m.evaluate_detector_on_p_paths(variants[(direction, name)], p_h1, h)
            s = m._cond_delay(taus, nu, horizon_del)
            rows.append(dict(distribution=dist, skew=round(spec["skew"], 2), direction=direction,
                             delta=delta, method=name, mean_delay=s["mean_delay"],
                             delay_ci95=s["delay_ci95"], trunc_delay=s["trunc_delay"],
                             P_miss=s["P_miss"], P_false_before=s["P_false_before"], threshold=h))
        taus = m.evaluate_detector_on_raw_paths(lambda llr=sc.llr: m.OracleCUSUM(llr), raw_h1, h_or)
        s = m._cond_delay(taus, nu, horizon_del)
        rows.append(dict(distribution=dist, skew=round(spec["skew"], 2), direction=direction,
                         delta=delta, method="Oracle", mean_delay=s["mean_delay"],
                         delay_ci95=s["delay_ci95"], trunc_delay=s["trunc_delay"],
                         P_miss=s["P_miss"], P_false_before=s["P_false_before"], threshold=h_or))
        print(f"  {dist:13s} delta={delta:<4} done", flush=True)
    return rows


def sim_level_shift_grid_mix(
    gamma_design=200,
    dists=("Normal", "Gamma(k=2)", "Exponential", "Gumbel(max)"),
    deltas=(0.5, 1.0, 2.0), nu=100,
    horizon_cal=2500, reps_cal=1500,
    horizon_del=2200, reps_del=4000,
    base_seed=20000, phase2="beta", n_jobs=1, save=True,
):
    """Level-shift grid of Section 6.4 with MixturePower added.

    Same design as cam_grid6.sim_level_shift_grid (defaults = notebook settings).
    n_jobs > 1 runs the in-control distributions in parallel (results identical).
    Returns dict(table, summary, rel, target_arl, thresholds, saved)."""
    # common target ARL: realised ARL of two-sided CAM at the design threshold
    _, p_h0 = _h0_pvalue_paths(base_seed, reps_cal, horizon_cal)
    variants = _variants(gamma_design, phase2)
    tr = m.compute_p_trajectories(variants[("two_sided", "CAM")], p_h0)
    target_arl, _ = m._arl_cens_from_trajs(tr, gamma_design, horizon_cal)
    thr = {("two_sided", "CAM"): gamma_design}
    for key, fac in variants.items():
        if key not in thr:
            thr[key] = m.calibrate_alarm_threshold(m.compute_p_trajectories(fac, p_h0),
                                                   target_arl, horizon_cal)

    args = [(d, deltas, nu, target_arl, thr, gamma_design, phase2,
             horizon_cal, reps_cal, horizon_del, reps_del, base_seed) for d in dists]
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_grid_dist, args)
    else:
        res = [_grid_dist(a) for a in args]
    df = pd.DataFrame([r for rr in res for r in rr])
    df.attrs["target_arl"] = target_arl

    piv = df.pivot_table(index=["distribution", "delta"], columns="method", values="trunc_delay")
    rel = piv.div(piv["Oracle"], axis=0)
    miss = df.pivot_table(index=["distribution", "delta"], columns="method", values="P_miss")
    summary = []
    for mth in METHODS:
        wins = int(sum(piv.loc[i, list(METHODS)].astype(float).idxmin() == mth for i in piv.index))
        summary.append(dict(method=mth,
                            rel_oracle_mean=float(np.nanmean(rel[mth])),
                            rel_oracle_ci95=float(m._ci95(rel[mth].values)),
                            **{f"rel_oracle_delta_{d}": float(rel[mth].xs(d, level="delta").mean())
                               for d in deltas},
                            P_miss_mean=float(np.nanmean(miss[mth])),
                            wins=wins, n_cells=len(piv.index)))
    summary = pd.DataFrame(summary).sort_values("rel_oracle_mean").reset_index(drop=True)

    saved = {}
    if save:
        saved["table"] = m.save_df(df, "table_grid_level_shift_mix.csv")
        saved["summary"] = m.save_df(summary, "table_grid_summary_mix.csv")
    thr_out = {"|".join(k): v for k, v in thr.items()}
    return dict(table=df, summary=summary, rel=rel, target_arl=target_arl,
                thresholds=thr_out, saved=saved)


def nu_sweep_mix(
    dist="Gamma(k=2)", delta=1.0, nus=(1, 25, 50, 100, 200, 400),
    gamma_design=200, horizon_cal=2500, reps_cal=1500,
    reps_sweep=1000, horizon_post=900, base_seed=40000, phase2="beta", save=True,
):
    """Changepoint sweep of Section 6.4 (Figure 4) with MixturePower added.

    Same design as the sweep in cam_grid6.make_figures (defaults = notebook settings):
    CAM at its design threshold, GlobalPlugin and MixturePower calibrated to the realised
    ARL of CAM. Returns dict(table, target_arl, thresholds, saved)."""
    one_sided = m._dist_spec(dist)["direction"] != "two_sided"
    if one_sided:
        cam_f = lambda: m.CAMSmallPCUSUM(gamma_design, phase2=phase2)
        mix_f = lambda: m.MixturePowerCUSUM(EPS_GRID)
    else:
        cam_f = lambda: m.CAMTwoSidedDetector(gamma_design, phase2=phase2)
        mix_f = lambda: MixturePowerBilateral(EPS_GRID)
    plug_f = lambda: m.GlobalPluginCUSUM()

    _, p_h0 = _h0_pvalue_paths(base_seed, reps_cal, horizon_cal)
    target, _ = m._arl_cens_from_trajs(m.compute_p_trajectories(cam_f, p_h0), gamma_design, horizon_cal)
    h_plug = m.calibrate_alarm_threshold(m.compute_p_trajectories(plug_f, p_h0), target, horizon_cal)
    h_mix = m.calibrate_alarm_threshold(m.compute_p_trajectories(mix_f, p_h0), target, horizon_cal)

    rows = []
    for nu in nus:
        sc, _, _ = m._scenario_for(dist, delta, nu=nu)
        H = nu + horizon_post
        pp, _ = m.build_paths(sc, H, m.make_seed_sequence(base_seed + 7 * nu, reps_sweep))
        row = dict(nu=nu)
        for name, fac, h in (("CAM", cam_f, gamma_design), ("GlobalPlugin", plug_f, h_plug),
                             ("MixturePower", mix_f, h_mix)):
            taus = np.array([m.alarm_time_from_signal(fac(), ps, h) for ps in pp], float)
            s = m._cond_delay(taus, nu, H)
            row[name] = s["mean_delay"]
            row[name + "_ci95"] = s["delay_ci95"]
        rows.append(row)
        print(f"  nu={nu:<4} done", flush=True)
    table = pd.DataFrame(rows)
    saved = {}
    if save:
        saved["table"] = m.save_df(table, "table_nu_dilution_mix.csv")
    return dict(table=table, target_arl=target,
                thresholds=dict(GlobalPlugin=h_plug, MixturePower=h_mix), saved=saved)


def make_regime_figure_mix(table, save_dir=None):
    """Figure 3: truncated delay relative to the per-cell oracle, averaged over the
    in-control distributions, by shift magnitude (four practical methods)."""
    save_dir = save_dir or m.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    df = table["table"] if isinstance(table, dict) else table
    piv = df.pivot_table(index=["distribution", "delta"], columns="method", values="trunc_delay")
    rel = piv.div(piv["Oracle"], axis=0).reset_index()
    deltas = sorted(df["delta"].unique())
    xticklab = [f"{'small' if d < 1 else 'medium' if d == 1 else 'large'} ({d})" for d in deltas]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    w = 0.8 / len(METHODS)
    x = np.arange(len(deltas))
    for i, mth in enumerate(METHODS):
        means = [rel.loc[rel["delta"] == d, mth].mean() for d in deltas]
        errs = [m._ci95(rel.loc[rel["delta"] == d, mth].values) for d in deltas]
        ax.bar(x + (i - (len(METHODS) - 1) / 2) * w, means, w, yerr=errs, label=mth,
               color=COLORS[mth], alpha=0.9, capsize=2)
    ax.axhline(1.0, color="k", ls="--", lw=1.1, label="oracle (=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklab, fontsize=9)
    ax.set_xlabel("Shift magnitude", fontsize=10)
    ax.set_ylabel("Delay relative to oracle", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
              columnspacing=1.0, handlelength=1.3, handletextpad=0.5)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return m._save_png_pdf(fig, save_dir, "fig_regime.png")


def plot_nu_dilution_mix(table, save_dir=None):
    """Figure 4: conditional detection delay against the changepoint location."""
    save_dir = save_dir or m.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    t = table["table"] if isinstance(table, dict) else table
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(t["nu"], t["CAM"], "o-", color=COLORS["CAM"], label="CAM (two-stage, localised)")
    ax.plot(t["nu"], t["MixturePower"], "D-", color=COLORS["MixturePower"],
            label="MixturePower (fixed bets, no estimation)")
    ax.plot(t["nu"], t["GlobalPlugin"], "s-", color=COLORS["GlobalPlugin"],
            label="GlobalPlugin (global estimate)")
    ax.set_xlabel(r"changepoint location $\nu$")
    ax.set_ylabel("Conditional detection delay")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return m._save_png_pdf(fig, save_dir, "fig_nu_dilution.png")


def grid_cells_latex(table):
    """Body of Table G.1: truncated delay and, in parentheses, delay relative to the oracle."""
    df = table["table"] if isinstance(table, dict) else table
    piv = df.pivot_table(index=["distribution", "delta"], columns="method", values="trunc_delay")
    rel = piv.div(piv["Oracle"], axis=0)
    lines = []
    for (dist, delta) in piv.index:
        cells = [f"{piv.loc[(dist, delta), 'Oracle']:.1f}"]
        cells += [f"{piv.loc[(dist, delta), mth]:.1f} ({rel.loc[(dist, delta), mth]:.2f})" for mth in METHODS]
        lines.append(f"{dist} & {delta:g} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)
