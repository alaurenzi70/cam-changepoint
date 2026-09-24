
"""
Refactored simulation pipeline for CAM paper.

Main experiments:
  sim1_false_alarm()
  sim2_delay_matched_arl()
  sim3_phase2_estimators()
  sim4_one_sided_vs_two_sided()

Compared with cam_simulations_v4.py, this version:
  - standardizes outputs into CSV + PNG/PDF
  - keeps paired Monte Carlo paths across methods
  - exposes a clean method factory
  - separates H0 calibration from H1 evaluation
  - adds a directional experiment for one-sided vs two-sided CAM
"""

import os
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zlib

# ---------------------------------------------------------------------
# 0. OUTPUT / GLOBAL CONFIG
# ---------------------------------------------------------------------

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_FIG_DPI = 160
_GAMMALN = np.frompyfunc(math.lgamma, 1, 1)


# ---------------------------------------------------------------------
# 1. LOW-LEVEL UTILITIES
# ---------------------------------------------------------------------

class _Fenwick:
    def __init__(self, m: int):
        self.m = m
        self.bit = np.zeros(m + 1, dtype=np.int64)

    def add(self, i: int, d: int = 1):
        while i <= self.m:
            self.bit[i] += d
            i += i & (-i)

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & (-i)
        return s


def make_seed_sequence(base_seed: int, reps: int) -> List[int]:
    return [base_seed + i for i in range(reps)]


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def save_df(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path


def save_figure(fig: plt.Figure, filename_base: str) -> Dict[str, str]:
    png = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    pdf = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    fig.savefig(png, dpi=DEFAULT_FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


# ---------------------------------------------------------------------
# 2. ONLINE RANDOMIZED CONFORMAL P-VALUES
# ---------------------------------------------------------------------


def online_conformal_pvalues(x_burn: np.ndarray, x_mon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomized online conformal p-values with score s(x)=x.

    p_t = ( # {old > new} + U * (# {old = new} + 1) ) / (n_old + 1)
    where the current point is included in the ranking via the +1 convention.
    """
    x_burn = np.asarray(x_burn, float)
    x_mon = np.asarray(x_mon, float)
    n, T = len(x_burn), len(x_mon)

    uniq = np.unique(np.concatenate([x_burn, x_mon]))
    cb = np.searchsorted(uniq, x_burn) + 1
    cm = np.searchsorted(uniq, x_mon) + 1

    bit = _Fenwick(len(uniq))
    for i in cb:
        bit.add(int(i))

    inserted = n
    ps = np.empty(T, dtype=float)
    u = rng.random(T)

    for t in range(T):
        c = int(cm[t])
        le = bit.prefix(c)
        eq = le - bit.prefix(c - 1)
        gt = inserted - le
        ps[t] = (gt + u[t] * (eq + 1.0)) / (inserted + 1.0)
        bit.add(c)
        inserted += 1

    return ps


# ---------------------------------------------------------------------
# 3. DENSITY ESTIMATORS ON [0,1] FOR PHASE II
# ---------------------------------------------------------------------


def beta_pdf(p: float, a: float, b: float, cap: float = 1e6) -> float:
    eps = 1e-12
    p = min(max(float(p), eps), 1.0 - eps)
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    logf = (a - 1.0) * math.log(p) + (b - 1.0) * math.log(1.0 - p) - logB
    return cap if logf > 13.8 else math.exp(logf)


def _beta_pdf_vec(x: np.ndarray, a, b) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12)
    logB = (_GAMMALN(a) + _GAMMALN(b) - _GAMMALN(a + b)).astype(float)
    logf = (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - logB
    return np.exp(np.clip(logf, -50, 13.8))


def chen_factor(p_eval: float, window: np.ndarray, bw: float, grid: np.ndarray) -> float:
    """Beta-kernel KDE of Chen type, normalized on a grid."""
    w = np.asarray(window, float)[None, :]
    u = np.asarray(grid, float)[:, None]
    a = u / bw + 1.0
    b = (1.0 - u) / bw + 1.0
    fhat = _beta_pdf_vec(w, a, b).mean(axis=1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    Z = trap(fhat, grid)
    fhat = fhat / max(float(Z), 1e-12)
    return float(np.interp(p_eval, grid, fhat))


def hist_factor(p: float, counts: np.ndarray, n_win: int, B: int, C: float) -> float:
    b = min(int(p * B), B - 1)
    return (C + counts[b]) / (C + n_win / B)


# ---------------------------------------------------------------------
# 4. DETECTORS
# ---------------------------------------------------------------------

class PowerCUSUM:
    """One-sided power-betting e-CUSUM (small-p side)."""
    def __init__(self, eps: float = 0.5):
        self.eps = eps
        self.M = 0.0

    def step(self, p: float) -> float:
        self.M = self.eps * (p ** (self.eps - 1.0)) * max(self.M, 1.0)
        return self.M


class GlobalPluginCUSUM:
    """Global plug-in e-CUSUM based on all past p-values (histogram)."""
    def __init__(self, B: int = 10, C: float = 10.0):
        self.B = B
        self.C = C
        self.counts = np.zeros(B, dtype=float)
        self.n = 0
        self.M = 0.0

    def step(self, p: float) -> float:
        f = hist_factor(p, self.counts, self.n, self.B, self.C)
        self.M = f * max(self.M, 1.0)
        self.counts[min(int(p * self.B), self.B - 1)] += 1.0
        self.n += 1
        return self.M


class MixturePowerCUSUM:
    """Average of e-CUSUMs across a grid of eps values."""
    def __init__(self, eps_grid: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)):
        self.eps = np.asarray(eps_grid, float)
        self.M = np.zeros(len(self.eps), dtype=float)

    def step(self, p: float) -> float:
        f = self.eps * (p ** (self.eps - 1.0))
        self.M = f * np.maximum(self.M, 1.0)
        return float(self.M.mean())


class CAMCUSUM:
    """
    One-sided CAM component (default: small-p side), two-stage.

    Phase I : fixed betting function (small-p by default)
    Phase II: plug-in estimate on the current candidate post-change window.
    """
    def __init__(
        self,
        gamma_design: float,
        eps: float = 0.5,
        gamma0: Optional[float] = None,
        phase2: str = "beta",
        lam: float = 0.05,
        B: int = 10,
        C: float = 2.0,
        ab_lo: float = 0.05,
        ab_hi: float = 50.0,
        min_win: int = 8,
        chen_bw: float = 0.05,
        chen_G: int = 80,
    ):
        self.gamma_design = gamma_design
        self.gamma0 = math.sqrt(gamma_design) if gamma0 is None else gamma0
        self.eps = eps
        self.phase2 = phase2
        self.lam = lam
        self.B = B
        self.C = C
        self.ab_lo = ab_lo
        self.ab_hi = ab_hi
        self.min_win = min_win
        self.chen_bw = chen_bw
        self.chen_grid = np.linspace(1e-3, 1 - 1e-3, chen_G)

        self.M = 0.0
        self.ps: List[float] = []
        self.run_start = 0
        self.counts = np.zeros(B, dtype=float)
        self.n_win = 0
        self.win_sum = 0.0
        self.win_sumsq = 0.0

    # ---- phase I ----
    def _finit(self, p: float) -> float:
        return self.eps * (p ** (self.eps - 1.0))

    # ---- phase II ----
    def _phase2(self, p: float) -> float:
        if self.phase2 == "hist":
            f = hist_factor(p, self.counts, self.n_win, self.B, self.C)
        elif self.phase2 == "chen":
            if self.n_win >= self.min_win:
                f = chen_factor(p, np.asarray(self.ps[self.run_start:], float), self.chen_bw, self.chen_grid)
            else:
                f = self._finit(p)   # era f = 1.0
        else:
            f = self._beta_mom(p)
        return (1.0 - self.lam) * f + self.lam * 1.0

    def _beta_mom(self, p: float) -> float:
        n = self.n_win
        if n < self.min_win:
            return self._finit(p)   # era return 1.0
        m = self.win_sum / n
        v = self.win_sumsq / n - m * m
        if v <= 1e-9 or v >= m * (1.0 - m) - 1e-9:
            return self._finit(p)   # era return 1.0
        phi = m * (1.0 - m) / v - 1.0
        a = min(max(m * phi, self.ab_lo), self.ab_hi)
        b = min(max((1.0 - m) * phi, self.ab_lo), self.ab_hi)
        return beta_pdf(p, a, b)

    def _bin(self, p: float) -> int:
        return min(int(p * self.B), self.B - 1)

    def _refresh(self):
        win = self.ps[self.run_start:]
        self.counts[:] = 0.0
        for q in win:
            self.counts[self._bin(q)] += 1.0
        self.n_win = len(win)
        arr = np.asarray(win, float)
        self.win_sum = float(arr.sum())
        self.win_sumsq = float((arr * arr).sum())

    def step(self, p: float) -> float:
        f = self._finit(p) if self.M < self.gamma0 else self._phase2(p)
        reset = self.M <= 1.0
        self.M = f * max(self.M, 1.0)
        self.ps.append(float(p))
        if reset:
            self.run_start = len(self.ps) - 1
            self._refresh()
        else:
            self.counts[self._bin(p)] += 1.0
            self.n_win += 1
            self.win_sum += p
            self.win_sumsq += p * p
        return self.M


class CAMLargePCUSUM(CAMCUSUM):
    """One-sided CAM component targeting large p-values."""
    def _finit(self, p: float) -> float:
        return self.eps * ((1.0 - p) ** (self.eps - 1.0))


class CAMStageIMixCUSUM(CAMCUSUM):
    """Ablation: mixing directly in Stage I (not the official detector)."""
    def _finit(self, p: float) -> float:
        return 0.5 * self.eps * (p ** (self.eps - 1.0)) + 0.5 * self.eps * ((1.0 - p) ** (self.eps - 1.0))


class CAMTwoSidedDetector:
    """Official CAM: average of small-p and large-p one-sided components."""
    def __init__(self, gamma_design: float, **kw):
        self.smallp = CAMCUSUM(gamma_design, **kw)
        self.largep = CAMLargePCUSUM(gamma_design, **kw)
        self.last_side = None

    def step(self, p: float) -> float:
        ms = self.smallp.step(p)
        ml = self.largep.step(p)
        self.last_side = "small-p" if ms >= ml else "large-p"
        return 0.5 * (ms + ml)


# Backward-compatible aliases
CAMUpCUSUM = CAMLargePCUSUM
CAMMixDetector = CAMTwoSidedDetector
CAMSmallPCUSUM = CAMCUSUM


class OracleCUSUM:
    def __init__(self, llr: Callable[[float], float]):
        self.llr = llr
        self.S = 0.0

    def step(self, x: float) -> float:
        self.S = max(0.0, self.S + self.llr(x))
        return math.exp(self.S)


# ---------------------------------------------------------------------
# 5. SCENARIOS AND PATH GENERATION
# ---------------------------------------------------------------------

@dataclass
class Scenario:
    f0_sample: Callable[[np.random.Generator, int], np.ndarray]
    f1_sample: Callable[[np.random.Generator, int], np.ndarray]
    llr: Optional[Callable[[float], float]] = None
    n_burn: int = 200
    nu: float = np.inf


def simulate_paths(sc: Scenario, horizon: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p-value path, raw monitoring path)."""
    rng = np.random.default_rng(seed)
    x_burn = sc.f0_sample(rng, sc.n_burn)
    n_pre = horizon if np.isinf(sc.nu) else min(int(sc.nu) - 1, horizon)
    x_mon = np.concatenate([
        sc.f0_sample(rng, n_pre),
        sc.f1_sample(rng, horizon - n_pre),
    ])
    ps = online_conformal_pvalues(x_burn, x_mon, rng)
    return ps, x_mon


def build_paths(sc: Scenario, horizon: int, seeds: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    p_paths, raw_paths = [], []
    for seed in seeds:
        ps, x = simulate_paths(sc, horizon, seed)
        p_paths.append(ps)
        raw_paths.append(x)
    return p_paths, raw_paths


# ---------------------------------------------------------------------
# 6. GENERIC EVALUATION UTILITIES
# ---------------------------------------------------------------------


def compute_detector_trajectory(detector, signal: np.ndarray) -> np.ndarray:
    tr = np.empty(len(signal), dtype=float)
    for t, z in enumerate(signal):
        tr[t] = detector.step(float(z))
    return tr


def alarm_time_from_signal(detector, signal: np.ndarray, threshold: float) -> float:
    for t, z in enumerate(signal):
        if detector.step(float(z)) >= threshold:
            return float(t + 1)
    return np.inf


def alarm_time_from_trajectory(traj: np.ndarray, threshold: float) -> float:
    hit = np.flatnonzero(np.asarray(traj) >= threshold)
    return float(hit[0] + 1) if len(hit) else np.inf


def empirical_alarm_cdf(taus: np.ndarray, m_grid: np.ndarray) -> np.ndarray:
    taus = np.asarray(taus, float)
    return np.array([np.mean(taus <= m) for m in m_grid], dtype=float)


def summarize_arl(taus: np.ndarray, horizon: int) -> Dict[str, float]:
    taus = np.asarray(taus, float)
    capped = np.where(np.isfinite(taus), taus, horizon)
    return {
        "arl_emp": float(capped.mean()),
        "arl_ci95": float(_ci95(capped)),
        "alarm_prob_by_horizon": float(np.mean(np.isfinite(taus))),
    }


def summarize_delays(taus: np.ndarray, nu: int = 1, horizon: Optional[int] = None) -> Dict[str, float]:
    taus = np.asarray(taus, float)
    finite = np.isfinite(taus)
    delays = taus[finite] - nu
    mean_delay = float(delays.mean()) if len(delays) else np.nan
    ci95 = float(_ci95(delays)) if len(delays) else np.nan
    cens = float(np.mean(~finite))
    if horizon is not None:
        trunc = np.where(finite, taus - nu, horizon - nu + 1).mean()
        trunc = float(trunc)
    else:
        trunc = np.nan
    return {
        "mean_delay": mean_delay,
        "delay_ci95": ci95,
        "censored_pct": 100.0 * cens,
        "truncated_mean_delay": trunc,
        "n_finite": int(finite.sum()),
    }


def evaluate_detector_on_p_paths(make_detector: Callable[[], object], p_paths: List[np.ndarray], threshold: float) -> np.ndarray:
    taus = []
    for ps in p_paths:
        taus.append(alarm_time_from_signal(make_detector(), ps, threshold))
    return np.asarray(taus, float)


def evaluate_detector_on_raw_paths(make_detector: Callable[[], object], raw_paths: List[np.ndarray], threshold: float) -> np.ndarray:
    taus = []
    for x in raw_paths:
        taus.append(alarm_time_from_signal(make_detector(), x, threshold))
    return np.asarray(taus, float)


def compute_p_trajectories(make_detector: Callable[[], object], p_paths: List[np.ndarray]) -> np.ndarray:
    return np.vstack([compute_detector_trajectory(make_detector(), ps) for ps in p_paths])


def compute_raw_trajectories(make_detector: Callable[[], object], raw_paths: List[np.ndarray]) -> np.ndarray:
    return np.vstack([compute_detector_trajectory(make_detector(), x) for x in raw_paths])


def _arl_cens_from_trajs(trajs: np.ndarray, h: float, horizon: int) -> Tuple[float, float]:
    hit = trajs >= h
    any_hit = hit.any(axis=1)
    tau = np.where(any_hit, np.argmax(hit, axis=1) + 1, horizon)
    return float(tau.mean()), float(np.mean(~any_hit))


def calibrate_alarm_threshold(trajs: np.ndarray, target_arl: float, horizon: int) -> float:
    """Binary search for a threshold delivering ARL close to target_arl."""
    lo, hi = 1.0, max(target_arl, 2.0)
    while _arl_cens_from_trajs(trajs, hi, horizon)[0] < target_arl and hi < 1e9:
        hi *= 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        arl_mid, _ = _arl_cens_from_trajs(trajs, mid, horizon)
        if arl_mid < target_arl:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# 7. METHOD FACTORIES
# ---------------------------------------------------------------------


def default_pvalue_methods(gamma_design: float, phase2: str = "beta") -> Dict[str, Callable[[], object]]:
    return {
        "CAM": lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2),
        "PowerCUSUM": lambda: PowerCUSUM(0.5),
        "GlobalPlugin": lambda: GlobalPluginCUSUM(B=10, C=10.0),
        "MixturePower": lambda: MixturePowerCUSUM(),
    }


def directional_cam_methods(gamma_design: float, phase2: str = "beta") -> Dict[str, Callable[[], object]]:
    return {
        "CAM-smallP": lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2),
        "CAM-largeP": lambda: CAMLargePCUSUM(gamma_design, phase2=phase2),
        "CAM-twoSided": lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2),
    }


# ---------------------------------------------------------------------
# 8. SIM 1 -- FALSE ALARM UNDER H0
# ---------------------------------------------------------------------


def sim1_false_alarm(
    gamma: float = 200,
    horizon: int = 1000,
    reps: int = 4000,
    base_seed: int = 1000,
    phase2: str = "beta",
    save: bool = True,
):
    methods = default_pvalue_methods(gamma, phase2=phase2)
    f0s = {
        "Gaussian": lambda r, s: r.normal(0.0, 1.0, s),
        "Lognormal": lambda r, s: r.lognormal(0.0, 1.0, s),
    }
    m_grid = np.arange(1, min(3 * int(gamma), horizon) + 1)

    table_rows = []
    curve_rows = []

    fig, axes = plt.subplots(1, len(f0s), figsize=(6 * len(f0s), 4), sharey=True)
    if len(f0s) == 1:
        axes = [axes]

    seeds = make_seed_sequence(base_seed, reps)

    for ax, (dist_name, f0_sample) in zip(axes, f0s.items()):
        sc = Scenario(f0_sample=f0_sample, f1_sample=f0_sample, nu=np.inf)
        p_paths, _ = build_paths(sc, horizon, seeds)

        for method_name, make_detector in methods.items():
            taus = evaluate_detector_on_p_paths(make_detector, p_paths, threshold=gamma)
            summ = summarize_arl(taus, horizon=horizon)
            table_rows.append({
                "distribution": dist_name,
                "method": method_name,
                "gamma": gamma,
                "horizon": horizon,
                "reps": reps,
                **summ,
            })

            cdf = empirical_alarm_cdf(taus, m_grid)
            for m, val in zip(m_grid, cdf):
                curve_rows.append({
                    "distribution": dist_name,
                    "method": method_name,
                    "m": int(m),
                    "fa_prob": float(val),
                })
            ax.plot(m_grid, cdf, label=method_name)

        # theoretical reference lines
        ax.plot(m_grid, np.minimum(m_grid / gamma, 1.0), "k--", lw=1.2, label="m/gamma")
        ax.plot(m_grid, np.minimum(2.0 * m_grid / gamma, 1.0), color="gray", ls=":", lw=1.2, label="2m/gamma")
        ax.set_title(dist_name)
        ax.set_xlabel("Monitoring horizon m")
        ax.set_ylabel("Empirical P(tau <= m)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    #fig.suptitle("Sim 1: False alarm probability under H0", y=1.03)
    #fig.tight_layout()

    table_df = pd.DataFrame(table_rows)
    curve_df = pd.DataFrame(curve_rows)

    saved = {}
    if save:
        saved["table"] = save_df(table_df, "table_sim1_false_alarm.csv")
        saved["curves"] = save_df(curve_df, "curves_sim1_false_alarm.csv")
        saved.update({f"figure_{k}": v for k, v in save_figure(fig, "fig_sim1_false_alarm").items()})
    else:
        plt.close(fig)

    return {
        "table": table_df,
        "curves": curve_df,
        "saved": saved,
    }


def make_sim1_paper_figure(gamma=200, horizon=600, reps=4000, base_seed=1000,
                           phase2="beta", save_name="fig_sim1_fa"):
    """Figure 2 of the paper: empirical P_0(tau <= m) under the in-control
    regime (standard normal F_0) for two-sided CAM and one-sided fixed power
    betting, with the finite-horizon bounds 2m/gamma and m/gamma.
    By Lemma 2.9 the in-control run-length law does not depend on F_0."""
    from matplotlib.lines import Line2D

    methods = {
        "CAM": (lambda: CAMTwoSidedDetector(gamma, phase2=phase2), "#1f77b4"),
        "PowerCUSUM": (lambda: PowerCUSUM(0.5), "#d62728"),
    }
    m_grid = np.arange(1, min(3 * int(gamma), horizon) + 1)
    seeds = make_seed_sequence(base_seed, reps)

    sc = Scenario(f0_sample=lambda r, s: r.normal(0.0, 1.0, s),
                  f1_sample=lambda r, s: r.normal(0.0, 1.0, s), nu=np.inf)
    p_paths, _ = build_paths(sc, horizon, seeds)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for name, (make_det, col) in methods.items():
        taus = evaluate_detector_on_p_paths(make_det, p_paths, threshold=gamma)
        ax.plot(m_grid, empirical_alarm_cdf(taus, m_grid), color=col, ls="-", lw=1.4)

    # theoretical bounds, same colour as the method they refer to
    ax.plot(m_grid, np.minimum(2.0 * m_grid / gamma, 1.0), color="#1f77b4", ls=":", lw=1.2)
    ax.plot(m_grid, np.minimum(m_grid / gamma, 1.0), color="#d62728", ls=":", lw=1.2)

    handles = [
        Line2D([], [], color="#1f77b4", lw=1.6, label=r"CAM (two-sided), bound $2m/\gamma$"),
        Line2D([], [], color="#d62728", lw=1.6, label=r"Power (one-sided), bound $m/\gamma$"),
        Line2D([], [], color="k", ls=":", lw=1.2, label="theoretical bound"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel(r"Monitoring horizon $m$", fontsize=10)
    ax.set_ylabel(r"Empirical $\mathbb{P}_0(\tau \leq m)$", fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, save_name)


# ---------------------------------------------------------------------
# 9. SIM 2 -- DETECTION DELAY AT MATCHED ARL
# ---------------------------------------------------------------------


def gaussian_mean_shift_scenario(delta: float, nu: int = 1, n_burn: int = 200) -> Scenario:
    return Scenario(
        f0_sample=lambda r, s: r.normal(0.0, 1.0, s),
        f1_sample=lambda r, s, d=delta: r.normal(d, 1.0, s),
        llr=lambda x, d=delta: d * x - 0.5 * d * d,
        n_burn=n_burn,
        nu=nu,
    )


def sim2_delay_matched_arl(
    gamma_design: float = 200,
    horizon_cal: int = 2500,
    reps_cal: int = 1500,
    horizon_del: int = 800,
    reps_del: int = 1000,
    delta_grid: Tuple[float, ...] = (0.5, 1.0, 1.5),
    base_seed: int = 5000,
    phase2: str = "beta",
    save: bool = True,
):
    # --- Step A. Realized ARL of official CAM under H0 ---
    sc_h0 = gaussian_mean_shift_scenario(delta=0.0, nu=np.inf)
    cal_seeds = make_seed_sequence(base_seed, reps_cal)
    p_paths_h0, raw_paths_h0 = build_paths(sc_h0, horizon_cal, cal_seeds)

    cam_factory = lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2)
    cam_trajs_h0 = compute_p_trajectories(cam_factory, p_paths_h0)
    cam_taus_h0 = np.array([alarm_time_from_trajectory(tr, gamma_design) for tr in cam_trajs_h0], float)
    target_arl = float(np.where(np.isfinite(cam_taus_h0), cam_taus_h0, horizon_cal).mean())

    # --- Step B. Calibrate competing methods to the same realized ARL ---
    p_methods = default_pvalue_methods(gamma_design, phase2=phase2)
    p_methods["CAM-oneSided"] = lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2)
    # CAM is the reference method: fixed external threshold gamma_design
    method_thresholds = {"CAM": gamma_design}
    method_arl_realized = {"CAM": target_arl}

    for name, make_detector in p_methods.items():
        if name == "CAM":
            continue
        trajs = compute_p_trajectories(make_detector, p_paths_h0)
        h = calibrate_alarm_threshold(trajs, target_arl=target_arl, horizon=horizon_cal)
        arl_h, _ = _arl_cens_from_trajs(trajs, h, horizon_cal)
        method_thresholds[name] = float(h)
        method_arl_realized[name] = float(arl_h)

    # --- Step C. Evaluate delays under H1 ---
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # evaluate every calibrated p-value method, plus the oracle;
    # deriving the list from p_methods keeps Steps B and C in sync.
    eval_order = list(p_methods.keys()) + ["OracleCUSUM"]

    for method_name in eval_order:
        ax.plot([], [], label=method_name)  # reserve legend order
    ax.cla()

    for method_name in eval_order:
        delays_curve = []
        for delta in delta_grid:
            sc_h1 = gaussian_mean_shift_scenario(delta=delta, nu=1)
            del_seeds = make_seed_sequence(base_seed + int(1000 * delta), reps_del)
            p_paths_h1, raw_paths_h1 = build_paths(sc_h1, horizon_del, del_seeds)

            I = delta * delta / 2.0
            benchmark = math.log(gamma_design) / I if I > 0 else np.nan

            if method_name == "OracleCUSUM":
                # threshold is calibrated separately for each delta because the oracle statistic depends on llr scale.
                raw_paths_h0_local = raw_paths_h0
                oracle_factory_h0 = lambda llr=sc_h1.llr: OracleCUSUM(llr)
                oracle_trajs_h0 = compute_raw_trajectories(oracle_factory_h0, raw_paths_h0_local)
                h = calibrate_alarm_threshold(oracle_trajs_h0, target_arl=target_arl, horizon=horizon_cal)
                arl_realized, _ = _arl_cens_from_trajs(oracle_trajs_h0, h, horizon_cal)
                taus = evaluate_detector_on_raw_paths(lambda llr=sc_h1.llr: OracleCUSUM(llr), raw_paths_h1, h)
                used_threshold = h
                used_arl = arl_realized
            else:
                make_detector = p_methods[method_name]
                h = method_thresholds[method_name]
                taus = evaluate_detector_on_p_paths(make_detector, p_paths_h1, h)
                used_threshold = h
                used_arl = method_arl_realized[method_name]

            summ = summarize_delays(taus, nu=1, horizon=horizon_del)
            rows.append({
                "method": method_name,
                "delta": delta,
                "gamma_design": gamma_design,
                "target_arl_from_cam": target_arl,
                "used_threshold": used_threshold,
                "realized_arl_h0": used_arl,
                "oracle_benchmark": benchmark,
                **summ,
            })
            delays_curve.append(summ["truncated_mean_delay"])
        ax.plot(delta_grid, delays_curve, marker="o", label=method_name)

    # oracle benchmark curve
    benchmark_curve = [math.log(gamma_design) / (d * d / 2.0) for d in delta_grid]
    ax.plot(delta_grid, benchmark_curve, "k--", lw=1.4, label="log(gamma)/I")
    ax.set_xlabel("Shift size delta")
    ax.set_ylabel("Truncated mean detection delay")
    ax.set_title("Sim 2: Detection delay at matched ARL")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    df = pd.DataFrame(rows)

    saved = {}
    if save:
        saved["table"] = save_df(df, "table_sim2_delay_matched_arl.csv")
        saved.update({f"figure_{k}": v for k, v in save_figure(fig, "fig_sim2_delay_matched_arl").items()})
    else:
        plt.close(fig)

    calib_df = pd.DataFrame([
        {"method": k, "threshold": method_thresholds[k], "realized_arl_h0": method_arl_realized[k]}
        for k in method_thresholds
    ])
    if save:
        saved["calibration"] = save_df(calib_df, "table_sim2_calibration.csv")

    return {
        "table": df,
        "calibration": calib_df,
        "saved": saved,
    }


# ---------------------------------------------------------------------
# 10. SIM 3 -- EFFECT OF THE PHASE II ESTIMATOR
# ---------------------------------------------------------------------


def sim3_phase2_estimators(
    gammas: Tuple[int, ...] = (100, 200, 500),
    delta: float = 0.5,
    horizon: int = 2500,
    reps: int = 400,
    base_seed: int = 8000,
    include_chen: bool = True,
    save: bool = True,
):
    estimators = ["hist", "beta"] + (["chen"] if include_chen else [])
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))

    label = {"hist": "Histogram", "beta": "Beta-MoM", "chen": "Chen-KDE"}

    for est in estimators:
        y = []
        for g in gammas:
            sc = gaussian_mean_shift_scenario(delta=delta, nu=1)
            seeds = make_seed_sequence(base_seed + 17 * g, reps)
            p_paths, _ = build_paths(sc, horizon, seeds)
            taus = evaluate_detector_on_p_paths(
                lambda est=est, g=g: CAMTwoSidedDetector(g, phase2=est),
                p_paths,
                threshold=g,
            )
            I = delta * delta / 2.0
            benchmark = math.log(g) / I
            summ = summarize_delays(taus, nu=1, horizon=horizon)
            rows.append({
                "gamma": g,
                "delta": delta,
                "estimator": est,
                "oracle_benchmark": benchmark,
                **summ,
            })
            y.append(summ["truncated_mean_delay"])
        ax.plot(gammas, y, marker="o", label=label[est])

    bench = [math.log(g) / (delta * delta / 2.0) for g in gammas]
    ax.plot(gammas, bench, "k--", lw=1.4, label="log(gamma)/I")
    ax.set_xlabel("Alarm threshold gamma")
    ax.set_ylabel("Truncated mean detection delay")
    ax.set_title(f"Sim 3: Phase II estimator comparison (delta={delta})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    df = pd.DataFrame(rows)

    saved = {}
    if save:
        saved["table"] = save_df(df, "table_sim3_phase2_estimators.csv")
        saved.update({f"figure_{k}": v for k, v in save_figure(fig, "fig_sim3_phase2_estimators").items()})
    else:
        plt.close(fig)

    return {
        "table": df,
        "saved": saved,
    }


# ---------------------------------------------------------------------
# 10b. SIM 3 CHECK -- PHASE II ESTIMATORS AT MATCHED ARL
# ---------------------------------------------------------------------


def sim3_matched_arl_check(
    gamma_design: float = 200,
    delta: float = 0.5,
    horizon_cal: int = 2500,
    reps_cal: int = 1500,
    horizon_del: int = 2000,
    reps_del: int = 4000,
    base_seed: int = 5000,
    phase2_ref: str = "beta",
    phase2_alt: str = "hist",
    save: bool = True,
):
    """Verifica citata nella Sez. 7.5: il vantaggio a soglia nominale dello
    stimatore alternativo (hist) sopravvive a parita' di ARL realizzato?

    Ricostruisce i path H0/H1 di sim2 (stessi seed, quindi confronto appaiato
    con la tabella della Sez. 7.3), misura l'ARL realizzato del CAM di
    riferimento (phase2_ref) a gamma_design, calibra le varianti alternative
    (two-sided e one-sided) allo stesso ARL e le valuta sulla cella delta.
    """
    # --- path H0 identici a sim2 ---
    sc_h0 = gaussian_mean_shift_scenario(delta=0.0, nu=np.inf)
    cal_seeds = make_seed_sequence(base_seed, reps_cal)
    p_h0, _ = build_paths(sc_h0, horizon_cal, cal_seeds)

    # --- target: ARL realizzato del riferimento a soglia di design ---
    ref2 = lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2_ref)
    trajs_ref2 = compute_p_trajectories(ref2, p_h0)
    taus_ref2 = np.array([alarm_time_from_trajectory(tr, gamma_design) for tr in trajs_ref2], float)
    target_arl = float(np.where(np.isfinite(taus_ref2), taus_ref2, horizon_cal).mean())

    detectors = {
        f"CAM-{phase2_ref} (two-sided)": (ref2, gamma_design),
        f"CAM-{phase2_ref} (one-sided)": (lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2_ref), None),
        f"CAM-{phase2_alt} (two-sided)": (lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2_alt), None),
        f"CAM-{phase2_alt} (one-sided)": (lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2_alt), None),
    }

    # --- calibrazione al target (il riferimento two-sided resta a gamma_design) ---
    calibrated = {}
    for name, (factory, fixed_h) in detectors.items():
        if fixed_h is not None:
            calibrated[name] = (factory, float(fixed_h), target_arl)
            continue
        trajs = compute_p_trajectories(factory, p_h0)
        h = calibrate_alarm_threshold(trajs, target_arl=target_arl, horizon=horizon_cal)
        arl, _ = _arl_cens_from_trajs(trajs, h, horizon_cal)
        calibrated[name] = (factory, float(h), float(arl))

    # --- valutazione sulla cella delta, path H1 identici a sim2 ---
    sc_h1 = gaussian_mean_shift_scenario(delta=delta, nu=1)
    h1_seeds = make_seed_sequence(base_seed + int(1000 * delta), reps_del)
    p_h1, _ = build_paths(sc_h1, horizon_del, h1_seeds)

    rows = []
    for name, (factory, h, arl) in calibrated.items():
        taus = evaluate_detector_on_p_paths(factory, p_h1, threshold=h)
        summ = summarize_delays(taus, nu=1, horizon=horizon_del)
        rows.append({
            "method": name,
            "delta": delta,
            "used_threshold": h,
            "realized_arl_h0": arl,
            "target_arl": target_arl,
            **summ,
        })

    df = pd.DataFrame(rows)
    saved = {}
    if save:
        saved["table"] = save_df(df, "table_sim3_matched_arl_check.csv")
    return {"table": df, "target_arl": target_arl, "saved": saved}


# ---------------------------------------------------------------------
# 11. SIM 4 -- ONE-SIDED VS TWO-SIDED CAM
# ---------------------------------------------------------------------


def sim4_one_sided_vs_two_sided(
    gamma_design: float = 200,
    horizon_cal: int = 2500,
    reps_cal: int = 1500,
    horizon_del: int = 1000,
    reps_del: int = 800,
    delta: float = 1.0,
    base_seed: int = 12000,
    phase2: str = "beta",
    save: bool = True,
):
    methods = directional_cam_methods(gamma_design, phase2=phase2)

    # Target ARL = realized ARL of official two-sided CAM under H0
    sc_h0 = gaussian_mean_shift_scenario(delta=0.0, nu=np.inf)
    cal_seeds = make_seed_sequence(base_seed, reps_cal)
    p_paths_h0, _ = build_paths(sc_h0, horizon_cal, cal_seeds)

    twosided_trajs_h0 = compute_p_trajectories(methods["CAM-twoSided"], p_paths_h0)
    twosided_taus_h0 = np.array([alarm_time_from_trajectory(tr, gamma_design) for tr in twosided_trajs_h0], float)
    target_arl = float(np.where(np.isfinite(twosided_taus_h0), twosided_taus_h0, horizon_cal).mean())

    thresholds = {"CAM-twoSided": gamma_design}
    arls = {"CAM-twoSided": target_arl}
    for name in ["CAM-smallP", "CAM-largeP"]:
        trajs = compute_p_trajectories(methods[name], p_paths_h0)
        h = calibrate_alarm_threshold(trajs, target_arl=target_arl, horizon=horizon_cal)
        arl_h, _ = _arl_cens_from_trajs(trajs, h, horizon_cal)
        thresholds[name] = float(h)
        arls[name] = float(arl_h)

    directional_scenarios = {
        "positive_shift_smallP": gaussian_mean_shift_scenario(delta=+delta, nu=1),
        "negative_shift_largeP": gaussian_mean_shift_scenario(delta=-delta, nu=1),
    }

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for j, (ax, (sc_name, sc)) in enumerate(zip(axes, directional_scenarios.items())):
        seeds = make_seed_sequence(base_seed + 1000 * (j + 1), reps_del)
        p_paths, _ = build_paths(sc, horizon_del, seeds)

        bars = []
        labels = []
        errs = []
        for method_name in ["CAM-smallP", "CAM-largeP", "CAM-twoSided"]:
            taus = evaluate_detector_on_p_paths(methods[method_name], p_paths, thresholds[method_name])
            summ = summarize_delays(taus, nu=1, horizon=horizon_del)
            rows.append({
                "scenario": sc_name,
                "delta": delta if "positive" in sc_name else -delta,
                "method": method_name,
                "used_threshold": thresholds[method_name],
                "realized_arl_h0": arls[method_name],
                **summ,
            })
            labels.append(method_name)
            bars.append(summ["truncated_mean_delay"])
            errs.append(summ["delay_ci95"] if np.isfinite(summ["delay_ci95"]) else 0.0)

        x = np.arange(len(labels))
        ax.bar(x, bars, yerr=errs, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(sc_name)
        ax.set_ylabel("Truncated mean detection delay")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Sim 4: One-sided vs two-sided CAM", y=1.02)
    fig.tight_layout()

    df = pd.DataFrame(rows)

    saved = {}
    if save:
        saved["table"] = save_df(df, "table_sim4_one_sided_vs_two_sided.csv")
        saved.update({f"figure_{k}": v for k, v in save_figure(fig, "fig_sim4_one_sided_vs_two_sided").items()})
    else:
        plt.close(fig)

    calib_df = pd.DataFrame([
        {"method": k, "threshold": thresholds[k], "realized_arl_h0": arls[k]}
        for k in thresholds
    ])
    if save:
        saved["calibration"] = save_df(calib_df, "table_sim4_calibration.csv")

    return {
        "table": df,
        "calibration": calib_df,
        "saved": saved,
    }


# ---------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------

# Test rapido

def run_all():
    out = {}
    out["sim1"] = sim1_false_alarm(gamma=200, horizon=1000, reps=300, base_seed=1000, phase2="beta", save=True)
    out["sim2"] = sim2_delay_matched_arl(
        gamma_design=200,
        horizon_cal=2500,
        reps_cal=300, #1000,
        horizon_del=800,
        reps_del=300, #800,
        delta_grid=(0.5, 1.0, 1.5),
        base_seed=5000,
        phase2="beta",
        save=True,
    )
    out["sim3"] = sim3_phase2_estimators(
        gammas=(100, 200, 500),
        delta=0.5,
        horizon=2500,
        reps=350,
        base_seed=8000,
        include_chen=True,
        save=True,
    )
    out["sim4"] = sim4_one_sided_vs_two_sided(
        gamma_design=200,
        horizon_cal=2500,
        reps_cal=300, #1000,
        horizon_del=1000,
        reps_del=300,#700,
        delta=1.0,
        base_seed=12000,
        phase2="beta",
        save=True,
    )
    return out




# =====================================================================
# 13. GRIGLIA INDUSTRIALE -- shift di livello su statistiche di processo
# =====================================================================
# Una statistica = una serie = un CAM. Unica anomalia: shift di livello.
# Distribuzioni in-control scelte per la forma nativa della statistica:
#   Normal      -> media subgroup        (simmetrica, bilaterale)
#   Gamma(k=2)  -> dev. standard within  (asimmetrica moderata, unilaterale su)
#   Exponential -> conteggi/particle     (asimmetria forte, unilaterale su)
#   Gumbel      -> max del lotto         (coda pesante/estremo, unilaterale su)
# Shift in unita' di deviazione standard in-control, cosi' e' comparabile.


class GaussianCUSUM:
    """Carta SPC parametrica: CUSUM gaussiano bilaterale su dati standardizzati
    z=(x-mu0)/sigma0. Tarato (soglia) sulla normale per ARL=gamma e applicato
    a tutte le distribuzioni: su dati asimmetrici la coda destra lo fa
    sovra-allarmare (ARL realizzato che crolla)."""
    def __init__(self, mu0, sigma0, k=0.5):
        self.mu0 = mu0; self.sigma0 = sigma0; self.k = k
        self.Sp = 0.0; self.Sm = 0.0
    def step(self, x):
        z = (x - self.mu0) / self.sigma0
        self.Sp = max(0.0, self.Sp + z - self.k)
        self.Sm = max(0.0, self.Sm - z - self.k)
        return max(self.Sp, self.Sm)


class PowerCUSUMBilateral:
    """Power-betting fisso bilaterale (= CAM senza Phase II), media e-detector."""
    def __init__(self, eps=0.5):
        self.eps = eps; self.Ms = 0.0; self.Ml = 0.0
    def step(self, p):
        self.Ms = self.eps * p ** (self.eps - 1.0) * max(self.Ms, 1.0)
        self.Ml = self.eps * (1.0 - p) ** (self.eps - 1.0) * max(self.Ml, 1.0)
        return 0.5 * (self.Ms + self.Ml)


def _dist_spec(dist):
    if dist == "Normal":
        return dict(sample=lambda r, s: r.normal(0.0, 1.0, s), mean=0.0, sd=1.0, skew=0.0,
                    direction="two_sided",
                    llr=lambda D: (lambda x: D * x - 0.5 * D * D))
    if dist == "Gamma(k=2)":
        k, th = 2.0, 1.0; sd = math.sqrt(k) * th
        def llr(D):
            def f(x):
                if x <= D + 1e-9:
                    return -50.0
                return max(-50.0, min(50.0, (k - 1) * math.log((x - D) / x) + D / th))
            return f
        return dict(sample=lambda r, s: r.gamma(k, th, s), mean=k * th, sd=sd,
                    skew=2.0 / math.sqrt(k), direction="one_sided_small_p", llr=llr)
    if dist == "Exponential":
        def llr(D):
            return lambda x: (D if x > D + 1e-9 else -50.0)
        return dict(sample=lambda r, s: r.exponential(1.0, s), mean=1.0, sd=1.0, skew=2.0,
                    direction="one_sided_small_p", llr=llr)
    if dist == "Gumbel(max)":
        sd = math.pi / math.sqrt(6.0)
        def llr(D):
            return lambda x: max(-50.0, min(50.0, D + math.exp(-min(x, 50.0)) * (1.0 - math.exp(D))))
        return dict(sample=lambda r, s: r.gumbel(0.0, 1.0, s),
                    mean=0.5772156649, sd=sd, skew=1.1395,
                    direction="one_sided_small_p", llr=llr)
    raise ValueError(dist)


def _scenario_for(dist, delta, nu=1):
    spec = _dist_spec(dist)
    D = delta * spec["sd"]
    f0 = spec["sample"]
    f1 = lambda r, s, base=spec["sample"], DD=D: base(r, s) + DD
    return Scenario(f0_sample=f0, f1_sample=f1, llr=spec["llr"](D), nu=nu), spec, D


def _cond_delay(taus, nu, horizon):
    """Ritardo condizionato a tau>=nu (post-change). Riporta anche falsi-prima e miss."""
    taus = np.asarray(taus, float)
    finite = np.isfinite(taus)
    false_before = float(np.mean(finite & (taus < nu)))
    detected = finite & (taus >= nu)
    delays = taus[detected] - nu
    mean = float(delays.mean()) if len(delays) else np.nan
    ci = float(_ci95(delays)) if len(delays) else np.nan
    miss = float(np.mean(~finite))
    # troncato robusto alla censura: post-change, miss conta horizon-nu
    post = finite & (taus >= nu)
    trunc_vals = np.where(post, np.minimum(taus, horizon) - nu, horizon - nu)
    # escludi i falsi-prima dal troncato (non sono ritardi di rilevazione)
    keep = ~(finite & (taus < nu))
    trunc = float(trunc_vals[keep].mean()) if keep.any() else np.nan
    return dict(mean_delay=mean, delay_ci95=ci, trunc_delay=trunc,
                P_false_before=false_before, P_miss=miss)


def _conformal_methods(direction, gamma_design, phase2="beta"):
    if direction == "two_sided":
        return {"CAM": lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2),
                "Power(fixed)": lambda: PowerCUSUMBilateral(0.5),
                "GlobalPlugin": lambda: GlobalPluginCUSUM()}
    return {"CAM": lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2),
            "Power(fixed)": lambda: PowerCUSUM(0.5),
            "GlobalPlugin": lambda: GlobalPluginCUSUM()}


def sim_level_shift_grid(
    gamma_design=200,
    dists=("Normal", "Gamma(k=2)", "Exponential", "Gumbel(max)"),
    deltas=(0.5, 1.0, 2.0),
    horizon_cal=1800, reps_cal=600,
    horizon_del=800, reps_del=600, nu=100,
    base_seed=20000, phase2="beta", save=True, verbose=True,
):
    # --- calibrazione conformale: ARL comune = ARL realizzato dal CAM bilaterale a design ---
    h0_seeds = make_seed_sequence(base_seed, reps_cal)
    p_h0, _ = build_paths(Scenario(lambda r, s: r.normal(0, 1, s),
                                   lambda r, s: r.normal(0, 1, s), nu=np.inf),
                          horizon_cal, h0_seeds)
    cam2 = lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2)
    tr = compute_p_trajectories(cam2, p_h0)
    target_arl, _ = _arl_cens_from_trajs(tr, gamma_design, horizon_cal)

    # soglie conformali (distribution-free), per direzionalita'
    conf_thr = {("two_sided", "CAM"): gamma_design}
    variants = {
        ("two_sided", "CAM"): cam2,
        ("two_sided", "Power(fixed)"): lambda: PowerCUSUMBilateral(0.5),
        ("two_sided", "GlobalPlugin"): lambda: GlobalPluginCUSUM(),
        ("one_sided_small_p", "CAM"): lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2),
        ("one_sided_small_p", "Power(fixed)"): lambda: PowerCUSUM(0.5),
        ("one_sided_small_p", "GlobalPlugin"): lambda: GlobalPluginCUSUM(),
    }
    for key, fac in variants.items():
        if key in conf_thr:
            continue
        tj = compute_p_trajectories(fac, p_h0)
        conf_thr[key] = calibrate_alarm_threshold(tj, target_arl, horizon_cal)

    rows = []
    for dist in dists:
        spec = _dist_spec(dist); direction = spec["direction"]
        # H0 raw per la calibrazione dell'oracle (dipende da F0 e dalla llr=dist+delta)
        raw_h0, = (build_paths(Scenario(spec["sample"], spec["sample"], nu=np.inf),
                               horizon_cal, h0_seeds)[1],)
        for delta in deltas:
            sc, _, D = _scenario_for(dist, delta, nu=nu)
            #del_seeds = make_seed_sequence(base_seed + 991 * int(100 * delta) + hash(dist) % 1000, reps_del)
            del_seeds = make_seed_sequence(base_seed + 991 * int(100 * delta) + zlib.crc32(dist.encode()) % 1000, reps_del)
            p_h1, raw_h1 = build_paths(sc, horizon_del, del_seeds)

            # oracle calibrato per questo scenario
            otr = compute_raw_trajectories(lambda llr=sc.llr: OracleCUSUM(llr), raw_h0)
            h_or = calibrate_alarm_threshold(otr, target_arl, horizon_cal)

            methods = _conformal_methods(direction, gamma_design, phase2)
            for name, fac in methods.items():
                taus = evaluate_detector_on_p_paths(fac, p_h1, conf_thr[(direction, name)])
                s = _cond_delay(taus, nu, horizon_del)
                rows.append(dict(distribution=dist, skew=round(spec["skew"], 2),
                                 direction=direction, delta=delta, method=name,
                                 mean_delay=s["mean_delay"], delay_ci95=s["delay_ci95"],
                                 trunc_delay=s["trunc_delay"], P_miss=s["P_miss"],
                                 P_false_before=s["P_false_before"],
                                 threshold=conf_thr[(direction, name)]))
            # oracle
            taus = evaluate_detector_on_raw_paths(lambda llr=sc.llr: OracleCUSUM(llr), raw_h1, h_or)
            s = _cond_delay(taus, nu, horizon_del)
            rows.append(dict(distribution=dist, skew=round(spec["skew"], 2),
                             direction=direction, delta=delta, method="Oracle",
                             mean_delay=s["mean_delay"], delay_ci95=s["delay_ci95"],
                             trunc_delay=s["trunc_delay"], P_miss=s["P_miss"],
                             P_false_before=s["P_false_before"], threshold=h_or))
            if verbose:
                print(f"  {dist:13s} delta={delta:<4} fatto")

    df = pd.DataFrame(rows)
    df.attrs["target_arl"] = target_arl

    # ---------- TABELLA DI SINTESI tra scenari ----------
    practical = ["CAM", "Power(fixed)", "GlobalPlugin"]
    # efficienza relativa all'oracle, per scenario
    piv = df.pivot_table(index=["distribution", "delta"], columns="method",
                         values="trunc_delay")
    rel = piv.div(piv["Oracle"], axis=0)          # ritardo / ritardo oracle
    miss = df.pivot_table(index=["distribution", "delta"], columns="method",
                          values="P_miss")
    summary_rows = []
    for m in practical:
        r = rel[m].values
        # vittorie: in quanti scenari m e' il piu' veloce tra i pratici
        wins = 0
        for idx in piv.index:
            vals = {mm: piv.loc[idx, mm] for mm in practical}
            if min(vals, key=vals.get) == m:
                wins += 1
        summary_rows.append(dict(
            method=m,
            rel_oracle_mean=float(np.nanmean(r)),
            rel_oracle_ci95=float(_ci95(r)),
            P_miss_mean=float(np.nanmean(miss[m].values)),
            wins=wins, n_scenari=len(piv.index)))
    summary = pd.DataFrame(summary_rows).sort_values("rel_oracle_mean").reset_index(drop=True)

    saved = {}
    if save:
        saved["table"] = save_df(df, "table_grid_level_shift.csv")
        saved["summary"] = save_df(summary, "table_grid_summary.csv")
    return dict(table=df, summary=summary, rel=rel, target_arl=target_arl, saved=saved)


# =====================================================================
# 14. SITUAZIONE 1 (stesso cast della griglia) -- ARL nominale vs realizzato
# =====================================================================
# Conformali a soglia NOMINALE gamma (la garanzia distribution-free: ARL>=gamma
# identico su tutte le distribuzioni). Baseline parametrico tarato sulla normale
# per ARL=gamma e applicato a tutte: rompe sulle asimmetriche.

def sim1_false_alarm_grid(
    gamma=200,
    dists=("Normal", "Gamma(k=2)", "Exponential", "Gumbel(max)"),
    horizon=2000, reps=2000, base_seed=1000, phase2="beta", k=0.5,
    save=True, verbose=True,
):
    """ARL nominale vs realizzato. Tutti i metodi progettati per lo STESSO ARL
    nominale = gamma. Conformali: soglia calibrata una volta sotto H0 (e' la
    stessa per tutte le distribuzioni perche' i p-value sono distribution-free).
    Parametrico: soglia tarata sulla NORMALE. Poi si misura l'ARL realizzato su
    ciascuna distribuzione: il conformale tiene ovunque, il parametrico rompe
    sulle asimmetriche."""
    seeds = make_seed_sequence(base_seed, reps)
    nrm = _dist_spec("Normal")
    p_n, raw_n = build_paths(Scenario(nrm["sample"], nrm["sample"], nu=np.inf), horizon, seeds)

    conf = {
        "CAM": lambda: CAMTwoSidedDetector(gamma, phase2=phase2),
        "PowerCUSUM": lambda: PowerCUSUM(0.5),
        "GlobalPlugin": lambda: GlobalPluginCUSUM(),
    }
    # soglie conformali calibrate a ARL=gamma (distribution-free: una volta, su Normale)
    conf_thr = {}
    for name, fac in conf.items():
        tj = compute_p_trajectories(fac, p_n)
        conf_thr[name] = calibrate_alarm_threshold(tj, target_arl=gamma, horizon=horizon)
    # soglia parametrica tarata sulla Normale
    param_tr = compute_raw_trajectories(lambda: GaussianCUSUM(0.0, 1.0, k), raw_n)
    h_param = calibrate_alarm_threshold(param_tr, target_arl=gamma, horizon=horizon)

    def _arl_row(dist, name, mtype, taus):
        cap = np.where(np.isfinite(taus), taus, horizon)
        return dict(distribution=dist, skew=round(_dist_spec(dist)["skew"], 2),
                    method=name, type=mtype, nominal_arl=gamma,
                    realized_arl=float(cap.mean()), realized_arl_ci95=float(_ci95(cap)),
                    p_false_within_gamma=float(np.mean(taus <= gamma)),
                    censored_pct=float(100 * np.mean(~np.isfinite(taus))))

    rows = []
    for dist in dists:
        spec = _dist_spec(dist)
        p_h0, raw_h0 = build_paths(Scenario(spec["sample"], spec["sample"], nu=np.inf), horizon, seeds)
        for name, fac in conf.items():
            taus = evaluate_detector_on_p_paths(fac, p_h0, conf_thr[name])
            rows.append(_arl_row(dist, name, "conformal", taus))
        taus = evaluate_detector_on_raw_paths(
            lambda mu=spec["mean"], sd=spec["sd"]: GaussianCUSUM(mu, sd, k), raw_h0, h_param)
        rows.append(_arl_row(dist, "GaussianCUSUM(param)", "parametric", taus))
        if verbose:
            print(f"  {dist:13s} fatto")

    df = pd.DataFrame(rows)
    saved = {}
    if save:
        saved["table"] = save_df(df, "table_sit1_arl_nominal_vs_realized.csv")
    return dict(table=df, conf_thresholds=conf_thr, h_param=h_param, saved=saved)


# =====================================================================
# 15. FIGURE -- rigenerazione in Colab dall'oggetto restituito dalla griglia
# =====================================================================


def make_regime_figure(out, save_dir=OUTPUT_DIR, style="bars", title=None,
                       figsize=None, logy=None):
    """Efficienza per regime dall'output di sim_level_shift_grid(...).
    style='bars' (barre per ampiezza di shift) o 'lines' (curve vs shift,
    mostra gli incroci). Etichette in inglese, niente titolo di default."""
    os.makedirs(save_dir, exist_ok=True)
    df = out["table"].copy()
    practical = [m for m in df["method"].unique() if m != "Oracle"]
    piv = df.pivot_table(index=["distribution", "delta"], columns="method", values="trunc_delay")
    rel = piv.div(piv["Oracle"], axis=0).reset_index()
    deltas = sorted(df["delta"].unique())
    colors = {"CAM": "C0", "Power(fixed)": "C3", "GlobalPlugin": "C2"}
    xticklab = [f"{'small' if d < 1 else 'medium' if d == 1 else 'large'} ({d})" for d in deltas]

    if style == "lines":
        logy = True if logy is None else logy
        fig, ax = plt.subplots(figsize=figsize or (5.6, 3.6))
        marks = {"CAM": "o", "Power(fixed)": "s", "GlobalPlugin": "^"}
        for mth in practical:
            means = [rel.loc[rel["delta"] == d, mth].mean() for d in deltas]
            errs = [_ci95(rel.loc[rel["delta"] == d, mth].values) for d in deltas]
            ax.errorbar(deltas, means, yerr=errs, fmt=marks.get(mth, "o"),
                        color=colors.get(mth), ms=6, capsize=3, label=mth)
        ax.axhline(1.0, color="k", ls="--", lw=1.1, label="oracle (=1)")
        ax.set_xticks(deltas); ax.set_xticklabels(xticklab, fontsize=9)
        ax.set_xlabel("Shift magnitude", fontsize=10)
        ax.set_ylabel("Delay relative to oracle", fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
        fname = "fig_regime_lines.png"
    else:  # bars
        logy = False if logy is None else logy
        fig, ax = plt.subplots(figsize=figsize or (5.8, 3.6))
        w = 0.8 / max(len(practical), 1); x = np.arange(len(deltas))
        for i, mth in enumerate(practical):
            means = [rel.loc[rel["delta"] == d, mth].mean() for d in deltas]
            errs = [_ci95(rel.loc[rel["delta"] == d, mth].values) for d in deltas]
            ax.bar(x + (i - (len(practical) - 1) / 2) * w, means, w, yerr=errs,
                   label=mth, color=colors.get(mth), alpha=0.9, capsize=2)
        ax.axhline(1.0, color="k", ls="--", lw=1.1, label="oracle (=1)")
        ax.set_xticks(x); ax.set_xticklabels(xticklab, fontsize=9)
        ax.set_xlabel("Shift magnitude", fontsize=10)
        ax.set_ylabel("Delay relative to oracle", fontsize=10)
        if logy:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7.5, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.1,
                  handlelength=1.3, handletextpad=0.5)
        fname = "fig_regime.png"

    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    p = os.path.join(save_dir, fname)
    fig.savefig(p, dpi=DEFAULT_FIG_DPI, bbox_inches="tight"); plt.close(fig)
    return p


def make_figures(out, save_dir=OUTPUT_DIR, nu_sweep=True,
                 nu_scenario=("Gamma(k=2)", 1.0), nus=(1, 25, 50, 100, 200, 400),
                 gamma_design=200, horizon_cal=1500, reps_cal=500,
                 reps_sweep=300, horizon_post=900, base_seed=40000, phase2="beta"):
    """Rigenera le due figure della Situazione 2 dall'output di
    sim_level_shift_grid(...).  Ritorna i percorsi salvati.

      fig_regime      : efficienza vs oracle per metodo x ampiezza di shift
                        (solo dai dati di `out`, nessun ricalcolo).
      fig_nu_dilution : ritardo vs nu (CAM vs GlobalPlugin); mostra che il
                        vantaggio del CAM cresce con la durata in-control.
                        Richiede una piccola risimulazione.
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = {}
    df = out["table"].copy()

    # ---------- Figura 1: efficienza per regime ----------
    paths["regime"] = make_regime_figure(out, save_dir=save_dir, style="bars")

    # ---------- Figura 2: sweep su nu (diluizione pre-change) ----------
    if nu_sweep:
        dist, delta = nu_scenario
        spec = _dist_spec(dist)
        one_sided = spec["direction"] != "two_sided"
        cam_factory = (lambda: CAMSmallPCUSUM(gamma_design, phase2=phase2)) if one_sided \
            else (lambda: CAMTwoSidedDetector(gamma_design, phase2=phase2))

        h0 = make_seed_sequence(base_seed, reps_cal)
        p_h0, _ = build_paths(Scenario(lambda r, s: r.normal(0, 1, s),
                                       lambda r, s: r.normal(0, 1, s), nu=np.inf), horizon_cal, h0)
        target, _ = _arl_cens_from_trajs(compute_p_trajectories(cam_factory, p_h0),
                                         gamma_design, horizon_cal)
        h_plug = calibrate_alarm_threshold(compute_p_trajectories(lambda: GlobalPluginCUSUM(), p_h0),
                                           target, horizon_cal)

        cam_d, plug_d = [], []
        for nu in nus:
            sc, _, _ = _scenario_for(dist, delta, nu=nu)
            H = nu + horizon_post
            seeds = make_seed_sequence(base_seed + 7 * nu, reps_sweep)
            pp, _ = build_paths(sc, H, seeds)
            tc = np.array([alarm_time_from_signal(cam_factory(), ps, gamma_design) for ps in pp], float)
            tp = np.array([alarm_time_from_signal(GlobalPluginCUSUM(), ps, h_plug) for ps in pp], float)
            cam_d.append(_cond_delay(tc, nu, H)["mean_delay"])
            plug_d.append(_cond_delay(tp, nu, H)["mean_delay"])

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(nus, cam_d, "o-", color="C0", label="CAM (two-stage, localised)")
        ax.plot(nus, plug_d, "s-", color="C2", label="GlobalPlugin (global estimate)")
        ax.set_xlabel(r"$\nu$  (in-control observations before the shift)")
        ax.set_ylabel("Conditional detection delay")
        ax.legend(frameon=False); ax.grid(alpha=0.3); fig.tight_layout()
        p = os.path.join(save_dir, "fig_nu_dilution.png"); fig.savefig(p, dpi=DEFAULT_FIG_DPI); plt.close(fig)
        paths["nu_dilution"] = p

    return paths


def make_sit1_figure(res1, save_dir=OUTPUT_DIR, conformal="CAM",
                     parametric="GaussianCUSUM(param)", style="bars",
                     title=None, figsize=None, dist_labels=None):
    """Figura Situazione 1 (ARL nominale vs realizzato) dal risultato di
    sim1_false_alarm_grid(...). Etichette in inglese, niente titolo di default
    (da mettere in LaTeX). style: 'bars' (barre per distribuzione) oppure
    'skew' (ARL vs asimmetria: mostra la degradazione monotona del parametrico)."""
    os.makedirs(save_dir, exist_ok=True)
    df = res1["table"]
    gamma = int(df["nominal_arl"].iloc[0])
    dists = list(dict.fromkeys(df["distribution"]))
    labmap = dist_labels or {"Gumbel(max)": "Gumbel"}
    lab = lambda d: labmap.get(d, d)

    def col(d, mth, c):
        sub = df[(df["distribution"] == d) & (df["method"] == mth)]
        return float(sub[c].values[0])

    cam = [col(d, conformal, "realized_arl") for d in dists]
    cam_e = [col(d, conformal, "realized_arl_ci95") for d in dists]
    par = [col(d, parametric, "realized_arl") for d in dists]
    par_e = [col(d, parametric, "realized_arl_ci95") for d in dists]
    skew = [col(d, conformal, "skew") for d in dists]

    if style == "skew":
        fig, ax = plt.subplots(figsize=figsize or (5.6, 3.6))
        order = np.argsort(skew)
        sk = np.array(skew)[order]
        camo, camoe = np.array(cam)[order], np.array(cam_e)[order]
        paro, paroe = np.array(par)[order], np.array(par_e)[order]
        names = [lab(dists[i]) for i in order]
        ax.errorbar(sk, camo, yerr=camoe, fmt="o", color="C0", ms=6,
                    capsize=3, label="CAM (conformal)")
        ax.errorbar(sk, paro, yerr=paroe, fmt="s", color="C3", ms=6,
                    capsize=3, label="Gaussian CUSUM (parametric)")
        ax.axhline(gamma, color="k", ls="--", lw=1.1, label=f"Nominal ARL = {gamma}")
        for xi, yi, nm in zip(sk, paro, names):
            ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=(4, -11),
                        fontsize=7.5, color="0.3")
        ax.set_xlabel("Skewness of in-control distribution", fontsize=10)
        ax.set_ylabel("In-control ARL", fontsize=10)
        ax.set_ylim(0, 1.25 * gamma)
        ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=1)
        fname = "fig_sit1_arl_skew.png"
    else:  # bars
        fig, ax = plt.subplots(figsize=figsize or (5.8, 3.6))
        x = np.arange(len(dists)); w = 0.36
        ax.bar(x - w / 2, cam, w, yerr=cam_e, color="C0", alpha=0.9, capsize=2,
               label="CAM (conformal)")
        ax.bar(x + w / 2, par, w, yerr=par_e, color="C3", alpha=0.9, capsize=2,
               label="Gaussian CUSUM (parametric)")
        ax.axhline(gamma, color="k", ls="--", lw=1.1, label=f"Nominal ARL = {gamma}")
        for i, d in enumerate(dists):
            if abs(par[i] - gamma) / gamma > 0.05:
                ax.text(x[i] + w / 2, par[i] + par_e[i] + 0.03 * gamma,
                        "%d%%" % round(100 * (par[i] / gamma - 1)), ha="center",
                        fontsize=8, color="C3")
        ax.set_xticks(x); ax.set_xticklabels([lab(d) for d in dists], fontsize=9)
        ax.set_ylabel("In-control ARL", fontsize=10)
        ax.set_ylim(0, 1.32 * gamma)
        ax.legend(frameon=False, fontsize=7.5, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.2,
                  handlelength=1.4, handletextpad=0.5)
        fname = "fig_sit1_arl.png"

    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(save_dir, fname)
    fig.savefig(p, dpi=DEFAULT_FIG_DPI, bbox_inches="tight"); plt.close(fig)
    return p


def make_pvalue_hist(dist="Exponential", delta=1.0, n_paths=300, horizon=300,
                     base_seed=12345, bins=20, save_dir=OUTPUT_DIR, figsize=None,
                     title=None):
    """Istogramma dei p-value conformali: sotto controllo (uniformi) vs dopo lo
    shift (massa verso 0). Mostra su COSA lavora il metodo."""
    os.makedirs(save_dir, exist_ok=True)
    spec = _dist_spec(dist); f0 = spec["sample"]
    p0 = np.concatenate(build_paths(Scenario(f0, f0, nu=np.inf), horizon,
                                    make_seed_sequence(base_seed, n_paths))[0])
    sc1, _, _ = _scenario_for(dist, delta, nu=1)
    p1 = np.concatenate(build_paths(sc1, horizon, make_seed_sequence(base_seed + 1, n_paths))[0])

    fig, axes = plt.subplots(1, 2, figsize=figsize or (6.6, 3.0), sharey=True)
    for ax, p, lab in zip(axes, [p0, p1], ["in control", "after shift"]):
        ax.hist(p, bins=bins, range=(0, 1), density=True, color="C0", alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform")
        ax.set_xlabel("conformal p-value", fontsize=10); ax.set_xlim(0, 1)
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("density", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.tick_params(labelsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    p = os.path.join(save_dir, "fig_pvalue_hist.png")
    fig.savefig(p, dpi=DEFAULT_FIG_DPI, bbox_inches="tight"); plt.close(fig)
    return p


def make_trajectory_plot(dist="Gamma(k=2)", delta=1.5, nu=100, gamma=200,
                         horizon=None, seed=None, phase2="beta",
                         save_dir=OUTPUT_DIR, figsize=None, title=None):
    """Una singola traiettoria della statistica CAM nel tempo: piatta sotto
    controllo, cresce dopo il changepoint, taglia la soglia. Mostra COME scatta
    l'allarme e le due fasi (esplorazione -> adattamento a gamma0)."""
    os.makedirs(save_dir, exist_ok=True)
    spec = _dist_spec(dist)
    one_sided = spec["direction"] != "two_sided"
    H = horizon or nu + 140
    det_factory = (lambda: CAMSmallPCUSUM(gamma, phase2=phase2)) if one_sided         else (lambda: CAMTwoSidedDetector(gamma, phase2=phase2))

    # scegli un seed che dia un allarme pulito poco dopo nu
    seeds = [seed] if seed is not None else list(range(40))
    M = None; alarm = None
    for sd in seeds:
        ps = build_paths(_scenario_for(dist, delta, nu=nu)[0], H, [sd])[0][0]
        det = det_factory(); m = np.empty(H); a = None
        for t, pp in enumerate(ps):
            m[t] = det.step(pp)
            if a is None and m[t] >= gamma:
                a = t + 1
        if seed is not None or (a is not None and nu + 3 <= a <= nu + 120):
            M, alarm = m, a; break
    if M is None:
        M, alarm = m, a

    t = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=figsize or (6.0, 3.4))
    ax.plot(t, np.maximum(M, 1e-3), color="C0", lw=1.4, label="CAM statistic")
    ax.axhline(gamma, color="C3", ls="--", lw=1.2, label=f"threshold = {gamma}")
    ax.axhline(math.sqrt(gamma), color="0.6", ls=":", lw=1.1,
               label="Phase I->II switch")
    ax.axvline(nu, color="k", ls="-.", lw=1.1, label=f"change point (nu={nu})")
    if alarm is not None:
        ax.plot(alarm, M[alarm - 1], "v", color="C3", ms=9, zorder=5)
        ax.annotate("alarm", (alarm, M[alarm - 1]), textcoords="offset points",
                    xytext=(4, 6), fontsize=8, color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("monitoring time", fontsize=10)
    ax.set_ylabel("detector value (log scale)", fontsize=10)
    ax.tick_params(labelsize=9); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    p = os.path.join(save_dir, "fig_trajectory.png")
    fig.savefig(p, dpi=DEFAULT_FIG_DPI, bbox_inches="tight"); plt.close(fig)
    return p



if __name__ == "__main__":
    print("Use cam_simulations_section7.ipynb to reproduce the paper; run_all() is a reduced-size smoke test.")
