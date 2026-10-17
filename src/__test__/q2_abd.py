"""Question 2.(a), 2.(b) and 2.(d): separable disutility models of a flexible consumer.

Self-contained script for my parts of Question 2. It reuses the shared starter code (``src/``) and the
quadratic-disutility model of Question 2.(c) (``src/model_q2.py``, written by a teammate) for the
comparison of 2.(d). Place this file in the repository root (next to ``main.py``) and run

    python q2_my_parts.py              # all parts: 2.(a) check, 2.(b) base case + sweep, 2.(d)
    python q2_my_parts.py --part a     # only 2.(a): Question 1 as a special case of Question 2
    python q2_my_parts.py --part b     # only 2.(b): linear disutility, base case + sweep of c^L
    python q2_my_parts.py --part d     # only 2.(d): linear vs quadratic vs Question 1

Outputs (CSV + PNG) are written to ``results/Q2_linear/`` (2.(b)) and ``results/Q2_comparison/``
(2.(a) and 2.(d)).

Models (all minimise a daily cost in DKK)::

    SeparableDisutilityModel       2.(a)  cost(I, E, PV) + sum_t D_t(L_t)  (generic; D_t given by a subclass)
      |- LinearDisutilityModel     2.(b)  D_t = c^L |L_t - l_ref_t|      (LP via dev_up, dev_down >= 0)
      |- UtilityAsDisutilityModel  2.(a)  D_t = u (L_max - L_t)          (Question 1 as a special case)
      |- LinearUtilityModel        Q1     D_t = -u L_t                   (Question 1, reference for 2.(d))
    QuadraticDisutilityModel       2.(c)  imported from src/model_q2.py

with cost_t = p_imp_t I_t - p_exp_t E_t + c_PV PV_t, p_imp = p + tau_imp, p_exp = p - tau_exp (DKK/kWh).
Loads, PV, imports and exports are in kWh/h (= kW over one hour).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from src.data_loader import InputData, load_question
from src.model import FlexibleConsumerModel, Results
from src.model_q2 import QuadraticDisutilityModel
from src.plotting import _finish, plot_duals, plot_inputs, plot_schedule

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EPS = 1e-6   # kWh/h, tolerance used to classify hours (at a bound, deviating, ...)


# =============================================================================== models
def effective_prices(d: InputData) -> tuple[np.ndarray, np.ndarray]:
    """Effective import and export prices p_imp = p + tau_imp, p_exp = p - tau_exp (DKK/kWh, 24 values)."""
    return d.energy_price + d.import_tariff, d.energy_price - d.export_tariff


class SeparableDisutilityModel(FlexibleConsumerModel):
    """2.(a): min sum_t [p_imp I - p_exp E + c_PV PV + D_t(L_t)] subject to the constraints of Question 1.

    Subclasses implement ``_disutility(t)`` (gurobipy expression, DKK) and ``disutility_values(load)``
    (numpy, DKK per hour), and may add auxiliary variables/constraints in ``_add_auxiliary()``.
    ``reference`` (kWh/h, 24 values) overrides ``data.reference_load`` when given.
    """

    def __init__(self, data: InputData, reference: np.ndarray | None = None, **kwargs):
        super().__init__(data, **kwargs)
        self.ref = np.asarray(data.reference_load if reference is None else reference, dtype=float)

    def _add_auxiliary(self) -> None:
        """Hook for auxiliary variables/constraints (none for a differentiable disutility)."""

    def _disutility(self, t: int):
        raise NotImplementedError

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def build(self) -> "SeparableDisutilityModel":
        d, m, T = self.data, self.m, self.T
        p_imp, p_exp = effective_prices(d)

        # variables: I, E, PV >= 0 as variable bounds; the load bounds are explicit constraints (duals)
        self.var["import"] = m.addVars(T, lb=0.0, name="import")
        self.var["export"] = m.addVars(T, lb=0.0, name="export")
        self.var["load"]   = m.addVars(T, lb=-GRB.INFINITY, name="load")
        self.var["pv"]     = m.addVars(T, lb=0.0, name="pv")
        I, E, L, PV = (self.var[k] for k in ("import", "export", "load", "pv"))
        self._add_auxiliary()

        # objective: procurement cost + disutility
        m.setObjective(gp.quicksum(p_imp[t]*I[t] - p_exp[t]*E[t] + d.pv_marginal_cost*PV[t]
                                   + self._disutility(t) for t in T), GRB.MINIMIZE)

        # constraints (explicit, so their duals are returned as dual_<name>)
        self.con["balance"]  = m.addConstrs((L[t] - PV[t] - I[t] + E[t] == 0 for t in T), name="balance")
        self.con["pv_max"]   = m.addConstrs((PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")
        self.con["load_max"] = m.addConstrs((L[t] - d.load_max_kWh <= 0 for t in T), name="load_max")
        self.con["load_min"] = m.addConstrs((d.load_min_kWh - L[t] <= 0 for t in T), name="load_min")
        m.update()
        return self


class LinearDisutilityModel(SeparableDisutilityModel):
    """2.(b): D_t = c^L |L_t - l_ref_t|, reformulated as an LP.

    L_t - l_ref_t = dev_up_t - dev_down_t,  dev_up_t, dev_down_t >= 0,  D_t = c^L (dev_up_t + dev_down_t).
    With c^L > 0 at most one of dev_up_t, dev_down_t is positive at the optimum, so their sum is |L - l_ref|.
    """

    def _add_auxiliary(self) -> None:
        m, T, L = self.m, self.T, self.var["load"]
        self.var["dev_up"]   = m.addVars(T, lb=0.0, name="dev_up")     # kWh/h above the reference
        self.var["dev_down"] = m.addVars(T, lb=0.0, name="dev_down")   # kWh/h below the reference
        up, dn = self.var["dev_up"], self.var["dev_down"]
        self.con["deviation"] = m.addConstrs((L[t] - self.ref[t] - up[t] + dn[t] == 0 for t in T),
                                             name="deviation")

    def _disutility(self, t: int):
        return self.data.linear_disutility * (self.var["dev_up"][t] + self.var["dev_down"][t])

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        return self.data.linear_disutility * np.abs(load - self.ref)


class UtilityAsDisutilityModel(SeparableDisutilityModel):
    """2.(a): Question 1 as a special case: l_ref_t = L_max and D_t = u (L_max - L_t) >= 0.

    Its objective equals the Question 1 objective (cost - u sum_t L_t) plus the constant 24 u L_max.
    """

    def __init__(self, data: InputData, **kwargs):
        super().__init__(data, reference=np.full(data.n_hours, data.load_max_kWh), **kwargs)

    def _disutility(self, t: int):
        return self.data.consumption_utility * (self.ref[t] - self.var["load"][t])

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        return self.data.consumption_utility * (self.ref - load)


class LinearUtilityModel(SeparableDisutilityModel):
    """Question 1: min sum_t cost_t - u L_t (constant marginal utility u, no reference profile).

    ``reference`` (e.g. the Question 2 profile) is only used to measure deviations in the metrics; the
    reported "disutility" is then minus the utility, -u L_t (DKK).
    """

    def _disutility(self, t: int):
        return -self.data.consumption_utility * self.var["load"][t]

    def disutility_values(self, load: np.ndarray) -> np.ndarray:
        return -self.data.consumption_utility * load


def quadratic_disutility_values(d: InputData, load: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """D_t = c^Q (L_t - l_ref_t)^2 of the teammate's 2.(c) model (DKK per hour)."""
    return d.quadratic_disutility * (load - ref)**2


# ============================================================ 2.(b).ii closed-form hypotheses
def marginal_costs(d: InputData) -> tuple[np.ndarray, np.ndarray]:
    """Hourly marginal cost (DKK/kWh) of serving one more kWh of load.

    m_low:  load below the PV availability: that PV kWh would otherwise be curtailed (saves c_PV) or
            exported (earns p_exp), so it costs max(c_PV, p_exp), capped at p_imp (PV dearer than
            imports is never used).
    m_high: load above the PV availability is imported: p_imp.
    """
    p_imp, p_exp = effective_prices(d)
    return np.minimum(p_imp, np.maximum(d.pv_marginal_cost, p_exp)), p_imp


def predict_linear_load(d: InputData, c_lin: float) -> np.ndarray:
    """2.(b).ii closed form of the optimal hourly load for D_t = c^L |L_t - l_ref_t| (kWh/h).

    Starting from l_ref, a kWh is shed iff its marginal cost exceeds c^L. Consuming above l_ref never pays
    (marginal cost > 0 > -c^L), so the load jumps between three candidates:
    l_ref (c^L > p_imp), max(L_min, min(l_ref, PV_max)) (m_low < c^L < p_imp), L_min (c^L < m_low).
    Ties (c^L equal to a threshold) give a non-unique optimum and are not covered.
    """
    m_low, m_high = marginal_costs(d)
    ref, lmin, pv = d.reference_load, d.load_min_kWh, d.pv_available
    low_part = np.where(c_lin > m_low, np.maximum(0.0, np.minimum(ref, pv) - lmin), 0.0)
    high_part = np.where(c_lin > m_high, np.maximum(0.0, ref - np.maximum(pv, lmin)), 0.0)
    return lmin + low_part + high_part


def linear_thresholds(d: InputData) -> list[float]:
    """Sorted values of c^L (DKK/kWh) at which the optimal load of at least one hour jumps."""
    m_low, m_high = marginal_costs(d)
    ref, lmin, pv = d.reference_load, d.load_min_kWh, d.pv_available
    has_low = np.minimum(ref, pv) - lmin > EPS          # some reference load can be served from PV
    has_high = ref - np.maximum(pv, lmin) > EPS         # some reference load must be imported
    return sorted({float(v) for v in np.round(np.r_[m_low[has_low], m_high[has_high]], 6)})


# ============================================================================== metrics
def daily_metrics(d: InputData, hourly: pd.DataFrame, disutility: np.ndarray, ref: np.ndarray) -> dict:
    """Daily metrics of 2.(b).iv / 2.(c).iv / 2.(d).

    procurement_cost, disutility, total (DKK); energy_kWh, abs_deviation_kWh (kWh);
    hours_deviating: hours with |L - l_ref| > 0;
    hours_dev_bound_binding: hours where the deviation reaches its bound, i.e. the load is fully shed
    (L = L_min < l_ref) or pushed to L_max > l_ref.
    """
    p_imp, p_exp = effective_prices(d)
    load = hourly["load"].to_numpy()
    cost = float((p_imp*hourly["import"] - p_exp*hourly["export"] + d.pv_marginal_cost*hourly["pv"]).sum())
    dev = load - ref
    at_min, at_max = load <= d.load_min_kWh + EPS, load >= d.load_max_kWh - EPS
    return {"procurement_cost": cost, "disutility": float(disutility.sum()), "total": cost + float(disutility.sum()),
            "energy_kWh": float(load.sum()), "abs_deviation_kWh": float(np.abs(dev).sum()),
            "hours_deviating": int((np.abs(dev) > EPS).sum()),
            "hours_dev_bound_binding": int(((at_min & (ref > d.load_min_kWh + EPS))
                                            | (at_max & (ref < d.load_max_kWh - EPS))).sum())}


def solve_with_metrics(kind: str, data: InputData, reference: np.ndarray | None = None) -> tuple[Results, dict]:
    """Build and solve one model (``kind`` in linear / quadratic / utility) and return (results, metrics)."""
    ref = np.asarray(data.reference_load if reference is None else reference, dtype=float)
    if kind == "quadratic":
        res = QuadraticDisutilityModel(data).build().solve()
        disutil = quadratic_disutility_values(data, res.hourly["load"].to_numpy(), ref)
    else:
        model = {"linear": LinearDisutilityModel, "utility": LinearUtilityModel}[kind](data, reference=ref).build()
        res = model.solve()
        disutil = model.disutility_values(res.hourly["load"].to_numpy())
    return res, daily_metrics(data, res.hourly, disutil, ref)


def sweep(kind: str, base: InputData, attr: str, values, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solve model ``kind`` for every value of the coefficient ``attr`` of ``base``.

    Returns (metrics, loads): one row of ``daily_metrics`` per value (column ``label``), and the hourly
    load of every run (hours x values, kWh/h).
    """
    rows, loads = [], {}
    for c in values:
        res, met = solve_with_metrics(kind, replace(base, **{attr: float(c)}))
        rows.append({label: float(c), **met})
        loads[float(c)] = res.hourly["load"].to_numpy()
    return pd.DataFrame(rows), pd.DataFrame(loads, index=pd.Index(range(base.n_hours), name="hour"))


# ============================================================================= plotting
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED, PV_FILL, GRID = "#52514e", "#f6d58a", "#e4e3df"


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def plot_loads_vs_reference(loads: dict[str, np.ndarray], reference: np.ndarray, pv_available: np.ndarray,
                            title: str, save_to: Path) -> plt.Figure:
    """Hourly load of one or several runs (``label -> 24 values``) against l_ref and the available PV."""
    h = np.arange(len(reference))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(h, pv_available, step="mid", color=PV_FILL, alpha=0.6, lw=0, label="PV available")
    ax.step(h, reference, where="mid", color=MUTED, ls="--", lw=1.5, label="reference $l^{ref}$")
    for i, (label, load) in enumerate(loads.items()):
        ax.step(h, load, where="mid", color=SERIES[i % len(SERIES)], lw=2, label=label)
    ax.set(xlabel="hour", ylabel="kWh/h", title=title, xticks=h[::2])
    _style(ax)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    return _finish(fig, save_to)


def plot_sweep_metrics(df: pd.DataFrame, x: str, xlabel: str, title: str, thresholds: list[float],
                       save_to: Path) -> plt.Figure:
    """2x2 panels of the sweep metrics against c^L. Between two consecutive ``thresholds`` the metrics are
    constant, so each run is drawn as a flat segment over its regime (faint vertical lines = thresholds)."""
    df = df.sort_values(x)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    panels = [(axes[0, 0], [("energy_kWh", "daily energy")], "kWh"),
              (axes[0, 1], [("abs_deviation_kWh", "total |deviation|")], "kWh"),
              (axes[1, 0], [("procurement_cost", "procurement cost"), ("disutility", "disutility"),
                            ("total", "total")], "DKK"),
              (axes[1, 1], [("hours_deviating", "hours deviating"),
                            ("hours_dev_bound_binding", "hours deviation bound binding")], "hours")]
    edges = np.r_[df[x].min(), thresholds, df[x].max()]
    for ax, cols, unit in panels:
        for i, (col, label) in enumerate(cols):
            for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                v = df[col][(df[x] >= a) & (df[x] <= b)]
                if len(v):
                    ax.hlines(v.iloc[0], a, b, color=SERIES[i], lw=2, label=label if j == 0 else None)
            ax.plot(df[x], df[col], "o", ms=4, color=SERIES[i])
        for th in thresholds:
            ax.axvline(th, color=MUTED, lw=0.5, alpha=0.4)
        ax.set_ylabel(unit)
        _style(ax)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11)
    return _finish(fig, save_to)


def plot_load_heatmap(loads: pd.DataFrame, reference: np.ndarray, xlabel: str, title: str,
                      save_to: Path) -> plt.Figure:
    """Load as a share of the reference (hours x swept values). Hours with l_ref = 0 are left blank."""
    with np.errstate(divide="ignore", invalid="ignore"):
        share = loads.to_numpy() / reference[:, None]
    share[reference <= 1e-9, :] = np.nan
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(share, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(loads.shape[1]), [f"{c:.3g}" for c in loads.columns], rotation=90, fontsize=7)
    ax.set(xlabel=xlabel, ylabel="hour", title=title)
    fig.colorbar(im, ax=ax, label="load / reference [-]")
    return _finish(fig, save_to)


def plot_hourly_response(curves: dict[str, pd.DataFrame], hours: list[int], save_to: Path) -> plt.Figure:
    """2.(d): load of selected hours vs the disutility coefficient, one panel per model.

    ``curves`` maps "panel title|x-axis label" to a DataFrame indexed by the coefficient, one column per hour.
    """
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8), sharey=True)
    for ax, (key, df) in zip(axes, curves.items()):
        title, xlabel = key.split("|")
        for i, t in enumerate(hours):
            ax.plot(df.index, df[t], "o-", ms=3, lw=2, color=SERIES[i], label=f"hour {t}")
        ax.set(xscale="log", xlabel=xlabel, title=title)
        _style(ax)
    axes[0].set_ylabel("load [kWh/h]")
    axes[0].legend(fontsize=8, frameon=False)
    return _finish(fig, save_to)


def plot_model_metrics(df: pd.DataFrame, save_to: Path) -> plt.Figure:
    """2.(d): horizontal bars of energy, total deviation and procurement cost per scenario."""
    metrics = [("energy_kWh", "daily energy [kWh]"), ("abs_deviation_kWh", "total |deviation| [kWh]"),
               ("procurement_cost", "procurement cost [DKK]")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.45 * len(df) + 1.6), sharey=True)
    y = np.arange(len(df))
    colors = [SERIES[{"linear": 0, "quadratic": 1, "Q1 utility": 2}[m]] for m in df["model"]]
    for ax, (col, label) in zip(axes, metrics):
        ax.barh(y, df[col], color=colors, height=0.7)
        for yi, v in zip(y, df[col]):
            ax.text(max(v, 0), yi, f" {v:.1f}", va="center", fontsize=7, color=MUTED)
        ax.set(title=label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, df["scenario"], fontsize=8)
    axes[0].invert_yaxis()
    return _finish(fig, save_to)


# ============================================================================ experiments
def run_part_a(out: Path) -> dict[str, float]:
    """2.(a): on Q1_caseA, Question 1 and the disutility model with l_ref = L_max, D = u (L_max - L) give the
    same load and objectives that differ by the constant 24 u L_max. Saves q2a_special_case_check.csv."""
    data = load_question("Q1_caseA")
    r_q1 = LinearUtilityModel(data, reference=np.zeros(data.n_hours)).build().solve()
    r_d = UtilityAsDisutilityModel(data).build().solve()
    check = {"Q1_objective": r_q1.objective, "disutility_objective": r_d.objective,
             "difference": r_d.objective - r_q1.objective,
             "24_u_Lmax": data.n_hours * data.consumption_utility * data.load_max_kWh,
             "max_load_gap_kWh": float(np.abs(r_q1.hourly["load"] - r_d.hourly["load"]).max())}
    pd.Series(check).to_csv(out / "q2a_special_case_check.csv", header=["value"])
    print("2.(a) check:", {k: round(v, 4) for k, v in check.items()}, "\n")
    return check


def cL_sweep_values(d: InputData) -> np.ndarray:
    """One c^L per regime: midpoints between consecutive thresholds, one value below the smallest and one
    above the largest, plus the base value - every regime of 2.(b).ii is visited without hitting a tie."""
    th = np.array(linear_thresholds(d))
    return np.unique(np.round(np.r_[th[0] / 2, (th[:-1] + th[1:]) / 2, th[-1] + 0.5, d.linear_disutility], 4))


def run_part_b(out: Path) -> pd.DataFrame:
    """2.(b).iv: base case of Q2_linear (schedule, duals, metrics, check of the 2.(b).ii prediction) and a
    sweep of c^L (DKK/kWh). Saves Q2_linear_hourly.csv / _summary.txt, q2b_base_metrics.csv, sweep_cL.csv,
    sweep_cL_load.csv, cL_thresholds.csv and the figures."""
    data = load_question("Q2_linear")
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    res, met = solve_with_metrics("linear", data)
    gap = np.abs(res.hourly["load"].to_numpy() - predict_linear_load(data, data.linear_disutility)).max()
    print(res, "\n", {k: round(v, 3) for k, v in met.items()})
    print(f"max |solver load - predicted load 2.(b).ii| = {gap:.2e} kWh/h\n")
    res.save(out)
    pd.Series(met).to_csv(out / "q2b_base_metrics.csv", header=["value"])
    plot_schedule(res, data, save_to=out / "schedule.png")
    plot_duals(res, data, save_to=out / "duals.png")
    plot_loads_vs_reference({f"optimal load, $c^L$={data.linear_disutility:g}": res.hourly["load"].to_numpy()},
                            data.reference_load, data.pv_available, "2.(b) linear disutility - base case",
                            save_to=out / "load_vs_reference.png")

    # sweep of c^L over all regimes
    df, loads = sweep("linear", data, "linear_disutility", cL_sweep_values(data), "cL")
    df["matches_prediction"] = [bool(np.allclose(loads[c], predict_linear_load(data, c), atol=1e-5))
                                for c in df["cL"]]
    th = linear_thresholds(data)
    m_low, m_high = marginal_costs(data)
    ref, pv, lmin = data.reference_load, data.pv_available, data.load_min_kWh
    has_low = np.minimum(ref, pv) - lmin > EPS          # hours with a PV-served part of the reference
    has_high = ref - np.maximum(pv, lmin) > EPS         # hours with an imported part of the reference
    hours_at = lambda m, mask, v: ", ".join(str(t) for t in np.flatnonzero(np.isclose(m, v) & mask))
    pd.DataFrame({"cL_threshold": th,
                  "hours_PV_layer (max(c_PV, p_exp))": [hours_at(m_low, has_low, v) for v in th],
                  "hours_import_layer (p_imp)": [hours_at(m_high, has_high, v) for v in th]}
                 ).to_csv(out / "cL_thresholds.csv", index=False)
    df.to_csv(out / "sweep_cL.csv", index=False)
    loads.rename(columns=lambda c: f"cL={c:.3g}").to_csv(out / "sweep_cL_load.csv")
    print(df.round(3).to_string(index=False))
    print(f"\npredicted c^L thresholds (DKK/kWh): {th}\n")
    plot_sweep_metrics(df, "cL", "$c^L$ [DKK/kWh]", "2.(b).iv sweep of the linear disutility coefficient",
                       th, save_to=out / "sweep_cL.png")
    plot_load_heatmap(loads, data.reference_load, "$c^L$ [DKK/kWh]",
                      "2.(b).iv optimal load / reference per hour and $c^L$", save_to=out / "sweep_cL_heatmap.png")
    return df


def run_part_d(out: Path, cL_values=(0.5, 1.43, 3.5), cQ_values=(0.1, 1.0, 10.0),
               response_hours=(8, 10, 13)) -> pd.DataFrame:
    """2.(d): compare Question 1 (linear utility, Q1_caseA), the linear (2.(b)) and the quadratic (2.(c))
    disutility models. Scenarios = base case of each model plus ``cL_values`` / ``cQ_values`` (one value per
    regime of the sweeps: below, around and above the effective-price thresholds). Deviations of the
    Question 1 model are measured against the Question 2 reference. Saves q2d_comparison.csv, the profiles,
    the metric bars and the response of ``response_hours`` to c^L and c^Q."""
    lin, quad, q1 = load_question("Q2_linear"), load_question("Q2_quadratic"), load_question("Q1_caseA")
    ref, pv = lin.reference_load, lin.pv_available

    res_q1, met_q1 = solve_with_metrics("utility", q1, reference=ref)
    rows = [{"model": "Q1 utility", "scenario": f"Q1 u={q1.consumption_utility:g}", **met_q1}]
    profiles: dict[str, dict[str, np.ndarray]] = {"linear": {}, "quadratic": {}}
    for kind, base, attr, values in [("linear", lin, "linear_disutility", cL_values),
                                     ("quadratic", quad, "quadratic_disutility", cQ_values)]:
        for c in values:
            res, met = solve_with_metrics(kind, replace(base, **{attr: float(c)}))
            tag = f"{kind} c={c:g}" + (" (base)" if np.isclose(c, getattr(base, attr)) else "")
            rows.append({"model": kind, "scenario": tag, **met})
            profiles[kind][tag] = res.hourly["load"].to_numpy()
    df = pd.DataFrame(rows)
    df.to_csv(out / "q2d_comparison.csv", index=False)
    print(df.round(3).to_string(index=False), "\n")

    for kind in profiles:
        plot_loads_vs_reference(profiles[kind], ref, pv, f"2.(d) {kind} disutility: load profiles",
                                save_to=out / f"q2d_profiles_{kind}.png")
    base_profiles = {"Q1 utility (no reference)": res_q1.hourly["load"].to_numpy(),
                     **{k: v for p in profiles.values() for k, v in p.items() if "(base)" in k}}
    plot_loads_vs_reference(base_profiles, ref, pv, "2.(d) base cases of the three objectives",
                            save_to=out / "q2d_profiles_base.png")
    plot_model_metrics(df, save_to=out / "q2d_metrics.png")

    # response of single hours to the coefficient: jumps (linear) vs gradual (quadratic)
    _, loads_l = sweep("linear", lin, "linear_disutility", np.logspace(-1, np.log10(4), 60), "cL")
    _, loads_q = sweep("quadratic", quad, "quadratic_disutility", np.logspace(-2, 1.5, 60), "cQ")
    hours = list(response_hours)
    plot_hourly_response({"linear disutility|$c^L$ [DKK/kWh]": loads_l.loc[hours].T,
                          "quadratic disutility|$c^Q$ [DKK/kWh$^2$]": loads_q.loc[hours].T},
                         hours, save_to=out / "q2d_hourly_response.png")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["a", "b", "d", "all"], default="all", help="which part of Q2 to run")
    args = parser.parse_args()

    out_b, out_cmp = RESULTS_DIR / "Q2_linear", RESULTS_DIR / "Q2_comparison"
    for p in (out_b, out_cmp):
        p.mkdir(parents=True, exist_ok=True)
    if args.part in ("a", "all"):
        run_part_a(out_cmp)
    if args.part in ("b", "all"):
        run_part_b(out_b)
    if args.part in ("d", "all"):
        run_part_d(out_cmp)
    print(f"Outputs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
