"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, Results
from src.model_q2b import LinearDisutilityModel
from src.model_q2c import QuadraticDisutilityModel
from src.model_q2e import SustainedDisutilityModel
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs

# ------------------------------------------------- Q2 (C) -------------------------------------------------

def run_q2_quadratic(question: str, out: Path, show: bool) -> Results:
    """Q2.(c): base case of the quadratic-disutility consumer (schedule, duals, cost/disutility split)."""
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = QuadraticDisutilityModel(data).build()
    results = model.solve()

    print(results, "\n")
    print({k: round(v, 3) for k, v in model.breakdown(results).items()}, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_sweep_cQ(out: Path, values=None) -> pd.DataFrame:
    """Q2.(c).iv: sweep the quadratic coefficient c^Q (DKK/kWh^2) on the Q2_quadratic data.

    Saves sweep_cQ.csv (cost, disutility, energy, deviation, hours at lower bound per c^Q)
    and sweep_cQ_load.csv (hourly load of every run).
    """
    values = np.unique(np.append(np.logspace(-1.3, 1, 9), 1.0)) if values is None else values
    base = load_question("Q2_quadratic")
    rows, loads = [], {}
    for c in values:
        model = QuadraticDisutilityModel(replace(base, quadratic_disutility=float(c))).build()
        res = model.solve()
        rows.append({"cQ": float(c), **model.breakdown(res)})
        loads[f"cQ={c:.3g}"] = res.hourly["load"]
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep_cQ.csv", index=False)
    pd.DataFrame(loads).to_csv(out / "sweep_cQ_load.csv", index_label="hour")
    print(df.round(3).to_string(index=False))
    return df

# ------------------------------------------------- Q2 (E) -------------------------------------------------

def check_q2e(question: str = "Q2_quadratic") -> None:
    """Q2.(e) validation: (1) rho = 0 reproduces Q2.(c); (2) breakdown adds up to the solver objective."""
    data = load_question(question)

    res_c = QuadraticDisutilityModel(data).build().solve()
    res_e = SustainedDisutilityModel(data, rho=0.0, c_S=1.0).build().solve()
    assert abs(res_e.objective - res_c.objective) < 1e-6, (res_e.objective, res_c.objective)
    assert (res_e.hourly["load"] - res_c.hourly["load"]).abs().max() < 1e-6
    print(f"Check 1 passed: rho = 0 reproduces Q2.(c) (objective {res_c.objective:.4f} DKK)")

    model = SustainedDisutilityModel(data).build()
    res = model.solve()
    b = model.breakdown(res)
    assert abs(b["total"] - res.objective) < 1e-6, (b["total"], res.objective)
    print("Check 2 passed: breakdown total equals solver objective\n")


def run_q2e_base(out: Path, show: bool, rho: float | None = None, c_S: float | None = None) -> Results:
    """Q2.(e): base case of the cumulative-disutility consumer (default rho = 0.8, c_S = 0.36)."""
    data = load_question("Q2_quadratic")
    model = SustainedDisutilityModel(data, rho=rho, c_S=c_S).build()
    results = model.solve()
    print(f"rho = {model.rho}, c_S = {model.c_S:.4f}")
    print(results, "\n")
    print({k: round(v, 3) for k, v in model.breakdown(results).items()}, "\n")
    results.save(out, tag="Q2e")
    plot_schedule(results, data, save_to=out / "schedule_Q2e.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_q2e_price_structure(out: Path, rho: float = 0.8, c_S: float = 0.36,
                            c_L: float = 1.8, c_Q: float = 1.0, delta: float = 0.5,
                            blocks=(1, 2, 3, 4, 6, 12)) -> pd.DataFrame:
    """Q2.(e).iv: effect of the temporal structure of prices on the models of (b), (c) and (e).

    Arranged as a square wave of block length k, solved with the high block first and the low block first.
    Flat reference (3 kWh/h) and no PV isolate the price order. Saves experiment_price_structure.csv and experiment_price_structure.png.
    """

    base = load_question("Q2_quadratic")
    n = base.n_hours
    ref = np.full(n, 3.0)
    mean_price = float(base.energy_price.mean())
    zeros = np.zeros(n)
    flat = replace(base, reference_load=ref, pv_profile=zeros, pv_available=zeros,
                   linear_disutility=c_L, quadratic_disutility=c_Q)

    models = {
        "linear":     lambda d: LinearDisutilityModel(d),
        "quadratic":  lambda d: QuadraticDisutilityModel(d),
        "cumulative": lambda d: SustainedDisutilityModel(d, rho=rho, c_S=c_S),
    }
    rows = []
    for start, sign in (("high first", 1.0), ("low first", -1.0)):
        for k in blocks:
            wave = np.where((np.arange(n) // k) % 2 == 0, 1.0, -1.0)
            price = mean_price + sign * delta * wave
            data = replace(flat, energy_price=price)
            ref_cost = float(((price + data.import_tariff) * ref).sum())   # following the reference exactly
            for name, make in models.items():
                res = make(data).build().solve()
                dev = res.hourly["load"] - ref
                rows.append({"start": start, "block": k, "model": name,
                             "surplus": -res.objective,
                             "flex_value": ref_cost - res.objective,
                             "abs_dev": float(dev.abs().sum()),
                             "hours_up": int((dev > 1e-6).sum()),
                             "hours_down": int((dev < -1e-6).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "experiment_price_structure.csv", index=False)
    print(df.pivot_table(index=["start", "block"], columns="model", values="surplus").round(2).to_string())
    print(df.pivot_table(index=["start", "block"], columns="model", values="abs_dev").round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, start in zip(axes, ("high first", "low first")):
        for name in models:
            q = df[(df.start == start) & (df.model == name)]
            ax.plot(q.block, q.surplus, "o-", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(blocks)); ax.set_xticklabels(list(blocks))
        ax.set_xlabel("price block length [h]"); ax.set_title(start); ax.legend(fontsize=8)
    axes[0].set_ylabel("daily surplus [DKK]")
    fig.tight_layout(); fig.savefig(out / "experiment_price_structure.png", dpi=150)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    parser.add_argument("--sweep-cq", action="store_true", help="run the c^Q sweep of Q2.(c).iv")
    parser.add_argument("--q2e", action="store_true", help="run the checks and the base case of Q2.(e)")
    parser.add_argument("--q2e-prices", action="store_true", help="run the price-structure experiment of Q2.(e).iv")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    if args.question == "Q2_quadratic":
        run_q2_quadratic(args.question, out, args.show)
        if args.sweep_cq:
            run_sweep_cQ(out)
        if args.q2e:
            check_q2e(args.question)
            run_q2e_base(out, args.show)
        if args.q2e_prices:
            run_q2e_price_structure(out)
        print(f"\nOutputs written to {out}")
        return

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()