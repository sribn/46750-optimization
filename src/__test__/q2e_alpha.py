"""Q2.(e) bonus: disutility NOT separable in time (accumulated-deviation / thermal-drift model).

Standalone script (data copied from data/*.json). To port into the repo: replace `load_base()`
by `load_question("Q2_quadratic")` and map the attributes (see README names).

Models compared (all: min  sum_t [ (p_t+tau_imp) I_t - (p_t-tau_exp) E_t + c_PV PV_t ] + D(d)):
  lin : D = cL * sum |d_t|                       (separable, LP)
  quad: D = cQ * sum d_t^2                       (separable, QP)
  acc : D = cS * sum s_t^2, s_t = rho s_{t-1} + d_t, s_0 = 0   (non-separable, QP)
with d_t = L_t - lref_t.
"""
import numpy as np, pandas as pd, gurobipy as gp
from gurobipy import GRB
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/mnt/user-data/outputs/q2e"); OUT.mkdir(parents=True, exist_ok=True)
T = range(24)

# ------------------------------------------------------------------ data (from data/*.json)
PRICE = np.array([1.1,1.05,1.0,0.9,0.85,1.01,1.05,1.2,1.4,1.6,1.5,1.1,1.05,1.0,0.95,1.0,1.2,1.5,2.1,2.5,2.2,1.8,1.4,1.2])
PVR = np.array([0,0,0,0,0,.05,.14,.21,.15,.12,.21,.25,.85,.75,.55,.43,.23,.05,.25,.25,0,0,0,0])
REFR = np.array([0,0,0,0,0,.04,.12,.25,.35,.45,.6,.7,.8,.78,.6,.4,.2,.08,.02,0,0,0,0,0])
D = dict(price=PRICE, pv=6.0*PVR, ref=6.0*REFR, Lmin=0.0, Lmax=6.0, cPV=1.41, t_imp=0.5, t_exp=0.4)

def solve(d, model, c=1.0, rho=0.7, with_duals=False):
    """Hourly schedule + metrics for model in {'lin','quad','acc'}. c = cL, cQ or cS."""
    m = gp.Model(); m.Params.OutputFlag = 0
    I  = m.addVars(T, lb=0); E = m.addVars(T, lb=0); PV = m.addVars(T, lb=0); L = m.addVars(T, lb=-GRB.INFINITY)
    dev = m.addVars(T, lb=-GRB.INFINITY)
    m.addConstrs((dev[t] == L[t] - d["ref"][t] for t in T), name="dev")
    m.addConstrs((I[t] - E[t] + PV[t] - L[t] == 0 for t in T), name="bal")
    m.addConstrs((PV[t] <= d["pv"][t] for t in T), name="pvmax")
    m.addConstrs((L[t] <= d["Lmax"] for t in T), name="Lmax")
    m.addConstrs((L[t] >= d["Lmin"] for t in T), name="Lmin")
    if model == "lin":
        a = m.addVars(T, lb=0)
        m.addConstrs((a[t] >= dev[t] for t in T)); m.addConstrs((a[t] >= -dev[t] for t in T))
        dis = c * gp.quicksum(a[t] for t in T)
    elif model == "quad":
        dis = c * gp.quicksum(dev[t]*dev[t] for t in T)
    else:
        s = m.addVars(T, lb=-GRB.INFINITY)
        m.addConstr(s[0] == dev[0])
        m.addConstrs((s[t] == rho*s[t-1] + dev[t] for t in range(1, 24)))
        dis = c * gp.quicksum(s[t]*s[t] for t in T)
    proc = gp.quicksum((d["price"][t]+d["t_imp"])*I[t] - (d["price"][t]-d["t_exp"])*E[t] + d["cPV"]*PV[t] for t in T)
    m.setObjective(proc + dis, GRB.MINIMIZE); m.optimize()
    assert m.Status == GRB.OPTIMAL
    h = pd.DataFrame({k: [v[t].X for t in T] for k, v in dict(imp=I, exp=E, pv=PV, load=L, dev=dev).items()})
    h["ref"] = d["ref"]; h["price"] = d["price"]
    if with_duals: h["dual_bal"] = [m.getConstrByName(f"bal[{t}]").Pi for t in T]
    pc = proc.getValue(); dv = dis.getValue()
    return h, dict(objective=m.ObjVal, procurement=pc, disutility=dv,
                   energy=h.load.sum(), abs_dev=h.dev.abs().sum(),
                   n_up=int((h.dev > 1e-6).sum()), n_down=int((h.dev < -1e-6).sum()),
                   max_run=max_run(h.dev.values))

def max_run(dv, tol=1e-6):
    """Longest run of consecutive hours with same-sign deviation."""
    best = cur = 0; last = 0
    for x in dv:
        sg = 0 if abs(x) < tol else (1 if x > 0 else -1)
        cur = cur + 1 if (sg != 0 and sg == last) else (1 if sg != 0 else 0)
        last = sg; best = max(best, cur)
    return best

def ref_cost(d):
    """Procurement cost when following the reference exactly (zero disutility): the baseline."""
    h, _ = solve(dict(d, Lmin=0, Lmax=6), "quad", c=1e9)   # huge c pins L = ref
    return h, _["procurement"]

if __name__ == "__main__":
    RHO = 0.7; CS = 1 - RHO**2        # calibration: isolated 1 kWh deviation costs 1 DKK (= cQ = 1)
    # ---------------- base case
    hq, mq = solve(D, "quad", 1.0)
    ha, ma = solve(D, "acc", CS, RHO)
    hl, ml = solve(D, "lin", 1.43)
    base = pd.DataFrame({"linear(cL=1.43)": ml, "quadratic(cQ=1)": mq, f"accumulated(cS={CS:.2f},rho={RHO})": ma}).T
    print(base.round(3).to_string()); base.round(4).to_csv(OUT/"base_metrics.csv")
    ha.round(4).to_csv(OUT/"base_acc_hourly.csv"); hq.round(4).to_csv(OUT/"base_quad_hourly.csv")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].step(range(24), D["ref"], where="mid", color="k", ls=":", label="reference")
    for h, n, col in [(hl, "linear", "C2"), (hq, "quadratic", "C0"), (ha, "accumulated (non-sep.)", "C3")]:
        ax[0].step(range(24), h.load, where="mid", color=col, label=n)
    ax[0].set_ylabel("load [kWh/h]"); ax[0].legend(fontsize=8)
    w = 0.38
    ax[1].bar(np.arange(24)-w/2, hq.dev, w, color="C0", label="quadratic"); ax[1].bar(np.arange(24)+w/2, ha.dev, w, color="C3", label="accumulated")
    ax[1].axhline(0, color="grey", lw=.8); ax[1].set_ylabel("deviation d [kWh/h]"); ax[1].set_xlabel("hour"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"base_schedule.png", dpi=150)

    # ---------------- experiment: temporal structure of prices (isolated: flat reference, no PV)
    # Same multiset of 24 prices (mean m, +/- delta) arranged as square waves with block length k.
    # Flat reference 3 kWh/h and PV off remove every other source of temporal structure.
    mean = PRICE.mean(); delta = 0.5; CL_EXP = 1.8      # cL between low (1.32) and high (2.32) effective import price
    DX = dict(D, ref=np.full(24, 3.0), pv=np.zeros(24))
    rows = []; profiles = {}
    for k in [1, 2, 3, 4, 6, 12]:
        sq = np.where((np.arange(24)//k) % 2 == 0, 1.0, -1.0)
        p = mean + delta*sq; profiles[k] = p
        dd = dict(DX, price=p)
        base_cost = float(((p + D["t_imp"]) * dd["ref"]).sum())   # cost of following the reference (zero disutility)
        for name, c in [("lin", CL_EXP), ("quad", 1.0), ("acc", CS)]:
            h, m_ = solve(dd, name, c, RHO)
            rows.append(dict(block=k, model=name, **m_, ref_cost=base_cost, flex_value=base_cost - m_["objective"]))
    ex = pd.DataFrame(rows); ex["surplus"] = -ex.objective
    ex.round(4).to_csv(OUT/"experiment_price_structure.csv", index=False)
    print(ex[["block","model","objective","procurement","disutility","abs_dev","max_run","flex_value"]].round(3).to_string(index=False))
    print("identical price multiset:", all(np.allclose(np.sort(profiles[1]), np.sort(profiles[k])) for k in profiles))
    piv = ex.pivot(index="block", columns="model", values="flex_value"); pdv = ex.pivot(index="block", columns="model", values="abs_dev")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for mdl, col in [("lin", "C2"), ("quad", "C0"), ("acc", "C3")]:
        ax[0].plot(piv.index, piv[mdl], "o-", color=col, label=mdl); ax[1].plot(pdv.index, pdv[mdl], "o-", color=col, label=mdl)
    for a, y in zip(ax, ["value of flexibility [DKK/day]", "total |deviation| [kWh]"]):
        a.set_xscale("log", base=2); a.set_xticks([1,2,3,4,6,12]); a.set_xticklabels([1,2,3,4,6,12]); a.set_xlabel("high/low price block length [h]"); a.set_ylabel(y); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"experiment_price_structure.png", dpi=150)
    # schedules for the two extreme profiles under the three models
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a, k in zip(ax, [1, 12]):
        dd = dict(DX, price=profiles[k]); a2 = a.twinx(); a2.step(range(24), profiles[k], where="mid", color="grey", lw=1); a2.set_ylabel("price [DKK/kWh]")
        for name, c, col in [("lin", CL_EXP, "C2"), ("quad", 1.0, "C0"), ("acc", CS, "C3")]:
            a.step(range(24), solve(dd, name, c, RHO)[0].load, where="mid", color=col, label=name)
        a.axhline(3, color="k", ls=":"); a.set_ylabel(f"load [kWh/h], block={k}"); a.legend(fontsize=8, loc="lower right")
    ax[1].set_xlabel("hour"); fig.tight_layout(); fig.savefig(OUT/"experiment_schedules.png", dpi=150)
