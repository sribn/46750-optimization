from __future__ import annotations
import gurobipy as gp
from gurobipy import GRB
from .model import FlexibleConsumerModel


class SustainedDisutilityModel(FlexibleConsumerModel):
    """Q2.(e): disutility on the cumulative (signed) deviation from the reference.

        s_t = rho * s_{t-1} + (L_t - ref_t),  s_{-1} = 0
        min  sum_t [ p_imp*I - p_exp*E + c_PV*PV + c_S * s_t^2 ]      (convex QP)

    rho [-]            persistence of the deviation (0 <= rho < 1)
    c_S [DKK/kWh^2]    calibrated as (1 - rho^2) * c_Q with c_Q = 1  ->  0.36 for rho = 0.8
    Code hour t = report hour t + 1.
    """
    RHO_DEFAULT = 0.8
    C_S_DEFAULT = 1 - RHO_DEFAULT**2          # = 0.36

    def __init__(self, data, rho: float | None = None, c_S: float | None = None, **kw):
        super().__init__(data, **kw)
        self.rho = self.RHO_DEFAULT if rho is None else rho
        self.c_S = (1 - self.rho**2) if c_S is None and rho is not None else (self.C_S_DEFAULT if c_S is None else c_S)

    def build(self) -> "SustainedDisutilityModel":
        d, m, T = self.data, self.m, self.T
        p_imp = d.energy_price + d.import_tariff          # DKK/kWh
        p_exp = d.energy_price - d.export_tariff          # DKK/kWh
        ref, rho, c_S = d.reference_load, self.rho, self.c_S

        self.var["import"] = m.addVars(T, lb=0.0, name="import")
        self.var["export"] = m.addVars(T, lb=0.0, name="export")
        self.var["load"]   = m.addVars(T, lb=-GRB.INFINITY, name="load")
        self.var["pv"]     = m.addVars(T, lb=0.0, name="pv")
        self.var["cumulative_deviation"] = m.addVars(T, lb=-GRB.INFINITY, name="cumulative_deviation")
        I, E, L, PV, s = (self.var[k] for k in ("import", "export", "load", "pv", "cumulative_deviation"))

        m.setObjective(
            gp.quicksum(p_imp[t]*I[t] - p_exp[t]*E[t] + d.pv_marginal_cost*PV[t]
                        + c_S * s[t] * s[t] for t in T),
            GRB.MINIMIZE)

        # linear state recursion (signed deviation): s_t = rho*s_{t-1} + L_t - ref_t
        self.con["cumulative_deviation"] = m.addConstrs(
            (s[t] - (rho * s[t - 1] if t > 0 else 0.0) - (L[t] - ref[t]) == 0 for t in T),
            name="cumulative_deviation")
        self.con["balance"]  = m.addConstrs((L[t] - PV[t] - I[t] + E[t] == 0 for t in T), name="balance")
        self.con["pv_max"]   = m.addConstrs((PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")
        self.con["load_max"] = m.addConstrs((L[t] - d.load_max_kWh <= 0 for t in T), name="load_max")
        self.con["load_min"] = m.addConstrs((d.load_min_kWh - L[t] <= 0 for t in T), name="load_min")
        m.update()
        return self

    def breakdown(self, results) -> dict[str, float]:
        """Procurement cost, disutility, energy and deviation metrics (DKK, kWh)."""
        d, h = self.data, results.hourly
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        cost = float((p_imp*h["import"] - p_exp*h["export"] + d.pv_marginal_cost*h["pv"]).sum())
        dev = h["load"] - d.reference_load
        s = h["cumulative_deviation"]
        disutil = float((self.c_S * s**2).sum())
        return {"procurement_cost": cost, "disutility": disutil, "total": cost + disutil,
                "energy_kWh": float(h["load"].sum()), "abs_deviation_kWh": float(dev.abs().sum()),
                "hours_up": int((dev > 1e-6).sum()), "hours_down": int((dev < -1e-6).sum()),
                "max_abs_cumulative_deviation_kWh": float(s.abs().max()),
                "final_cumulative_deviation_kWh": float(s.iloc[-1]),
                "hours_at_lower_bound": int((h["load"] <= d.load_min_kWh + 1e-6).sum())}