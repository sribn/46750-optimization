from __future__ import annotations
import gurobipy as gp
from gurobipy import GRB
from .model import FlexibleConsumerModel


class QuadraticDisutilityModel(FlexibleConsumerModel):
    """Q2.(c): min  p_imp*I - p_exp*E + c_PV*PV + c_Q*(L - l_ref)^2  (convex QP)."""

    def build(self) -> "QuadraticDisutilityModel":
        d, m, T = self.data, self.m, self.T
        p_imp = d.energy_price + d.import_tariff          # DKK/kWh
        p_exp = d.energy_price - d.export_tariff          # DKK/kWh
        cQ, ref = d.quadratic_disutility, d.reference_load

        # variables (lb=0 default is fine for I, E, PV; load is bounded by constraints)
        self.var["import"] = m.addVars(T, lb=0.0, name="import")
        self.var["export"] = m.addVars(T, lb=0.0, name="export")
        self.var["load"]   = m.addVars(T, lb=-GRB.INFINITY, name="load")
        self.var["pv"]     = m.addVars(T, lb=0.0, name="pv")
        I, E, L, PV = (self.var[k] for k in ("import", "export", "load", "pv"))

        # objective: procurement cost + quadratic disutility
        m.setObjective(
            gp.quicksum(p_imp[t]*I[t] - p_exp[t]*E[t] + d.pv_marginal_cost*PV[t]
                        + cQ*(L[t] - ref[t])*(L[t] - ref[t]) for t in T),
            GRB.MINIMIZE)

        # constraints (explicit, so duals are returned as dual_<name>)
        self.con["balance"] = m.addConstrs((L[t] - PV[t] - I[t] + E[t] == 0 for t in T), name="balance")
        self.con["pv_max"]  = m.addConstrs((PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")
        self.con["load_max"] = m.addConstrs((L[t] - d.load_max_kWh <= 0 for t in T), name="load_max")
        self.con["load_min"] = m.addConstrs((d.load_min_kWh - L[t] <= 0 for t in T), name="load_min")
        m.update()
        return self

    def breakdown(self, results) -> dict[str, float]:
        """Procurement cost, total disutility, energy and absolute deviation (DKK, kWh)."""
        d, h = self.data, results.hourly
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        cost = float((p_imp*h["import"] - p_exp*h["export"] + d.pv_marginal_cost*h["pv"]).sum())
        dev = h["load"] - d.reference_load
        disutil = float((d.quadratic_disutility * dev**2).sum())
        return {"procurement_cost": cost, "disutility": disutil, "total": cost + disutil,
                "energy_kWh": float(h["load"].sum()), "abs_deviation_kWh": float(dev.abs().sum()),
                "hours_at_lower_bound": int((h["load"] <= d.load_min_kWh + 1e-6).sum())}