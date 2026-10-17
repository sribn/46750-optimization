from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .model import FlexibleConsumerModel


class LinearDisutilityModel(FlexibleConsumerModel):
    """Q2.(b): min p_imp*I - p_exp*E + c_PV*PV + c_L*|L - l_ref| (LP)."""

    def build(self) -> "LinearDisutilityModel":
        d, m, T = self.data, self.m, self.T
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        cL, ref = d.linear_disutility, d.reference_load

        # Decision variables
        self.var["import"] = m.addVars(T, lb=0.0, name="import")
        self.var["export"] = m.addVars(T, lb=0.0, name="export")
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, name="load")
        self.var["pv"] = m.addVars(T, lb=0.0, name="pv")
        self.var["dev_up"] = m.addVars(T, lb=0.0, name="dev_up")
        self.var["dev_down"] = m.addVars(T, lb=0.0, name="dev_down")
        I, E, L, PV, up, down = (
            self.var[k] for k in ("import", "export", "load", "pv", "dev_up", "dev_down")
        )

        # Objective: procurement cost plus absolute-deviation disutility
        m.setObjective(
            gp.quicksum(
                p_imp[t] * I[t]
                - p_exp[t] * E[t]
                + d.pv_marginal_cost * PV[t]
                + cL * (up[t] + down[t])
                for t in T
            ),
            GRB.MINIMIZE,
        )

        # Constraints are explicit so their duals are included in Results
        self.con["balance"] = m.addConstrs(
            (L[t] - PV[t] - I[t] + E[t] == 0 for t in T), name="balance"
        )
        self.con["pv_max"] = m.addConstrs(
            (PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max"
        )
        self.con["load_max"] = m.addConstrs(
            (L[t] - d.load_max_kWh <= 0 for t in T), name="load_max"
        )
        self.con["load_min"] = m.addConstrs(
            (d.load_min_kWh - L[t] <= 0 for t in T), name="load_min"
        )
        self.con["deviation"] = m.addConstrs(
            (L[t] - ref[t] - up[t] + down[t] == 0 for t in T), name="deviation"
        )

        m.update()
        return self

    def breakdown(self, results) -> dict[str, float]:
        """Return procurement cost, disutility and daily energy/deviation metrics."""
        d, h = self.data, results.hourly
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        cost = float(
            (p_imp * h["import"] - p_exp * h["export"] + d.pv_marginal_cost * h["pv"]).sum()
        )
        deviation = h["load"] - d.reference_load
        disutility = float((d.linear_disutility * deviation.abs()).sum())
        return {
            "procurement_cost": cost,
            "disutility": disutility,
            "total": cost + disutility,
            "energy_kWh": float(h["load"].sum()),
            "abs_deviation_kWh": float(deviation.abs().sum()),
            "hours_deviating": int((deviation.abs() > 1e-6).sum()),
            "hours_at_lower_bound": int((h["load"] <= d.load_min_kWh + 1e-6).sum()),
        }


def marginal_costs(data) -> tuple[np.ndarray, np.ndarray]:
    """Return marginal costs below PV availability and above PV availability (DKK/kWh)."""
    p_imp = data.energy_price + data.import_tariff
    p_exp = data.energy_price - data.export_tariff
    m_low = np.minimum(p_imp, np.maximum(data.pv_marginal_cost, p_exp))
    return m_low, p_imp


def predict_linear_load(data, c_L: float) -> np.ndarray:
    """Closed-form candidate load for Q2.(b).ii, away from threshold ties."""
    m_low, m_high = marginal_costs(data)
    ref, lmin, pv = data.reference_load, data.load_min_kWh, data.pv_available
    low_part = np.where(c_L > m_low, np.maximum(0.0, np.minimum(ref, pv) - lmin), 0.0)
    high_part = np.where(c_L > m_high, np.maximum(0.0, ref - np.maximum(pv, lmin)), 0.0)
    return lmin + low_part + high_part


def linear_thresholds(data, tolerance: float = 1e-6) -> list[float]:
    """Sorted c^L values where at least one hourly load can change regime."""
    m_low, m_high = marginal_costs(data)
    ref, lmin, pv = data.reference_load, data.load_min_kWh, data.pv_available
    has_low = np.minimum(ref, pv) - lmin > tolerance
    has_high = ref - np.maximum(pv, lmin) > tolerance
    return sorted({float(v) for v in np.round(np.r_[m_low[has_low], m_high[has_high]], 6)})
