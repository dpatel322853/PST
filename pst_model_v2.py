
"""
pst_model_v2.py
Screening / engineering estimation model for LP Ethylene Superheater low-temperature PST.

IMPORTANT:
- This tool is NOT a final process safety design basis.
- Do not use for final design, IPL credit, SIS validation, or MOC approval without
  validation against plant data, detailed design data, approved valve data, and
  approved engineering methods.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import math
import json
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Pipe DB (minimal, extendable)
# -----------------------------------------------------------------------------
NPS_DB = {
    2: {"od_m": 0.06033, "schedules_mm": {"5": 1.65, "10": 2.11, "40": 3.91, "80": 5.54}},
    3: {"od_m": 0.08890, "schedules_mm": {"10": 3.05, "40": 5.49, "80": 7.62}},
    4: {"od_m": 0.11430, "schedules_mm": {"10": 3.05, "40": 6.02, "80": 8.56}},
    6: {"od_m": 0.16828, "schedules_mm": {"10": 3.40, "40": 7.11, "80": 10.97}},
    8: {"od_m": 0.21908, "schedules_mm": {"10": 3.76, "40": 8.18, "80": 12.70}},
    10: {"od_m": 0.27305, "schedules_mm": {"10": 4.19, "40": 9.27, "80": 12.70}},
    12: {"od_m": 0.32385, "schedules_mm": {"10": 4.57, "40": 10.31, "80": 17.48}},
    14: {"od_m": 0.35560, "schedules_mm": {"10": 4.78, "40": 11.13, "80": 19.05}},
    16: {"od_m": 0.40640, "schedules_mm": {"10": 4.78, "40": 12.70, "80": 21.44}},
}

AIR_DEFAULTS = {
    "rho": 1.165,
    "mu": 1.85e-5,
    "k": 0.0263,
    "cp": 1007.0,
}

R_UNIV = 8.31446261815324  # J/mol.K
SQRT2 = math.sqrt(2.0)


@dataclass
class Geometry:
    nps_in: float
    od_m: float
    wall_thickness_m: float
    id_m: float
    area_flow_m2: float
    area_i_per_m: float
    area_o_per_m: float
    vol_fluid_per_m3_per_m: float
    vol_metal_per_m3_per_m: float


@dataclass
class MaterialProps:
    density: float
    cp: float
    k: float


@dataclass
class PREthyleneConfig:
    # Ethylene critical data / PR EOS inputs
    name: str = "Ethylene"
    molecular_weight_kg_per_mol: float = 0.02805316
    tc_k: float = 282.34
    pc_pa: float = 5.041e6
    acentric: float = 0.087
    # Ideal-gas Cp approximation (mass basis). User should update if better data exist.
    cp_ideal_ref_jkgk: float = 2200.0
    cp_ideal_ref_temp_k: float = 273.15
    cp_ideal_slope_jkgk2: float = 0.0
    # Transport properties (screening approximation). Pressure dependence neglected.
    mu_ref_pa_s: float = 9.0e-6
    mu_ref_temp_k: float = 273.15
    mu_exp: float = 0.7
    k_ref_w_mk: float = 0.020
    k_ref_temp_k: float = 273.15
    k_exp: float = 0.7
    phase_root: str = "vapor_largest_z"


@dataclass
class SectionInputs:
    name: str
    length_m: float
    geometry: Geometry
    metal: MaterialProps
    n_cells: int
    pressure_bar_a: float


@dataclass
class SensorConfig:
    tag: str
    tau_s: float = 60.0
    offset_c: float = 0.0
    fail_mode: str = "healthy"  # healthy, stuck_high, stuck_low, frozen_initial
    frozen_value_c: Optional[float] = None


@dataclass
class ModelInputs:
    superheater_inlet_temp_c: float
    initial_superheater_outlet_temp_c: float
    failure_target_temp_c: float
    superheater_tau_s: float
    model_mode: str  # bounding, dynamic, ramp
    failure_start_time_s: float = 0.0
    ramp_duration_s: float = 60.0

    pr_fluid: PREthyleneConfig = field(default_factory=PREthyleneConfig)
    flowrate_kgph: float = 31600.0

    section_a: SectionInputs = None
    section_b: SectionInputs = None

    ambient_temp_c: float = 30.0
    air_velocity_mps: float = 1.5
    air: Dict[str, float] = field(default_factory=lambda: dict(AIR_DEFAULTS))

    tt_setpoint_c: float = 0.0
    mdmt_c: float = -29.0

    sensors: List[SensorConfig] = field(default_factory=list)
    vote_k: int = 2
    vote_n: int = 3
    logic_solver_delay_s: float = 0.0

    xv_closure_time_s: float = 10.0
    xv_dead_time_s: float = 0.0
    valve_characteristic: str = "linear"  # linear, equal_percentage_like, stuck_open, fail_partial
    valve_exponent: float = 1.0
    valve_min_open_frac: float = 0.0

    initial_pipe_metal_temp_c: Optional[float] = None
    initial_pipe_fluid_temp_c: Optional[float] = None

    t_end_s: float = 3600.0
    output_dt_s: float = 1.0
    cfl_target: float = 0.3


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def k_to_c(t_k: float) -> float:
    return t_k - 273.15


def calculate_geometry(nps_in: float,
                       schedule: Optional[str] = None,
                       wall_thickness_mm: Optional[float] = None) -> Geometry:
    if nps_in not in NPS_DB:
        raise ValueError(f"NPS {nps_in} not found in DB")
    od_m = NPS_DB[nps_in]["od_m"]
    if wall_thickness_mm is None:
        if schedule is None:
            raise ValueError("Either schedule or wall_thickness_mm must be provided")
        wall_thickness_m = NPS_DB[nps_in]["schedules_mm"][schedule] / 1000.0
    else:
        wall_thickness_m = wall_thickness_mm / 1000.0

    id_m = od_m - 2.0 * wall_thickness_m
    if id_m <= 0.0:
        raise ValueError("Invalid thickness; ID <= 0")

    area_flow_m2 = math.pi * id_m**2 / 4.0
    area_i_per_m = math.pi * id_m
    area_o_per_m = math.pi * od_m
    vol_fluid_per_m3_per_m = area_flow_m2
    vol_metal_per_m3_per_m = math.pi * (od_m**2 - id_m**2) / 4.0
    return Geometry(
        nps_in=nps_in,
        od_m=od_m,
        wall_thickness_m=wall_thickness_m,
        id_m=id_m,
        area_flow_m2=area_flow_m2,
        area_i_per_m=area_i_per_m,
        area_o_per_m=area_o_per_m,
        vol_fluid_per_m3_per_m=vol_fluid_per_m3_per_m,
        vol_metal_per_m3_per_m=vol_metal_per_m3_per_m,
    )


def calculate_pipe_metal_mass(geometry: Geometry, length_m: float, metal_density: float) -> float:
    return geometry.vol_metal_per_m3_per_m * length_m * metal_density


def calculate_fluid_hold_up_mass(geometry: Geometry, length_m: float, fluid_density: float) -> float:
    return geometry.vol_fluid_per_m3_per_m * length_m * fluid_density


def calculate_fluid_velocity(m_dot_kg_s: float, fluid_density: float, area_flow_m2: float) -> float:
    if fluid_density <= 0 or area_flow_m2 <= 0:
        return 0.0
    return m_dot_kg_s / (fluid_density * area_flow_m2)


def reynolds_number(rho: float, velocity: float, diameter: float, mu: float) -> float:
    if mu <= 0:
        return np.nan
    return rho * velocity * diameter / mu


def prandtl_number(cp: float, mu: float, k: float) -> float:
    if k <= 0:
        return np.nan
    return cp * mu / k


class PREthylene:
    def __init__(self, cfg: PREthyleneConfig):
        self.cfg = cfg
        self.R = R_UNIV
        self.MW = cfg.molecular_weight_kg_per_mol
        self.Tc = cfg.tc_k
        self.Pc = cfg.pc_pa
        self.omega = cfg.acentric
        self.a0 = 0.45724 * self.R**2 * self.Tc**2 / self.Pc
        self.b = 0.07780 * self.R * self.Tc / self.Pc
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2

    def alpha(self, T_k: float) -> float:
        tr = max(T_k / self.Tc, 1e-9)
        m = 1.0 + self.kappa * (1.0 - math.sqrt(tr))
        return m * m

    def dalpha_dT(self, T_k: float) -> float:
        tr = max(T_k / self.Tc, 1e-9)
        m = 1.0 + self.kappa * (1.0 - math.sqrt(tr))
        return -(self.kappa * m) / (self.Tc * math.sqrt(tr))

    def pr_roots(self, T_k: float, P_pa: float) -> np.ndarray:
        aT = self.a0 * self.alpha(T_k)
        A = aT * P_pa / (self.R**2 * T_k**2)
        B = self.b * P_pa / (self.R * T_k)
        coeffs = [1.0, -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)]
        roots = np.roots(coeffs)
        roots = np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)])
        roots = np.sort(roots)
        return roots

    def z_factor(self, T_k: float, P_pa: float, phase: Optional[str] = None) -> float:
        phase = phase or self.cfg.phase_root
        roots = self.pr_roots(T_k, P_pa)
        if len(roots) == 0:
            raise ValueError("No real PR Z-factor roots found")
        if phase in ("vapor_largest_z", "vapor"):
            return float(np.max(roots))
        if phase in ("liquid_smallest_z", "liquid"):
            return float(np.min(roots))
        return float(np.max(roots))

    def density_kg_m3(self, T_k: float, P_pa: float, phase: Optional[str] = None) -> float:
        Z = self.z_factor(T_k, P_pa, phase)
        rho = P_pa * self.MW / (max(Z, 1e-9) * self.R * T_k)
        return rho

    def cp_ideal_mass(self, T_k: float) -> float:
        cfg = self.cfg
        return max(100.0, cfg.cp_ideal_ref_jkgk + cfg.cp_ideal_slope_jkgk2 * (T_k - cfg.cp_ideal_ref_temp_k))

    def h_residual_molar_j_mol(self, T_k: float, P_pa: float, phase: Optional[str] = None) -> float:
        Z = self.z_factor(T_k, P_pa, phase)
        aT = self.a0 * self.alpha(T_k)
        daT_dT = self.a0 * self.dalpha_dT(T_k)
        B = self.b * P_pa / (self.R * T_k)
        logterm = math.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
        hres = self.R * T_k * (Z - 1.0) + ((T_k * daT_dT - aT) / (2.0 * SQRT2 * self.b)) * logterm
        return hres

    def cp_real_mass(self, T_k: float, P_pa: float, phase: Optional[str] = None) -> float:
        # cp = cp_ideal + d(H_res)/dT|P; numerical derivative for maintainability
        dT = max(0.5, 0.002 * T_k)
        T1 = max(130.0, T_k - dT)
        T2 = T_k + dT
        hres1 = self.h_residual_molar_j_mol(T1, P_pa, phase)
        hres2 = self.h_residual_molar_j_mol(T2, P_pa, phase)
        cp_res_molar = (hres2 - hres1) / (T2 - T1)
        cp_real = self.cp_ideal_mass(T_k) + cp_res_molar / self.MW
        return max(100.0, cp_real)

    def mu_pa_s(self, T_k: float, P_pa: float) -> float:
        # Screening transport approximation: temperature-dependent only.
        cfg = self.cfg
        return max(1e-7, cfg.mu_ref_pa_s * (T_k / cfg.mu_ref_temp_k) ** cfg.mu_exp)

    def k_w_mk(self, T_k: float, P_pa: float) -> float:
        cfg = self.cfg
        return max(1e-4, cfg.k_ref_w_mk * (T_k / cfg.k_ref_temp_k) ** cfg.k_exp)

    def props_mass_basis(self, T_c: float, P_bar_a: float) -> Dict[str, float]:
        T_k = c_to_k(T_c)
        P_pa = max(1e3, P_bar_a * 1e5)
        rho = self.density_kg_m3(T_k, P_pa)
        cp = self.cp_real_mass(T_k, P_pa)
        mu = self.mu_pa_s(T_k, P_pa)
        k = self.k_w_mk(T_k, P_pa)
        return {"rho": rho, "cp": cp, "mu": mu, "k": k, "z": self.z_factor(T_k, P_pa)}


def build_pr_property_table(pr: PREthylene, pressure_bar_a: float, t_min_c: float = -120.0, t_max_c: float = 120.0, n: int = 481) -> Dict[str, np.ndarray]:
    temps_c = np.linspace(t_min_c, t_max_c, n)
    rho = np.zeros(n)
    cp = np.zeros(n)
    mu = np.zeros(n)
    k = np.zeros(n)
    z = np.zeros(n)
    for i, tc in enumerate(temps_c):
        props = pr.props_mass_basis(float(tc), pressure_bar_a)
        rho[i] = props["rho"]
        cp[i] = props["cp"]
        mu[i] = props["mu"]
        k[i] = props["k"]
        z[i] = props["z"]
    return {"temps_c": temps_c, "rho": rho, "cp": cp, "mu": mu, "k": k, "z": z}


def interp_pr_props(table: Dict[str, np.ndarray], t_c: float) -> Dict[str, float]:
    tc = float(np.clip(t_c, table["temps_c"][0], table["temps_c"][-1]))
    return {
        "rho": float(np.interp(tc, table["temps_c"], table["rho"])),
        "cp": float(np.interp(tc, table["temps_c"], table["cp"])),
        "mu": float(np.interp(tc, table["temps_c"], table["mu"])),
        "k": float(np.interp(tc, table["temps_c"], table["k"])),
        "z": float(np.interp(tc, table["temps_c"], table["z"])),
    }


def internal_heat_transfer_coefficient(m_dot_kg_s: float, rho: float, cp: float, mu: float, k: float,
                                       geometry: Geometry) -> Tuple[float, float, float, float]:
    v = calculate_fluid_velocity(m_dot_kg_s, rho, geometry.area_flow_m2)
    re = reynolds_number(rho, v, geometry.id_m, mu)
    pr = prandtl_number(cp, mu, k)
    if np.isnan(re) or np.isnan(pr) or re <= 0 or pr <= 0:
        return 1.0, v, re, pr
    if re < 2300.0:
        nu = 3.66
    else:
        nu = 0.023 * (re ** 0.8) * (pr ** 0.4)
    h = max(1.0, nu * k / geometry.id_m)
    return h, v, re, pr


def external_heat_transfer_coefficient(air_velocity_mps: float, outer_diameter_m: float,
                                       air: Dict[str, float]) -> Tuple[float, float, float]:
    rho = air["rho"]
    mu = air["mu"]
    k = air["k"]
    cp = air["cp"]
    re = reynolds_number(rho, air_velocity_mps, outer_diameter_m, mu)
    pr = prandtl_number(cp, mu, k)
    if re <= 0 or pr <= 0:
        nu = 1.0
    else:
        nu = 0.3 + ((0.62 * re**0.5 * pr**(1/3)) / ((1 + (0.4 / pr)**(2/3))**0.25)) * ((1 + (re / 282000.0)**(5/8))**(4/5))
    h = max(1.0, nu * k / outer_diameter_m)
    return h, re, pr


def overall_uo(hi: float, ho: float, geometry: Geometry, metal_k: float) -> float:
    ri = geometry.id_m / 2.0
    ro = geometry.od_m / 2.0
    ai = 2.0 * math.pi * ri
    ao = 2.0 * math.pi * ro
    r_i = 1.0 / max(hi * ai, 1e-12)
    r_w = math.log(max(ro / ri, 1.0 + 1e-12)) / max(2.0 * math.pi * metal_k, 1e-12)
    r_o = 1.0 / max(ho * ao, 1e-12)
    return 1.0 / (max((r_i + r_w + r_o) * ao, 1e-12))


def valve_open_fraction(t_s: float, trip_time_s: Optional[float], inp: ModelInputs) -> float:
    if trip_time_s is None:
        return 1.0
    if inp.valve_characteristic == "stuck_open":
        return 1.0
    t_start = trip_time_s + inp.logic_solver_delay_s + inp.xv_dead_time_s
    if t_s < t_start:
        return 1.0
    if inp.xv_closure_time_s <= 0:
        return inp.valve_min_open_frac
    x = max(inp.valve_min_open_frac, 1.0 - (t_s - t_start) / inp.xv_closure_time_s)
    if inp.valve_characteristic == "equal_percentage_like":
        x = inp.valve_min_open_frac + (x - inp.valve_min_open_frac) ** max(inp.valve_exponent, 1e-6)
    elif inp.valve_characteristic == "fail_partial":
        x = max(inp.valve_min_open_frac, x)
    return float(min(1.0, max(inp.valve_min_open_frac, x)))


def superheater_outlet_temperature(t_s: float, inp: ModelInputs) -> float:
    if t_s <= inp.failure_start_time_s:
        return inp.initial_superheater_outlet_temp_c
    te = t_s - inp.failure_start_time_s
    t0 = inp.initial_superheater_outlet_temp_c
    tf = inp.failure_target_temp_c
    mode = inp.model_mode.lower()
    if mode == "bounding":
        return tf
    if mode == "dynamic":
        tau = max(inp.superheater_tau_s, 1e-6)
        return tf + (t0 - tf) * math.exp(-te / tau)
    if mode == "ramp":
        frac = min(1.0, te / max(inp.ramp_duration_s, 1e-9))
        return t0 + frac * (tf - t0)
    return tf


def build_grid(section_a: SectionInputs, section_b: SectionInputs) -> pd.DataFrame:
    rows = []
    x = 0.0
    for sec_idx, sec in enumerate([section_a, section_b]):
        dx = sec.length_m / sec.n_cells
        for i in range(sec.n_cells):
            rows.append({
                "section": sec.name,
                "section_idx": sec_idx,
                "cell_idx_in_section": i,
                "dx": dx,
                "x_start": x,
                "x_mid": x + 0.5 * dx,
                "x_end": x + dx,
                "id_m": sec.geometry.id_m,
                "od_m": sec.geometry.od_m,
                "ai": sec.geometry.area_i_per_m * dx,
                "ao": sec.geometry.area_o_per_m * dx,
                "vol_f": sec.geometry.vol_fluid_per_m3_per_m * dx,
                "vol_m": sec.geometry.vol_metal_per_m3_per_m * dx,
                "metal_rho": sec.metal.density,
                "metal_cp": sec.metal.cp,
                "metal_k": sec.metal.k,
                "pressure_bar_a": sec.pressure_bar_a,
            })
            x += dx
    return pd.DataFrame(rows)


def recommend_time_step(inp: ModelInputs) -> float:
    pr = PREthylene(inp.pr_fluid)
    # conservative estimate using coldest and highest density / velocity among sections at initial full flow
    m_dot = inp.flowrate_kgph / 3600.0
    dens = []
    for sec in [inp.section_a, inp.section_b]:
        props = pr.props_mass_basis(min(inp.initial_superheater_outlet_temp_c, inp.failure_target_temp_c), sec.pressure_bar_a)
        dens.append(props["rho"])
    vA = calculate_fluid_velocity(m_dot, max(dens[0],1e-6), inp.section_a.geometry.area_flow_m2)
    vB = calculate_fluid_velocity(m_dot, max(dens[1],1e-6), inp.section_b.geometry.area_flow_m2)
    dx_min = min(inp.section_a.length_m / inp.section_a.n_cells, inp.section_b.length_m / inp.section_b.n_cells)
    dt_cfl = inp.cfl_target * dx_min / max(vA, vB, 1e-9)
    # Runtime/performance guard for Streamlit screening model.
    return float(min(max(0.05, dt_cfl), 0.5))


def calculate_characteristic_time_constants(inp: ModelInputs) -> pd.DataFrame:
    pr = PREthylene(inp.pr_fluid)
    rows = []
    m_dot = inp.flowrate_kgph / 3600.0
    for sec in [inp.section_a, inp.section_b]:
        fluid_props = pr.props_mass_basis(inp.initial_superheater_outlet_temp_c, sec.pressure_bar_a)
        hi, vel, re, prn = internal_heat_transfer_coefficient(m_dot, fluid_props["rho"], fluid_props["cp"], fluid_props["mu"], fluid_props["k"], sec.geometry)
        ho, re_air, pr_air = external_heat_transfer_coefficient(inp.air_velocity_mps, sec.geometry.od_m, inp.air)
        uo = overall_uo(hi, ho, sec.geometry, sec.metal.k)
        m_m = calculate_pipe_metal_mass(sec.geometry, sec.length_m, sec.metal.density)
        c_m = m_m * sec.metal.cp
        tau_wall_ambient = c_m / max(ho * sec.geometry.area_o_per_m * sec.length_m, 1e-12)
        tau_wall_internal = c_m / max(hi * sec.geometry.area_i_per_m * sec.length_m, 1e-12)
        m_f = calculate_fluid_hold_up_mass(sec.geometry, sec.length_m, fluid_props["rho"])
        c_f = m_f * fluid_props["cp"]
        tau_fluid_internal = c_f / max(hi * sec.geometry.area_i_per_m * sec.length_m, 1e-12)
        residence = sec.length_m / max(vel, 1e-12)
        rows.append({
            "section": sec.name,
            "pressure_bar_a": sec.pressure_bar_a,
            "density_kg_m3": fluid_props["rho"],
            "cp_j_kgk": fluid_props["cp"],
            "viscosity_pa_s": fluid_props["mu"],
            "conductivity_w_mk": fluid_props["k"],
            "z_factor": fluid_props["z"],
            "velocity_mps": vel,
            "re": re,
            "pr": prn,
            "h_internal_W_m2K": hi,
            "h_external_W_m2K": ho,
            "Uo_W_m2K": uo,
            "fluid_residence_s": residence,
            "tau_fluid_to_wall_s": tau_fluid_internal,
            "tau_wall_to_fluid_s": tau_wall_internal,
            "tau_wall_to_ambient_s": tau_wall_ambient,
        })
    return pd.DataFrame(rows)


def init_sensors(inp: ModelInputs) -> List[Dict[str, float]]:
    if not inp.sensors:
        inp.sensors = [SensorConfig(f"TT{i+1}", tau_s=60.0) for i in range(3)]
    sensors = []
    t0 = inp.initial_superheater_outlet_temp_c
    for s in inp.sensors:
        val = s.frozen_value_c if s.frozen_value_c is not None else t0 + s.offset_c
        sensors.append({"tag": s.tag, "tau_s": s.tau_s, "offset_c": s.offset_c, "fail_mode": s.fail_mode, "value_c": val})
    return sensors


def update_sensor(sensor_state: Dict[str, float], true_temp_c: float, dt: float, initial_temp_c: float) -> float:
    mode = sensor_state["fail_mode"]
    tau = sensor_state["tau_s"]
    offset = sensor_state["offset_c"]
    y = sensor_state["value_c"]
    if mode == "stuck_high":
        return float(max(y, initial_temp_c + 50.0))
    if mode == "stuck_low":
        return float(min(y, initial_temp_c - 50.0))
    if mode == "frozen_initial":
        return float(y)
    target = true_temp_c + offset
    if tau > 1e-9:
        y = y + dt * (target - y) / tau
    else:
        y = target
    return float(y)


def voted_trip(sensor_values: List[float], setpoint_c: float, k: int) -> bool:
    count = sum(1 for x in sensor_values if x <= setpoint_c)
    return count >= k


def scenario_templates() -> List[Dict[str, object]]:
    # Extensive but not exhaustive scenario library; all remain editable by user.
    return [
        {"name": "Base_Bounding_Full_Duty_Loss", "description": "Immediate collapse of SH outlet to failure target; nominal protection.",
         "overrides": {"model_mode": "bounding", "failure_target_temp_c": -40.0}},
        {"name": "Base_Dynamic_Full_Duty_Loss", "description": "1st-order SH outlet decay after full heating medium failure.",
         "overrides": {"model_mode": "dynamic", "failure_target_temp_c": -40.0, "superheater_tau_s": 120.0}},
        {"name": "Partial_Duty_Loss_50pct", "description": "SH outlet decays to midpoint between normal and inlet temperature.",
         "overrides": {"model_mode": "dynamic", "failure_target_temp_c": -5.0, "superheater_tau_s": 180.0}},
        {"name": "Slow_Heat_Collapse", "description": "Slow exchanger response due to thermal inventory / shell-side holdup.",
         "overrides": {"model_mode": "dynamic", "failure_target_temp_c": -40.0, "superheater_tau_s": 300.0}},
        {"name": "Fast_Heat_Collapse", "description": "Very fast exchanger response; conservative than base dynamic.",
         "overrides": {"model_mode": "dynamic", "failure_target_temp_c": -40.0, "superheater_tau_s": 30.0}},
        {"name": "Trip_Setpoint_Degraded", "description": "Lower trip setpoint leaves less margin.",
         "overrides": {"tt_setpoint_c": -5.0}},
        {"name": "High_TT_Lag", "description": "Sensor lag doubled (common-cause degraded response).",
         "overrides": {"sensors": [{"tag":"TT1","tau_s":120.0},{"tag":"TT2","tau_s":120.0},{"tag":"TT3","tau_s":120.0}] }},
        {"name": "One_TT_Frozen_Normal", "description": "One of three sensors stuck at initial normal reading; 2oo3 still available.",
         "overrides": {"sensors": [{"tag":"TT1","tau_s":60.0,"fail_mode":"frozen_initial"},{"tag":"TT2","tau_s":60.0},{"tag":"TT3","tau_s":60.0}] }},
        {"name": "One_TT_Stuck_High", "description": "One TT fails high; 2oo3 logic still possible but delayed sensitivity.",
         "overrides": {"sensors": [{"tag":"TT1","tau_s":60.0,"fail_mode":"stuck_high"},{"tag":"TT2","tau_s":60.0},{"tag":"TT3","tau_s":60.0}] }},
        {"name": "One_TT_Stuck_Low_Spurious", "description": "One TT stuck low; checks robustness of 2oo3 against spurious trip.",
         "overrides": {"sensors": [{"tag":"TT1","tau_s":60.0,"fail_mode":"stuck_low"},{"tag":"TT2","tau_s":60.0},{"tag":"TT3","tau_s":60.0}] }},
        {"name": "Logic_Solver_Delay", "description": "Trip voting decision plus logic solver adds 5 s.",
         "overrides": {"logic_solver_delay_s": 5.0}},
        {"name": "Valve_Slow_Closure", "description": "Actuator / valve closes slower.",
         "overrides": {"xv_closure_time_s": 20.0}},
        {"name": "Valve_Deadtime_5s", "description": "Actuator deadtime before movement.",
         "overrides": {"xv_dead_time_s": 5.0}},
        {"name": "Valve_Fails_Partial_20pct_Open", "description": "Valve only closes to 20% open.",
         "overrides": {"valve_characteristic": "fail_partial", "valve_min_open_frac": 0.20}},
        {"name": "Valve_Stuck_Open", "description": "Valve never closes, worst protective failure case.",
         "overrides": {"valve_characteristic": "stuck_open"}},
        {"name": "Equal_Percentage_Like_Closure", "description": "Non-linear closure shape for sensitivity screening.",
         "overrides": {"valve_characteristic": "equal_percentage_like", "valve_exponent": 2.0}},
        {"name": "High_Ethylene_Flow", "description": "Increased throughput reduces residence time but increases convective transport.",
         "overrides": {"flowrate_kgph": 40000.0}},
        {"name": "Low_Ethylene_Flow", "description": "Reduced throughput changes transport and thermal inventory response.",
         "overrides": {"flowrate_kgph": 20000.0}},
        {"name": "Low_Ambient", "description": "Cooler ambient reduces heat gain to pipe.",
         "overrides": {"ambient_temp_c": 10.0}},
        {"name": "High_Ambient", "description": "Warmer ambient increases heat gain to pipe.",
         "overrides": {"ambient_temp_c": 45.0}},
        {"name": "High_Wind", "description": "Higher air velocity raises external h.",
         "overrides": {"air_velocity_mps": 5.0}},
        {"name": "Low_Wind", "description": "Quieter air reduces ambient heat transfer.",
         "overrides": {"air_velocity_mps": 0.2}},
        {"name": "Thin_CS_Wall", "description": "Less downstream CS thermal mass.",
         "overrides": {"section_b_wall_mm": 7.0}},
        {"name": "Thick_CS_Wall", "description": "Higher downstream CS thermal mass.",
         "overrides": {"section_b_wall_mm": 14.0}},
        {"name": "Low_Initial_SH_Outlet", "description": "Processor already operating closer to trip limit.",
         "overrides": {"initial_superheater_outlet_temp_c": 10.0}},
        {"name": "StartUp_Cold_Metal", "description": "Initial pipe metal temperature colder than fluid due to startup / upset.",
         "overrides": {"initial_pipe_metal_temp_c": 0.0}},
        {"name": "Delayed_Failure_Initiation", "description": "Heating-medium failure starts later in simulation.",
         "overrides": {"failure_start_time_s": 120.0}},
        {"name": "Pressure_Low_Downstream", "description": "Lower downstream absolute pressure changes PR density and transport characteristics.",
         "overrides": {"section_b_pressure_bar_a": 6.0}},
        {"name": "Pressure_High_Downstream", "description": "Higher downstream absolute pressure changes PR density and transport characteristics.",
         "overrides": {"section_b_pressure_bar_a": 20.0}},
        {"name": "Combined_Worst_Credible", "description": "Fast collapse + high TT lag + logic delay + slow valve + low ambient + thin CS wall.",
         "overrides": {
             "model_mode": "dynamic", "superheater_tau_s": 20.0, "failure_target_temp_c": -40.0,
             "sensors": [{"tag":"TT1","tau_s":120.0},{"tag":"TT2","tau_s":120.0},{"tag":"TT3","tau_s":120.0}],
             "logic_solver_delay_s": 5.0, "xv_dead_time_s": 2.0, "xv_closure_time_s": 20.0,
             "ambient_temp_c": 10.0, "section_b_wall_mm": 7.0
         }},
    ]


def deep_copy_inputs(inp: ModelInputs) -> ModelInputs:
    # Reconstruct dataclasses from asdict for scenario manager
    d = asdict(inp)
    pr_fluid = PREthyleneConfig(**d["pr_fluid"])
    gA = Geometry(**d["section_a"]["geometry"])
    gB = Geometry(**d["section_b"]["geometry"])
    mA = MaterialProps(**d["section_a"]["metal"])
    mB = MaterialProps(**d["section_b"]["metal"])
    sA = SectionInputs(name=d["section_a"]["name"], length_m=d["section_a"]["length_m"], geometry=gA, metal=mA,
                       n_cells=d["section_a"]["n_cells"], pressure_bar_a=d["section_a"]["pressure_bar_a"])
    sB = SectionInputs(name=d["section_b"]["name"], length_m=d["section_b"]["length_m"], geometry=gB, metal=mB,
                       n_cells=d["section_b"]["n_cells"], pressure_bar_a=d["section_b"]["pressure_bar_a"])
    sensors = [SensorConfig(**x) for x in d["sensors"]]
    mi = ModelInputs(
        superheater_inlet_temp_c=d["superheater_inlet_temp_c"],
        initial_superheater_outlet_temp_c=d["initial_superheater_outlet_temp_c"],
        failure_target_temp_c=d["failure_target_temp_c"],
        superheater_tau_s=d["superheater_tau_s"],
        model_mode=d["model_mode"],
        failure_start_time_s=d["failure_start_time_s"],
        ramp_duration_s=d["ramp_duration_s"],
        pr_fluid=pr_fluid,
        flowrate_kgph=d["flowrate_kgph"],
        section_a=sA,
        section_b=sB,
        ambient_temp_c=d["ambient_temp_c"],
        air_velocity_mps=d["air_velocity_mps"],
        air=d["air"],
        tt_setpoint_c=d["tt_setpoint_c"],
        mdmt_c=d["mdmt_c"],
        sensors=sensors,
        vote_k=d["vote_k"],
        vote_n=d["vote_n"],
        logic_solver_delay_s=d["logic_solver_delay_s"],
        xv_closure_time_s=d["xv_closure_time_s"],
        xv_dead_time_s=d["xv_dead_time_s"],
        valve_characteristic=d["valve_characteristic"],
        valve_exponent=d["valve_exponent"],
        valve_min_open_frac=d["valve_min_open_frac"],
        initial_pipe_metal_temp_c=d["initial_pipe_metal_temp_c"],
        initial_pipe_fluid_temp_c=d["initial_pipe_fluid_temp_c"],
        t_end_s=d["t_end_s"],
        output_dt_s=d["output_dt_s"],
        cfl_target=d["cfl_target"],
    )
    return mi


def apply_scenario(base_inp: ModelInputs, scenario: Dict[str, object]) -> ModelInputs:
    mi = deep_copy_inputs(base_inp)
    ov = scenario.get("overrides", {})
    for k, v in ov.items():
        if k == "section_a_wall_mm":
            g = calculate_geometry(mi.section_a.geometry.nps_in, wall_thickness_mm=float(v))
            mi.section_a = SectionInputs(mi.section_a.name, mi.section_a.length_m, g, mi.section_a.metal, mi.section_a.n_cells, mi.section_a.pressure_bar_a)
        elif k == "section_b_wall_mm":
            g = calculate_geometry(mi.section_b.geometry.nps_in, wall_thickness_mm=float(v))
            mi.section_b = SectionInputs(mi.section_b.name, mi.section_b.length_m, g, mi.section_b.metal, mi.section_b.n_cells, mi.section_b.pressure_bar_a)
        elif k == "section_a_pressure_bar_a":
            mi.section_a.pressure_bar_a = float(v)
        elif k == "section_b_pressure_bar_a":
            mi.section_b.pressure_bar_a = float(v)
        elif k == "sensors":
            sensors = []
            for i, sd in enumerate(v):
                tag = sd.get("tag", f"TT{i+1}")
                sensors.append(SensorConfig(tag=tag,
                                            tau_s=sd.get("tau_s", 60.0),
                                            offset_c=sd.get("offset_c", 0.0),
                                            fail_mode=sd.get("fail_mode", "healthy"),
                                            frozen_value_c=sd.get("frozen_value_c", None)))
            mi.sensors = sensors
        else:
            setattr(mi, k, v)
    return mi


def simulate_transient(inp: ModelInputs, enable_trip_and_closure: bool = True) -> Dict[str, object]:
    pr = PREthylene(inp.pr_fluid)
    grid = build_grid(inp.section_a, inp.section_b)
    dt = recommend_time_step(inp)
    n_steps = int(math.ceil(inp.t_end_s / dt)) + 1
    store_every = max(1, int(round(inp.output_dt_s / dt)))

    t0_f = inp.initial_pipe_fluid_temp_c if inp.initial_pipe_fluid_temp_c is not None else inp.initial_superheater_outlet_temp_c
    t0_w = inp.initial_pipe_metal_temp_c if inp.initial_pipe_metal_temp_c is not None else inp.initial_superheater_outlet_temp_c
    tf = np.full(len(grid), t0_f, dtype=float)
    tw = np.full(len(grid), t0_w, dtype=float)

    m_w = grid["vol_m"].to_numpy() * grid["metal_rho"].to_numpy()
    cp_w = grid["metal_cp"].to_numpy()
    ai = grid["ai"].to_numpy()
    ao = grid["ao"].to_numpy()
    vol_f = grid["vol_f"].to_numpy()
    pressures_bar = grid["pressure_bar_a"].to_numpy()
    section_idx = grid["section_idx"].to_numpy()
    pr_table_a = build_pr_property_table(pr, inp.section_a.pressure_bar_a)
    pr_table_b = build_pr_property_table(pr, inp.section_b.pressure_bar_a)
    is_cs = section_idx == 1
    a_end_idx = inp.section_a.n_cells - 1

    sensors = init_sensors(inp)
    trip_time = None
    true_cross_time = None
    closure_complete_time = None
    mdmt_time = None
    voted_state = False

    ts_records = []
    profile_records = []

    for step in range(n_steps):
        t = step * dt
        sh_out = superheater_outlet_temperature(t, inp)

        true_sensor_t = float(tf[a_end_idx])
        if true_cross_time is None and true_sensor_t <= inp.tt_setpoint_c:
            true_cross_time = t

        sensor_vals = []
        for i, s in enumerate(sensors):
            sensors[i]["value_c"] = update_sensor(s, true_sensor_t, dt, inp.initial_superheater_outlet_temp_c)
            sensor_vals.append(sensors[i]["value_c"])

        if enable_trip_and_closure:
            voted = voted_trip(sensor_vals, inp.tt_setpoint_c, inp.vote_k)
            if (not voted_state) and voted and trip_time is None:
                trip_time = t
            voted_state = voted
            opening = valve_open_fraction(t, trip_time, inp)
            t_start_move = None if trip_time is None else trip_time + inp.logic_solver_delay_s + inp.xv_dead_time_s
            if t_start_move is not None and closure_complete_time is None and opening <= inp.valve_min_open_frac + 1e-6 and inp.valve_characteristic != "stuck_open":
                closure_complete_time = t
        else:
            opening = 1.0

        m_dot = (inp.flowrate_kgph / 3600.0) * opening
        tf_old = tf.copy()
        tw_old = tw.copy()
        tf_new = tf.copy()
        tw_new = tw.copy()

        # Runtime optimization: update PR properties and h-values using section-average temperatures
        # at each time step (still pressure-dependent by section), then apply to all cells in section.
        sec_cache = {}
        for sec in [0, 1]:
            mask = section_idx == sec
            t_mean = float(np.mean(tf_old[mask]))
            geom = inp.section_a.geometry if sec == 0 else inp.section_b.geometry
            Pbar = inp.section_a.pressure_bar_a if sec == 0 else inp.section_b.pressure_bar_a
            props_sec = interp_pr_props(pr_table_a if sec == 0 else pr_table_b, t_mean)
            hi_sec, vel_sec, re_sec, pr_sec = internal_heat_transfer_coefficient(m_dot, props_sec["rho"], props_sec["cp"], props_sec["mu"], props_sec["k"], geom)
            ho_sec, _, _ = external_heat_transfer_coefficient(inp.air_velocity_mps, geom.od_m, inp.air)
            sec_cache[sec] = {"props": props_sec, "hi": hi_sec, "vel": vel_sec, "re": re_sec, "pr": pr_sec, "ho": ho_sec}

        for i in range(len(grid)):
            sec = int(section_idx[i])
            cache = sec_cache[sec]
            rho = cache["props"]["rho"]
            cp = cache["props"]["cp"]
            hi = cache["hi"]
            ho = cache["ho"]

            t_up = sh_out if i == 0 else tf_old[i-1]
            m_f = max(1e-9, vol_f[i] * rho)
            adv = m_dot * cp * (t_up - tf_old[i])
            q_int = hi * ai[i] * (tw_old[i] - tf_old[i])
            d_tf = (adv + q_int) / max(m_f * cp, 1e-9)

            q_wall = hi * ai[i] * (tf_old[i] - tw_old[i]) + ho * ao[i] * (inp.ambient_temp_c - tw_old[i])
            d_tw = q_wall / max(m_w[i] * cp_w[i], 1e-9)

            tf_new[i] = tf_old[i] + dt * d_tf
            tw_new[i] = tw_old[i] + dt * d_tw

        tf = tf_new
        tw = tw_new

        if mdmt_time is None and np.any(tw[is_cs] <= inp.mdmt_c):
            mdmt_time = t

        if step % store_every == 0 or step == n_steps - 1:
            min_cs_wall = float(np.min(tw[is_cs])) if np.any(is_cs) else np.nan
            min_cs_fluid = float(np.min(tf[is_cs])) if np.any(is_cs) else np.nan
            props_a = sec_cache[0]["props"]
            hi_a, vel_a, re_a, pr_a = sec_cache[0]["hi"], sec_cache[0]["vel"], sec_cache[0]["re"], sec_cache[0]["pr"]
            ts_records.append({
                "time_s": t,
                "superheater_outlet_temp_c": sh_out,
                "sensor_true_temp_c": true_sensor_t,
                **{f"sensor_{j+1}_temp_c": sensor_vals[j] for j in range(len(sensor_vals))},
                "voted_trip_active": int(voted_state) if enable_trip_and_closure else 0,
                "valve_open_frac": opening,
                "flowrate_kg_s": m_dot,
                "section_a_exit_density_kg_m3": props_a["rho"],
                "section_a_exit_cp_j_kgk": props_a["cp"],
                "section_a_exit_mu_pa_s": props_a["mu"],
                "section_a_exit_k_w_mk": props_a["k"],
                "section_a_exit_z": props_a["z"],
                "section_a_exit_hi_w_m2k": hi_a,
                "section_a_exit_velocity_mps": vel_a,
                "section_a_exit_re": re_a,
                "section_a_exit_pr": pr_a,
                "section_a_exit_fluid_temp_c": float(tf[a_end_idx]),
                "section_a_exit_wall_temp_c": float(tw[a_end_idx]),
                "min_cs_fluid_temp_c": min_cs_fluid,
                "min_cs_wall_temp_c": min_cs_wall,
            })
            for j in range(len(grid)):
                profile_records.append({
                    "time_s": t,
                    "x_m": float(grid.iloc[j]["x_mid"]),
                    "section": grid.iloc[j]["section"],
                    "pressure_bar_a": float(pressures_bar[j]),
                    "fluid_temp_c": float(tf[j]),
                    "wall_temp_c": float(tw[j]),
                })

        if enable_trip_and_closure:
            if closure_complete_time is not None and mdmt_time is not None and t > max(closure_complete_time, mdmt_time) + 5 * inp.output_dt_s:
                break
        else:
            if mdmt_time is not None and t > mdmt_time + 5 * inp.output_dt_s:
                break

    ts_df = pd.DataFrame(ts_records)
    profiles_df = pd.DataFrame(profile_records)
    char_df = calculate_characteristic_time_constants(inp)

    pr_a_init = interp_pr_props(pr_table_a, inp.initial_superheater_outlet_temp_c)
    pr_b_init = interp_pr_props(pr_table_b, inp.initial_superheater_outlet_temp_c)
    summary = {
        "dt_s": dt,
        "time_to_true_tt_threshold_s": true_cross_time,
        "trip_time_s": trip_time,
        "instrument_lag_realized_s": None if (trip_time is None or true_cross_time is None) else max(0.0, trip_time - true_cross_time),
        "logic_solver_delay_s": inp.logic_solver_delay_s if enable_trip_and_closure else None,
        "xv_dead_time_s": inp.xv_dead_time_s if enable_trip_and_closure else None,
        "closure_complete_time_s": closure_complete_time,
        "first_mdmt_time_s": mdmt_time,
        "protection_status": ("NOT_APPLICABLE_UNPROTECTED" if not enable_trip_and_closure else ("UNKNOWN" if closure_complete_time is None or mdmt_time is None else ("ADEQUATE (screening)" if closure_complete_time < mdmt_time else "INADEQUATE (screening)"))),
        "section_a_fluid_volume_m3": inp.section_a.geometry.vol_fluid_per_m3_per_m * inp.section_a.length_m,
        "section_b_fluid_volume_m3": inp.section_b.geometry.vol_fluid_per_m3_per_m * inp.section_b.length_m,
        "section_a_fluid_hold_up_kg": calculate_fluid_hold_up_mass(inp.section_a.geometry, inp.section_a.length_m, pr_a_init["rho"]),
        "section_b_fluid_hold_up_kg": calculate_fluid_hold_up_mass(inp.section_b.geometry, inp.section_b.length_m, pr_b_init["rho"]),
        "section_a_metal_mass_kg": calculate_pipe_metal_mass(inp.section_a.geometry, inp.section_a.length_m, inp.section_a.metal.density),
        "section_b_metal_mass_kg": calculate_pipe_metal_mass(inp.section_b.geometry, inp.section_b.length_m, inp.section_b.metal.density),
        "base_flowrate_kg_s": inp.flowrate_kgph / 3600.0,
        "note": "Protected case should be compared against unprotected available PST. Protected MDMT may never be reached within the simulated horizon.",
    }
    return {
        "inputs": inp,
        "timeseries": ts_df,
        "profiles": profiles_df,
        "grid": grid,
        "characteristics": char_df,
        "summary": summary,
    }


def calculate_pst(inp: ModelInputs, protected_results: Dict[str, object], unprotected_results: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    if unprotected_results is None:
        unprotected_results = simulate_transient(inp, enable_trip_and_closure=False)
    sp = protected_results["summary"]
    su = unprotected_results["summary"]
    rows = [
        {"item": "Available PST (unprotected initiating event to first CS wall reaching MDMT)", "value_s": su.get("first_mdmt_time_s")},
        {"item": "Time to process trip threshold at TT location (true temperature)", "value_s": sp.get("time_to_true_tt_threshold_s")},
        {"item": "Instrument / sensor lag realized until trip", "value_s": sp.get("instrument_lag_realized_s")},
        {"item": "Logic solver delay", "value_s": sp.get("logic_solver_delay_s")},
        {"item": "Valve actuation deadtime", "value_s": sp.get("xv_dead_time_s")},
        {"item": "Trip signal time", "value_s": sp.get("trip_time_s")},
        {"item": "Valve stroke time to fully closed", "value_s": None if sp.get("trip_time_s") is None or sp.get("closure_complete_time_s") is None else sp.get("closure_complete_time_s") - sp.get("trip_time_s")},
        {"item": "Protective action complete (valve fully closed)", "value_s": sp.get("closure_complete_time_s")},
        {"item": "Time until first CS wall segment reaches MDMT with protection active", "value_s": sp.get("first_mdmt_time_s")},
        {"item": "Safety margin vs unprotected PST (available PST - full closure time)", "value_s": None if su.get("first_mdmt_time_s") is None or sp.get("closure_complete_time_s") is None else su.get("first_mdmt_time_s") - sp.get("closure_complete_time_s")},
        {"item": "Safety margin vs protected MDMT time (if reached)", "value_s": None if sp.get("first_mdmt_time_s") is None or sp.get("closure_complete_time_s") is None else sp.get("first_mdmt_time_s") - sp.get("closure_complete_time_s")},
    ]
    return pd.DataFrame(rows)


def summarize_scenario_result(inp: ModelInputs, protected_results: Dict[str, object], unprotected_results: Optional[Dict[str, object]] = None, scenario_name: str = "Base") -> Dict[str, object]:
    if unprotected_results is None:
        unprotected_results = simulate_transient(inp, enable_trip_and_closure=False)
    sp = protected_results["summary"]
    su = unprotected_results["summary"]
    return {
        "scenario": scenario_name,
        "available_pst_unprotected_s": su.get("first_mdmt_time_s"),
        "true_tt_threshold_s": sp.get("time_to_true_tt_threshold_s"),
        "trip_time_s": sp.get("trip_time_s"),
        "closure_complete_s": sp.get("closure_complete_time_s"),
        "mdmt_time_protected_s": sp.get("first_mdmt_time_s"),
        "margin_vs_unprotected_s": None if su.get("first_mdmt_time_s") is None or sp.get("closure_complete_time_s") is None else su.get("first_mdmt_time_s") - sp.get("closure_complete_time_s"),
        "status": ("ADEQUATE (screening)" if su.get("first_mdmt_time_s") is not None and sp.get("closure_complete_time_s") is not None and su.get("first_mdmt_time_s") > sp.get("closure_complete_time_s") else "CHECK / UNKNOWN"),
    }


def run_scenario_manager(base_inp: ModelInputs, selected_scenarios: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    library = scenario_templates()
    if selected_scenarios:
        library = [s for s in library if s["name"] in selected_scenarios]
    rows = []
    artifacts = {}
    for sc in library:
        mi = apply_scenario(base_inp, sc)
        res_p = simulate_transient(mi, enable_trip_and_closure=True)
        res_u = simulate_transient(mi, enable_trip_and_closure=False)
        rows.append(summarize_scenario_result(mi, res_p, res_u, sc["name"]))
        artifacts[sc["name"]] = {"inputs": mi, "protected": res_p, "unprotected": res_u, "description": sc.get("description", "")}
    df = pd.DataFrame(rows)
    return df, artifacts


def validation_checks(inp: ModelInputs, results: Dict[str, object]) -> pd.DataFrame:
    s = results["summary"]
    char = results["characteristics"]
    checks = []
    checks.append({"check": "Time step positive and <= 1 s", "value": s["dt_s"], "criterion": "0 < dt <= 1", "status": "PASS" if 0 < s["dt_s"] <= 1 else "CHECK"})
    for _, row in char.iterrows():
        checks.append({"check": f"{row['section']} PR Z-factor positive", "value": row['z_factor'], "criterion": "> 0", "status": "PASS" if row['z_factor'] > 0 else "CHECK"})
        checks.append({"check": f"{row['section']} internal h plausible", "value": row['h_internal_W_m2K'], "criterion": "> 1 W/m2.K", "status": "PASS" if row['h_internal_W_m2K'] > 1.0 else "CHECK"})
        checks.append({"check": f"{row['section']} external h plausible", "value": row['h_external_W_m2K'], "criterion": "> 2 W/m2.K", "status": "PASS" if row['h_external_W_m2K'] > 2.0 else "CHECK"})
    checks.append({"check": "Protected summary available", "value": s['protection_status'], "criterion": "text", "status": "PASS" if isinstance(s['protection_status'], str) else "CHECK"})
    return pd.DataFrame(checks)


def generate_sample_inputs(model_mode: str = "bounding") -> ModelInputs:
    ss = MaterialProps(density=8000.0, cp=500.0, k=16.0)
    cs = MaterialProps(density=7850.0, cp=470.0, k=45.0)
    g_a = calculate_geometry(10, schedule="40")
    g_b = calculate_geometry(12, schedule="40")
    sensors = [SensorConfig("TT1", 60.0), SensorConfig("TT2", 60.0), SensorConfig("TT3", 60.0)]
    pr_cfg = PREthyleneConfig(
        cp_ideal_ref_jkgk=2200.0,
        cp_ideal_ref_temp_k=273.15,
        cp_ideal_slope_jkgk2=0.0,
        mu_ref_pa_s=9e-6,
        k_ref_w_mk=0.020,
    )
    return ModelInputs(
        superheater_inlet_temp_c=-40.0,
        initial_superheater_outlet_temp_c=30.0,
        failure_target_temp_c=-40.0,
        superheater_tau_s=120.0,
        model_mode=model_mode,
        failure_start_time_s=0.0,
        ramp_duration_s=60.0,
        pr_fluid=pr_cfg,
        flowrate_kgph=31600.0,
        section_a=SectionInputs("A: SH outlet to XV inlet (SS)", 10.0, g_a, ss, n_cells=10, pressure_bar_a=10.0),
        section_b=SectionInputs("B: XV outlet through CS line (CS)", 500.0, g_b, cs, n_cells=50, pressure_bar_a=10.0),
        ambient_temp_c=30.0,
        air_velocity_mps=1.5,
        air=dict(AIR_DEFAULTS),
        tt_setpoint_c=0.0,
        mdmt_c=-29.0,
        sensors=sensors,
        vote_k=2,
        vote_n=3,
        logic_solver_delay_s=0.0,
        xv_closure_time_s=10.0,
        xv_dead_time_s=0.0,
        valve_characteristic="linear",
        valve_exponent=1.0,
        valve_min_open_frac=0.0,
        initial_pipe_metal_temp_c=None,
        initial_pipe_fluid_temp_c=None,
        t_end_s=1800.0,
        output_dt_s=1.0,
        cfl_target=0.3,
    )


def sample_inputs_json() -> str:
    inp = generate_sample_inputs()
    d = asdict(inp)
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    inp = generate_sample_inputs(model_mode="bounding")
    res_p = simulate_transient(inp, enable_trip_and_closure=True)
    res_u = simulate_transient(inp, enable_trip_and_closure=False)
    print(pd.DataFrame([summarize_scenario_result(inp, res_p, res_u, 'Base')]))
