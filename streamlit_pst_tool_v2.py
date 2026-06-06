
"""
streamlit_pst_tool_v2.py
Streamlit UI for PR-EOS PST model, scenario manager, and automated report export.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pst_model_v2 import (
    NPS_DB, AIR_DEFAULTS, PREthyleneConfig, MaterialProps, SectionInputs, SensorConfig,
    ModelInputs, calculate_geometry, generate_sample_inputs, simulate_transient,
    calculate_pst, calculate_characteristic_time_constants, validation_checks,
    run_scenario_manager, scenario_templates
)
from report_export_v2 import export_excel_report, export_word_report, export_pdf_report


def plot_results(results, selected_times_s=None):
    ts = results["timeseries"]
    pf = results["profiles"]
    if selected_times_s is None:
        tmax = float(ts["time_s"].max())
        selected_times_s = [0.0, min(60.0, tmax), min(120.0, tmax), min(300.0, tmax)]
    times_available = ts["time_s"].to_numpy()
    chosen = []
    for t_sel in selected_times_s:
        idx = int(abs(times_available - t_sel).argmin())
        chosen.append(float(times_available[idx]))
    chosen = sorted(set(chosen))

    figs = {}
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=ts["time_s"], y=ts["superheater_outlet_temp_c"], name="SH outlet BC"))
    f1.add_trace(go.Scatter(x=ts["time_s"], y=ts["section_a_exit_fluid_temp_c"], name="TT location fluid"))
    f1.add_trace(go.Scatter(x=ts["time_s"], y=ts["min_cs_fluid_temp_c"], name="Min CS fluid"))
    f1.update_layout(title="Fluid temperature vs time", xaxis_title="Time (s)", yaxis_title="Temperature (°C)")
    figs["fluid_vs_time"] = f1

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=ts["time_s"], y=ts["section_a_exit_wall_temp_c"], name="Section A exit wall"))
    f2.add_trace(go.Scatter(x=ts["time_s"], y=ts["min_cs_wall_temp_c"], name="Min CS wall"))
    f2.update_layout(title="Metal temperature vs time", xaxis_title="Time (s)", yaxis_title="Temperature (°C)")
    figs["metal_vs_time"] = f2

    f3 = go.Figure()
    for t in chosen:
        d = pf[pf["time_s"].round(9) == round(t,9)]
        f3.add_trace(go.Scatter(x=d["x_m"], y=d["fluid_temp_c"], mode="lines", name=f"t={t:.0f}s"))
    f3.update_layout(title="Fluid temperature vs distance", xaxis_title="Distance from SH outlet (m)", yaxis_title="Fluid temperature (°C)")
    figs["fluid_vs_distance"] = f3

    f4 = go.Figure()
    for t in chosen:
        d = pf[pf["time_s"].round(9) == round(t,9)]
        f4.add_trace(go.Scatter(x=d["x_m"], y=d["wall_temp_c"], mode="lines", name=f"t={t:.0f}s"))
    f4.update_layout(title="Wall temperature vs distance", xaxis_title="Distance from SH outlet (m)", yaxis_title="Wall temperature (°C)")
    figs["wall_vs_distance"] = f4

    f5 = go.Figure()
    f5.add_trace(go.Scatter(x=ts["time_s"], y=100*ts["valve_open_frac"], name="Valve open (%)"))
    f5.update_layout(title="Valve position vs time", xaxis_title="Time (s)", yaxis_title="Opening (%)")
    figs["valve_vs_time"] = f5

    f6 = go.Figure()
    f6.add_trace(go.Scatter(x=ts["time_s"], y=ts["flowrate_kg_s"], name="Flowrate (kg/s)"))
    f6.update_layout(title="Flowrate vs time", xaxis_title="Time (s)", yaxis_title="Flowrate (kg/s)")
    figs["flow_vs_time"] = f6
    return figs


def build_input_from_sidebar():
    with st.sidebar:
        st.header("Screening disclaimer")
        st.warning("Not for final design without validation against plant data, detailed design, approved valve data, and approved engineering methods.")
        model_mode = st.selectbox("Superheater failure model", ["bounding", "dynamic", "ramp"], index=0)
        sh_in = st.number_input("Superheater inlet temperature (°C)", value=-40.0)
        sh_out = st.number_input("Initial SH outlet temperature (°C)", value=30.0)
        fail_target = st.number_input("Failure target outlet temperature (°C)", value=-40.0)
        tau_hex = st.number_input("SH dynamic tau (s)", min_value=1.0, value=120.0)
        fail_start = st.number_input("Failure start time (s)", min_value=0.0, value=0.0)
        ramp_dur = st.number_input("Ramp duration if ramp model (s)", min_value=1.0, value=60.0)
        flowrate = st.number_input("Ethylene flowrate (kg/h)", min_value=0.0, value=31600.0)

        st.header("Pressure-dependent PR EOS inputs")
        p_a = st.number_input("Section A pressure (bar abs)", min_value=1.0, value=10.0)
        p_b = st.number_input("Section B pressure (bar abs)", min_value=1.0, value=10.0)
        cp_ref = st.number_input("Ideal-gas Cp ref (J/kg.K)", min_value=100.0, value=2200.0)
        cp_ref_t = st.number_input("Ideal-gas Cp ref temp (K)", min_value=100.0, value=273.15)
        cp_slope = st.number_input("Ideal-gas Cp slope (J/kg.K²)", value=0.0)
        mu_ref = st.number_input("Viscosity ref (Pa.s)", min_value=1e-8, value=9.0e-6, format="%.8f")
        mu_exp = st.number_input("Viscosity exponent vs T", value=0.7)
        k_ref = st.number_input("Thermal conductivity ref (W/m.K)", min_value=1e-4, value=0.020, format="%.5f")
        k_exp = st.number_input("Thermal conductivity exponent vs T", value=0.7)

        st.header("Section A (SS)")
        nps_a = st.selectbox("Section A NPS (inch)", list(NPS_DB.keys()), index=list(NPS_DB.keys()).index(10))
        sch_a = st.selectbox("Section A schedule", list(NPS_DB[nps_a]["schedules_mm"].keys()), index=list(NPS_DB[nps_a]["schedules_mm"].keys()).index("40") if "40" in NPS_DB[nps_a]["schedules_mm"] else 0)
        len_a = st.number_input("Section A length (m)", min_value=0.1, value=10.0)
        cells_a = st.number_input("Section A cells", min_value=2, value=10)

        st.header("Section B (CS)")
        nps_b = st.selectbox("Section B NPS (inch)", list(NPS_DB.keys()), index=list(NPS_DB.keys()).index(12))
        sch_b = st.selectbox("Section B schedule", list(NPS_DB[nps_b]["schedules_mm"].keys()), index=list(NPS_DB[nps_b]["schedules_mm"].keys()).index("40") if "40" in NPS_DB[nps_b]["schedules_mm"] else 0)
        len_b = st.number_input("Section B length (m)", min_value=1.0, value=500.0)
        cells_b = st.number_input("Section B cells", min_value=5, value=50)

        st.header("Metal properties")
        rho_ss = st.number_input("SS density (kg/m³)", value=8000.0)
        cp_ss = st.number_input("SS Cp (J/kg.K)", value=500.0)
        k_ss = st.number_input("SS conductivity (W/m.K)", value=16.0)
        rho_cs = st.number_input("CS density (kg/m³)", value=7850.0)
        cp_cs = st.number_input("CS Cp (J/kg.K)", value=470.0)
        k_cs = st.number_input("CS conductivity (W/m.K)", value=45.0)

        st.header("Protection / environment")
        t_amb = st.number_input("Ambient temperature (°C)", value=30.0)
        v_air = st.number_input("Air velocity (m/s)", min_value=0.0, value=1.5)
        tt_sp = st.number_input("Trip setpoint (°C)", value=0.0)
        mdmt = st.number_input("CS MDMT (°C)", value=-29.0)
        vote_k = st.number_input("Voting logic k (kooN)", min_value=1, value=2)
        vote_n = st.number_input("Voting logic N", min_value=1, value=3)
        logic_delay = st.number_input("Logic solver delay (s)", min_value=0.0, value=0.0)
        xv_stroke = st.number_input("XV closure time (s)", min_value=0.0, value=10.0)
        xv_dead = st.number_input("XV deadtime (s)", min_value=0.0, value=0.0)
        valve_char = st.selectbox("Valve characteristic", ["linear", "equal_percentage_like", "fail_partial", "stuck_open"], index=0)
        valve_exp = st.number_input("Valve exponent (non-linear option)", min_value=0.1, value=1.0)
        valve_min = st.number_input("Valve minimum open fraction", min_value=0.0, max_value=1.0, value=0.0)

        st.header("Sensors (2oo3 or user-configurable)")
        tau1 = st.number_input("TT1 lag (s)", min_value=0.0, value=60.0)
        tau2 = st.number_input("TT2 lag (s)", min_value=0.0, value=60.0)
        tau3 = st.number_input("TT3 lag (s)", min_value=0.0, value=60.0)
        fmodes = ["healthy", "frozen_initial", "stuck_high", "stuck_low"]
        f1 = st.selectbox("TT1 fail mode", fmodes, index=0)
        f2 = st.selectbox("TT2 fail mode", fmodes, index=0)
        f3 = st.selectbox("TT3 fail mode", fmodes, index=0)

        st.header("Numerics")
        t_end = st.number_input("Simulation end time (s)", min_value=60.0, value=1800.0)
        output_dt = st.number_input("Output interval (s)", min_value=0.1, value=1.0)
        cfl = st.number_input("CFL target", min_value=0.05, max_value=0.8, value=0.3)

    g_a = calculate_geometry(nps_a, schedule=sch_a)
    g_b = calculate_geometry(nps_b, schedule=sch_b)
    ss = MaterialProps(rho_ss, cp_ss, k_ss)
    cs = MaterialProps(rho_cs, cp_cs, k_cs)
    prc = PREthyleneConfig(cp_ideal_ref_jkgk=cp_ref, cp_ideal_ref_temp_k=cp_ref_t, cp_ideal_slope_jkgk2=cp_slope,
                           mu_ref_pa_s=mu_ref, mu_exp=mu_exp, k_ref_w_mk=k_ref, k_exp=k_exp)
    sensors = [
        SensorConfig("TT1", tau1, fail_mode=f1),
        SensorConfig("TT2", tau2, fail_mode=f2),
        SensorConfig("TT3", tau3, fail_mode=f3),
    ]
    inp = ModelInputs(
        superheater_inlet_temp_c=sh_in,
        initial_superheater_outlet_temp_c=sh_out,
        failure_target_temp_c=fail_target,
        superheater_tau_s=tau_hex,
        model_mode=model_mode,
        failure_start_time_s=fail_start,
        ramp_duration_s=ramp_dur,
        pr_fluid=prc,
        flowrate_kgph=flowrate,
        section_a=SectionInputs("A: SH outlet to XV inlet (SS)", len_a, g_a, ss, int(cells_a), pressure_bar_a=p_a),
        section_b=SectionInputs("B: XV outlet through CS line (CS)", len_b, g_b, cs, int(cells_b), pressure_bar_a=p_b),
        ambient_temp_c=t_amb,
        air_velocity_mps=v_air,
        air=dict(AIR_DEFAULTS),
        tt_setpoint_c=tt_sp,
        mdmt_c=mdmt,
        sensors=sensors,
        vote_k=int(vote_k),
        vote_n=int(vote_n),
        logic_solver_delay_s=logic_delay,
        xv_closure_time_s=xv_stroke,
        xv_dead_time_s=xv_dead,
        valve_characteristic=valve_char,
        valve_exponent=valve_exp,
        valve_min_open_frac=valve_min,
        t_end_s=t_end,
        output_dt_s=output_dt,
        cfl_target=cfl,
    )
    return inp


def main():
    st.set_page_config(page_title="PST Screening Tool v2", layout="wide")
    st.title("LP Ethylene Superheater PST Screening Tool v2")
    st.caption("Pressure-dependent PR EOS + scenario manager + PDF/Excel/Word export")

    inp = build_input_from_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Profiles / Plots", "Scenario Manager", "Validation", "Reports"])
    results_p = simulate_transient(inp, enable_trip_and_closure=True)
    results_u = simulate_transient(inp, enable_trip_and_closure=False)
    pst_df = calculate_pst(inp, results_p, results_u)
    figs = plot_results(results_p)
    char = results_p["characteristics"]
    ts = results_p["timeseries"]
    summary_u = results_u["summary"]
    summary_p = results_p["summary"]

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Available PST unprotected (s)", f"{summary_u['first_mdmt_time_s']:.1f}" if summary_u['first_mdmt_time_s'] is not None else f"> {inp.t_end_s:.0f}")
        c2.metric("Trip time (s)", f"{summary_p['trip_time_s']:.1f}" if summary_p['trip_time_s'] is not None else "N/A")
        c3.metric("Protective action complete (s)", f"{summary_p['closure_complete_time_s']:.1f}" if summary_p['closure_complete_time_s'] is not None else "N/A")
        margin = None if summary_u['first_mdmt_time_s'] is None or summary_p['closure_complete_time_s'] is None else summary_u['first_mdmt_time_s'] - summary_p['closure_complete_time_s']
        c4.metric("Margin vs unprotected PST (s)", f"{margin:.1f}" if margin is not None else "N/A")
        st.dataframe(pst_df, use_container_width=True)
        st.subheader("Characteristic values")
        st.dataframe(char, use_container_width=True)
        st.subheader("Protected / unprotected raw summaries")
        st.json({"protected": summary_p, "unprotected": summary_u})

    with tab2:
        for k in ["fluid_vs_time", "metal_vs_time", "fluid_vs_distance", "wall_vs_distance", "valve_vs_time", "flow_vs_time"]:
            st.plotly_chart(figs[k], use_container_width=True)
        with st.expander("Protected timeseries preview"):
            st.dataframe(ts.head(200), use_container_width=True)

    with tab3:
        lib = scenario_templates()
        choices = [x["name"] for x in lib]
        selected = st.multiselect("Select scenarios to run", choices, default=[choices[0], choices[1], choices[-1]])
        if st.button("Run selected scenarios"):
            df_s, artifacts = run_scenario_manager(inp, selected_scenarios=selected)
            st.dataframe(df_s, use_container_width=True)
            if not df_s.empty:
                fig = px.bar(df_s, x="scenario", y="margin_vs_unprotected_s", color="status", title="Scenario margin vs unprotected PST")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                st.session_state["scenario_df"] = df_s

    with tab4:
        st.dataframe(validation_checks(inp, results_p), use_container_width=True)
        st.markdown("### Key limitations / not for final design without validation")
        st.markdown("""
- PR EOS is used for density and non-ideal Cp screening adjustment for pure ethylene.
- Transport properties remain simplified screening approximations unless you extend the correlations.
- Exchanger dynamics are still bounding, dynamic first-order, or ramped screening models.
- Pressure drop, phase change, axial conduction, fittings/supports/valve-body thermal masses, and radiation are omitted.
- Use plant data and approved methods before any safety-credit or design decision.
""")

    with tab5:
        outprefix = st.text_input("Report file prefix", value="pst_screening_report")
        scenario_df = st.session_state.get("scenario_df", pd.DataFrame())
        if st.button("Generate Excel / Word / PDF reports"):
            xlsx = export_excel_report(f"{outprefix}.xlsx", results_p, results_u, scenario_df)
            docx = export_word_report(f"{outprefix}.docx", results_p, results_u, scenario_df)
            pdf = export_pdf_report(f"{outprefix}.pdf", results_p, results_u, scenario_df)
            for fp in [xlsx, docx, pdf]:
                with open(fp, "rb") as f:
                    st.download_button(label=f"Download {Path(fp).name}", data=f.read(), file_name=Path(fp).name)


if __name__ == "__main__":
    main()
