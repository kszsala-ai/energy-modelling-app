import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Environmental-Energy Decision Laboratory", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: #F7F9FC; color: #1F2937;}
    .block-container {padding-top: 1.0rem; max-width: 1400px;}
    h1,h2,h3 {color: #1E293B !important;}
    .lab-card {background: #FFFFFF; border: 1px solid #D1D5DB; border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.7rem;}
    .soft {color: #475569;}
    .risk-box {padding: 0.8rem 1rem; border-radius: 10px; border-left: 6px solid; background:#F8FAFC;}
    .stTabs [data-baseweb="tab-list"] {gap: 6px; background:#E5E7EB; padding:6px; border-radius:10px;}
    .stTabs [data-baseweb="tab"] {height:44px; padding:0 14px; border-radius:8px; font-weight:600; color:#334155;}
    .stTabs [aria-selected="true"] {background:#DBEAFE !important; color:#1D4ED8 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


def heating_from_temperature(temp_c: float) -> float:
    return max(5.0, min(100.0, 100 - ((temp_c + 15) * 2.8)))


def calculate_heating_intensity(temperature: float) -> float:
    return heating_from_temperature(float(temperature))


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


def get_nearest_scenarios(data: pd.DataFrame, t: float, w: float, tr: float, r: float, top_n: int = 50) -> pd.DataFrame:
    scenario_heating = calculate_heating_intensity(t)
    dist = (
        (data["temperature"].sub(t).abs() / 40.0)
        + (data["wind_speed"].sub(w).abs() / 12.0)
        + (data["traffic_intensity"].sub(tr).abs() / 100.0)
        + (data["renewable_share"].sub(r).abs() / 60.0)
        + (data["heating_intensity"].sub(scenario_heating).abs() / 100.0)
    )
    out = data.copy()
    out["similarity_distance"] = dist
    return out.nsmallest(min(top_n, len(out)), "similarity_distance").copy()


def risk_style(risk: str):
    mapper = {
        "low": ("#DCFCE7", "#166534"),
        "moderate": ("#FEF3C7", "#92400E"),
        "high": ("#FFEDD5", "#9A3412"),
        "alarm": ("#FEE2E2", "#991B1B"),
    }
    return mapper.get(str(risk).lower(), ("#E2E8F0", "#334155"))


DATA_FILE = "smog_energy_dataset.csv"
required_columns = [
    "date", "year", "month", "temperature", "wind_speed", "heating_intensity",
    "traffic_intensity", "renewable_share", "energy_demand", "local_emission",
    "dispersion_factor", "PM10", "CO2_emission", "smog_risk"
]
df = load_data(DATA_FILE)
missing = [c for c in required_columns if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(missing)}")
    st.stop()

predefined = {
    "Custom": {"temperature": 0, "wind_speed": 3.0, "traffic_intensity": 50, "renewable_share": 25},
    "Winter smog episode": {"temperature": -8, "wind_speed": 1.2, "traffic_intensity": 78, "renewable_share": 12},
    "Summer clean air": {"temperature": 22, "wind_speed": 6.5, "traffic_intensity": 35, "renewable_share": 35},
    "Energy transition": {"temperature": 6, "wind_speed": 4.0, "traffic_intensity": 45, "renewable_share": 55},
    "High traffic urban day": {"temperature": 4, "wind_speed": 2.0, "traffic_intensity": 92, "renewable_share": 20},
}

st.sidebar.markdown("## Scenario Controls")
scenario_name = st.sidebar.selectbox("Scenario mode", list(predefined.keys()))
defs = predefined[scenario_name]
temperature = st.sidebar.slider("Outdoor temperature [°C]", -15, 25, int(defs["temperature"]))
wind_speed = st.sidebar.slider("Wind speed [m/s]", 0.0, 12.0, float(defs["wind_speed"]), 0.1)
traffic_intensity = st.sidebar.slider("Traffic intensity [%]", 0, 100, int(defs["traffic_intensity"]))
renewable_share = st.sidebar.slider("Renewable energy share [%]", 0, 60, int(defs["renewable_share"]))
calculated_heating = calculate_heating_intensity(float(temperature))
st.sidebar.metric("Calculated heating intensity [%]", f"{calculated_heating:.1f}")
run = st.sidebar.button("Analyze Decision Scenario", type="primary")
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if run:
    st.session_state.run_analysis = True

filtered_df = get_nearest_scenarios(df, float(temperature), float(wind_speed), float(traffic_intensity), float(renewable_share), 50) if st.session_state.run_analysis else pd.DataFrame(columns=df.columns)

st.title("Environmental-Energy Teaching Laboratory")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introduction", "Dataset & Variables", "Environmental Relationships", "Decision Laboratory", "Model Interpretation", "Student Tasks"])

with tab1:
    st.markdown("### Educational Environmental-Energy System Simulator")
    st.markdown(
        """
        <div class='lab-card'>
        <b>Project description.</b><br>
        This application presents a simplified environmental-energy system simulator designed for educational purposes.
        The system demonstrates how meteorological conditions, heating demand, traffic intensity, and renewable energy
        transition may influence air quality and energy-system behavior.
        It operates on a synthetic dataset and uses nearest-scenario estimation instead of deterministic forecasting.
        The goal is to support systems thinking and environmental-engineering reasoning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset Characteristics")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            """
            <div class='lab-card'>
            <ul>
              <li>Synthetic environmental-energy dataset created for educational use.</li>
              <li>Period: <b>2021–2023</b>.</li>
              <li>Temporal resolution: <b>daily</b> (1 record per day).</li>
              <li>Simplified representation of environmental-energy interactions.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class='lab-card'>
            <b>The dataset simulates:</b>
            <ul>
              <li>seasonal temperature variability,</li>
              <li>heating demand changes,</li>
              <li>urban pollution episodes,</li>
              <li>renewable energy transition,</li>
              <li>air-quality responses.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    var_table = pd.DataFrame([
        ["Outdoor temperature", "°C", "Ambient atmospheric temperature affecting heating demand."],
        ["Wind speed", "m/s", "Atmospheric mixing and pollutant dispersion conditions."],
        ["Traffic intensity", "%", "Transport-related emission pressure in the urban system."],
        ["Renewable energy share", "%", "Estimated contribution of low-emission energy sources."],
        ["Heating intensity", "%", "Calculated thermal-demand pressure linked to outdoor temperature."],
        ["PM10 concentration", "µg/m³", "Estimated airborne particulate matter concentration."],
        ["CO₂ emission index", "-", "Simplified greenhouse-gas emission pressure indicator."],
        ["Energy demand index", "0–100", "Simplified total energy-system demand level."],
        ["Smog risk category", "-", "Categorical air-quality risk level (low/moderate/high/alarm)."],
    ], columns=["Variable", "Unit", "Meaning"])

    st.markdown("### Variable guide")
    st.dataframe(var_table, use_container_width=True, hide_index=True)

    st.markdown("### How does the system respond?")
    st.markdown(
        """
        <div class='lab-card'>
        <b>STEP 1</b> Environmental conditions are selected (weather, traffic, renewable-energy conditions)<br>
        ↓<br>
        <b>STEP 2</b> The system estimates heating demand<br>
        ↓<br>
        <b>STEP 3</b> The application identifies the most similar historical/synthetic scenarios<br>
        ↓<br>
        <b>STEP 4</b> Estimated environmental response is generated:
        PM10 concentration, CO₂ emissions, energy demand, smog risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why similar scenarios instead of exact predictions?")
    st.info(
        "Real environmental systems operate under uncertainty. Predictive systems rarely know exact future values. "
        "Many environmental-AI tools search for similar historical situations and estimate a typical response. "
        "This laboratory follows the same idea: it estimates environmental response from nearest scenarios rather than deterministic forecasts."
    )

with tab2:
    st.markdown("### Variables")
    a,b,c = st.columns(3)
    a.markdown("**Inputs:** temperature, wind, traffic, renewable share")
    b.markdown("**Model:** heating intensity, local emission, dispersion")
    c.markdown("**Outputs:** PM10, CO₂ index, demand, smog risk")
    st.dataframe(df.head(10), use_container_width=True)
    st.dataframe(df[["temperature","wind_speed","traffic_intensity","renewable_share","PM10","CO2_emission","energy_demand"]].describe().round(2), use_container_width=True)

with tab3:
    st.markdown("### Exploratory plotting laboratory")
    plot_type = st.selectbox("Plot type", ["Scatter", "Boxplot", "Histogram", "Time series"], key="plot_type")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x_var = st.selectbox("X variable", numeric_cols + ["date"], key="x_var")
    y_var = st.selectbox("Y variable", numeric_cols, index=numeric_cols.index("PM10") if "PM10" in numeric_cols else 0, key="y_var")
    color_var = st.selectbox("Optional color variable", ["None"] + ["smog_risk"] + numeric_cols, key="color_var")
    color_arg = None if color_var == "None" else color_var

    if plot_type == "Scatter":
        fig = px.scatter(df, x=x_var, y=y_var, color=color_arg)
    elif plot_type == "Boxplot":
        fig = px.box(df, x=x_var if x_var in ["smog_risk", "month", "year"] else "smog_risk", y=y_var, color=color_arg)
    elif plot_type == "Histogram":
        fig = px.histogram(df, x=y_var, color=color_arg)
    else:
        mode = st.radio("Temporal display mode", ["Markers", "Lines+Markers"], horizontal=True)
        fig = px.scatter(df.sort_values("date"), x="date", y=y_var, color=color_arg)
        if mode == "Lines+Markers":
            fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Temporal analysis")
    vars_time = st.multiselect("Variables vs time", numeric_cols, default=["PM10"]) 
    temporal_kind = st.radio("Chart style", ["Markers", "Lines+Markers", "Bars"], horizontal=True)
    if vars_time:
        tdf = df[["date"] + vars_time].dropna().melt(id_vars="date", var_name="variable", value_name="value")
        if temporal_kind == "Bars":
            tfig = px.bar(tdf, x="date", y="value", color="variable", barmode="group")
        else:
            tfig = px.scatter(tdf, x="date", y="value", color="variable")
            if temporal_kind == "Lines+Markers":
                tfig.update_traces(mode="lines+markers")
        st.plotly_chart(tfig, use_container_width=True)

with tab4:
    st.markdown("### Decision Laboratory")
    if not st.session_state.run_analysis:
        st.info("Run scenario analysis from sidebar.")
    else:
        avg_pm10 = filtered_df["PM10"].mean()
        avg_co2 = filtered_df["CO2_emission"].mean()
        avg_energy = filtered_df["energy_demand"].mean()
        dominant_risk = filtered_df["smog_risk"].mode().iloc[0]
        pm10_exceed_days = int((filtered_df["PM10"] > 50).sum())
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("PM10 concentration [µg/m³]", f"{avg_pm10:.1f}")
        c2.metric("CO₂ emission index [-]", f"{avg_co2:.1f}")
        c3.metric("Energy demand index [0–100]", f"{avg_energy:.1f}")
        c4.metric("Smog risk category", str(dominant_risk).capitalize())
        c5.metric("Days exceeding PM10 daily limit", f"{pm10_exceed_days} / 35")
        bg, fg = risk_style(dominant_risk)
        st.markdown(f"<div class='risk-box' style='background:{bg}; border-color:{fg}; color:{fg};'><b>Estimated environmental response:</b> based on 50 nearest synthetic scenarios.</div>", unsafe_allow_html=True)
        st.caption("According to EU air-quality regulations, daily PM10 concentration above 50 µg/m³ should not occur more than 35 times per year.")
        interp=[]
        interp.append(f"Heating intensity responds to temperature ({temperature}°C → {calculated_heating:.1f}%).")
        interp.append("High wind improves dispersion." if wind_speed >= 6 else "Low wind limits dispersion and may increase PM accumulation." if wind_speed <= 2.5 else "Moderate wind has mixed dispersion effects.")
        interp.append("High traffic increases PM burden." if traffic_intensity >= 70 else "Traffic contribution is moderate.")
        interp.append("Higher renewable share lowers emission pressure." if renewable_share >= 40 else "Lower renewable share maintains stronger conventional-emission pressure.")
        st.markdown("#### Dynamic engineering interpretation")
        for i in interp:
            st.markdown(f"- {i}")

with tab5:
    st.markdown("### Model interpretation")
    st.markdown("**Inputs → Heating calculation → Nearest-scenario search → Environmental response estimation → Interpretation**")
    st.info("This is uncertainty-aware estimation: similar conditions can still produce varied outcomes.")

    st.markdown("### Build a simplified model")
    candidate_inputs = st.multiselect("Choose input variables", ["temperature","wind_speed","traffic_intensity","renewable_share","heating_intensity","local_emission","dispersion_factor"], default=["temperature","wind_speed","traffic_intensity","renewable_share"])
    output_var = st.selectbox("Choose output variable", ["PM10","CO2_emission","energy_demand"])
    if candidate_inputs:
        corr = df[candidate_inputs + [output_var]].corr(numeric_only=True)[output_var].drop(output_var).abs().sort_values(ascending=True)
        st.plotly_chart(px.bar(corr, orientation="h", title=f"What most influences {output_var} in this dataset?", labels={"value":"|correlation|", "index":"variable"}), use_container_width=True)

with tab6:
    st.markdown("### Interactive laboratory tasks")
    st.markdown("Use the sidebar controls to manipulate conditions, then complete guided engineering tasks below.")

    if not st.session_state.run_analysis:
        st.info("Step 1: Set scenario controls in the sidebar and click **Analyze Decision Scenario**. Then return to this tab.")
    else:
        avg_pm10 = filtered_df["PM10"].mean()
        avg_co2 = filtered_df["CO2_emission"].mean()
        avg_energy = filtered_df["energy_demand"].mean()
        dominant_risk = filtered_df["smog_risk"].mode().iloc[0]
        pm10_exceed_days = int((filtered_df["PM10"] > 50).sum())

        # ---------------- Basic ----------------
        st.markdown("#### 🟢 BASIC TASK — Identify low PM10 conditions")
        st.markdown("Goal: Find conditions that produce **estimated PM10 response below 35 µg/m³**.")
        if st.button("Check BASIC task"):
            if avg_pm10 < 35:
                st.success("✅ Task completed: estimated PM10 response is below 35 µg/m³.")
                st.info("Engineering feedback: low PM10 often appears with better dispersion and/or lower emission pressure.")
            else:
                st.warning("⚠ Not yet completed. Try increasing wind speed, reducing traffic intensity, or improving renewable share.")

        basic_note = st.text_area("What environmental changes improved air quality?", key="basic_note")
        with st.expander("Hint / example interpretation (BASIC)"):
            st.markdown("Example: Higher wind speed improved pollutant dispersion, while lower traffic reduced particulate emission pressure.")

        # ---------------- Intermediate ----------------
        st.markdown("#### 🟠 INTERMEDIATE TASK — Reduce PM10 with moderate demand")
        st.markdown("Goal: Keep **PM10 < 50 µg/m³** while maintaining **energy demand ≤ 70**.")
        if st.button("Check INTERMEDIATE task"):
            if avg_pm10 < 50 and avg_energy <= 70:
                st.success("✅ Task completed: PM10 is below the EU daily threshold and energy demand remains moderate.")
            elif avg_pm10 < 50 and avg_energy > 70:
                st.warning("⚠ PM10 target met, but energy demand exceeded the limit.")
            else:
                st.warning("⚠ Task not completed yet. Explore temperature, wind, and traffic combinations.")

        inter_note = st.text_area("How did you balance air quality and energy demand?", key="inter_note")
        with st.expander("Hint / example interpretation (INTERMEDIATE)"):
            st.markdown("Example: Mild temperature and stronger wind reduced PM10, while avoiding very high heating demand kept system load moderate.")

        # ---------------- Advanced ----------------
        st.markdown("#### 🔵 ADVANCED TASK — Scenario comparison and renewable transition")
        st.markdown("Goal: Compare two scenarios side-by-side and interpret how renewable-energy transition changes environmental response.")

        preset_names = list(predefined.keys())
        left_name = st.selectbox("LEFT scenario", preset_names, index=preset_names.index("Winter smog episode") if "Winter smog episode" in preset_names else 0)
        right_name = st.selectbox("RIGHT scenario", preset_names, index=preset_names.index("Summer clean air") if "Summer clean air" in preset_names else 0)

        left_vals = predefined[left_name]
        right_vals = predefined[right_name]

        left_df = get_nearest_scenarios(df, float(left_vals["temperature"]), float(left_vals["wind_speed"]), float(left_vals["traffic_intensity"]), float(left_vals["renewable_share"]), 50)
        right_df = get_nearest_scenarios(df, float(right_vals["temperature"]), float(right_vals["wind_speed"]), float(right_vals["traffic_intensity"]), float(right_vals["renewable_share"]), 50)

        def summarize(d):
            return {
                "PM10": d["PM10"].mean(),
                "CO2": d["CO2_emission"].mean(),
                "energy": d["energy_demand"].mean(),
                "risk": d["smog_risk"].mode().iloc[0],
                "exceed": int((d["PM10"] > 50).sum()),
            }

        ls, rs = summarize(left_df), summarize(right_df)
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"##### LEFT: {left_name}")
            st.metric("PM10 [µg/m³]", f"{ls['PM10']:.1f}")
            st.metric("CO₂ index [-]", f"{ls['CO2']:.1f}")
            st.metric("Energy demand [0–100]", f"{ls['energy']:.1f}")
            st.metric("Smog risk", str(ls['risk']).capitalize())
            st.metric("Exceedance days", f"{ls['exceed']} / 35")

        with col_r:
            st.markdown(f"##### RIGHT: {right_name}")
            st.metric("PM10 [µg/m³]", f"{rs['PM10']:.1f}")
            st.metric("CO₂ index [-]", f"{rs['CO2']:.1f}")
            st.metric("Energy demand [0–100]", f"{rs['energy']:.1f}")
            st.metric("Smog risk", str(rs['risk']).capitalize())
            st.metric("Exceedance days", f"{rs['exceed']} / 35")

        if st.button("Check ADVANCED task"):
            if right_vals["renewable_share"] > left_vals["renewable_share"] and rs["CO2"] <= ls["CO2"]:
                st.success("✅ Task completed: higher renewable share is associated with reduced CO₂ burden in the compared scenario.")
            else:
                st.warning("⚠ Try comparing a low-renewable scenario with a higher-renewable scenario and inspect CO₂ response.")

        adv_note = st.text_area("What does this comparison show about renewable transition and air quality?", key="adv_note")
        with st.expander("Hint / example interpretation (ADVANCED)"):
            st.markdown("Example: The scenario with higher renewable share showed lower CO₂ pressure; PM10 also depended strongly on wind and heating conditions.")

        st.markdown("#### Automatic educational summary")
        auto_lines = [
            "Changing conditions modifies the whole environmental-energy system response.",
            "Lower wind speed can increase pollution accumulation.",
            "Higher heating demand can intensify local emissions during colder conditions.",
            "Higher renewable-energy share can reduce environmental burden, especially CO₂ pressure.",
        ]
        for line in auto_lines:
            st.markdown(f"- {line}")

        st.markdown("#### Generate classroom report")
        report = f"""# Classroom report

Active scenario (sidebar): {scenario_name}
- Temperature: {temperature} °C
- Wind speed: {wind_speed} m/s
- Traffic intensity: {traffic_intensity} %
- Renewable share: {renewable_share} %

Estimated environmental response:
- PM10: {avg_pm10:.1f} µg/m³
- CO2 index: {avg_co2:.1f}
- Energy demand: {avg_energy:.1f}
- Smog risk: {dominant_risk}
- PM10 exceedance days: {pm10_exceed_days} / 35

Student conclusions:
BASIC: {basic_note}
INTERMEDIATE: {inter_note}
ADVANCED: {adv_note}
"""
        st.download_button("Download classroom report (.md)", data=report, file_name="classroom_report.md", mime="text/markdown")

st.caption("Educational note: synthetic dataset + simplified environmental-engineering logic for classroom reasoning.")
