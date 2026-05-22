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
    st.markdown("### Interactive educational digital twin")
    st.info("Synthetic dataset, uncertainty-aware nearest-scenario estimation, simplified environmental-energy cause-effect reasoning.")
    c1,c2,c3 = st.columns(3)
    c1.markdown("<div class='lab-card'><b>🌡 Conditions</b><br><span class='soft'>Weather + energy mix inputs</span></div>", unsafe_allow_html=True)
    c2.markdown("<div class='lab-card'><b>🏭 Response</b><br><span class='soft'>PM10, CO₂, demand, smog risk</span></div>", unsafe_allow_html=True)
    c3.markdown("<div class='lab-card'><b>🧭 Decisions</b><br><span class='soft'>Interpret and compare scenarios</span></div>", unsafe_allow_html=True)

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
        smog_alert_days = int((filtered_df["PM10"] > 100).sum())
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("PM10 concentration [µg/m³]", f"{avg_pm10:.1f}")
        c2.metric("CO₂ emission index [-]", f"{avg_co2:.1f}")
        c3.metric("Energy demand index [0–100]", f"{avg_energy:.1f}")
        c4.metric("Smog risk category", str(dominant_risk).capitalize())
        c5.metric("Estimated winter smog-alert days [days/year]", f"{smog_alert_days}")
        bg, fg = risk_style(dominant_risk)
        st.markdown(f"<div class='risk-box' style='background:{bg}; border-color:{fg}; color:{fg};'><b>Decision summary:</b> based on 50 nearest synthetic scenarios.</div>", unsafe_allow_html=True)
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
    st.markdown("### Interactive student tasks")
    if not st.session_state.run_analysis:
        st.warning("Run scenario first.")
    else:
        avg_pm10 = filtered_df["PM10"].mean()
        avg_co2 = filtered_df["CO2_emission"].mean()
        avg_energy = filtered_df["energy_demand"].mean()
        dominant_risk = filtered_df["smog_risk"].mode().iloc[0]

        st.markdown("#### TASK 1")
        st.write("Reduce PM10 below 35 µg/m³ while renewable share ≤ 40% and energy demand ≤ 70")
        if st.button("Check Task 1"):
            if avg_pm10 < 35 and renewable_share <= 40 and avg_energy <= 70:
                st.success("✅ Successful environmental mitigation strategy")
            elif avg_pm10 < 35 and avg_energy > 70:
                st.warning("⚠ PM10 reduced, but energy demand exceeded threshold")
            else:
                st.warning("⚠ Task conditions not met")

        st.markdown("#### TASK 2")
        st.write("Reduce CO₂ emissions without increasing smog risk")
        if st.button("Check Task 2"):
            if avg_co2 < df["CO2_emission"].median() and str(dominant_risk).lower() in ["low","moderate"]:
                st.success("✅ CO₂ reduced with controlled smog risk")
            else:
                st.warning("⚠ CO₂ and smog-risk target not yet achieved")

        st.markdown("#### TASK 3")
        st.write("Find conditions producing HIGH smog risk")
        if st.button("Check Task 3"):
            if str(dominant_risk).lower() == "high":
                st.success("✅ High smog risk identified")
            else:
                st.warning("⚠ Current scenario is not high risk")

        st.markdown("#### Generate classroom report")
        student_note = st.text_area("Student conclusions")
        report = f"""# Classroom report\n\nScenario: {scenario_name}\n- Temperature: {temperature} °C\n- Wind speed: {wind_speed} m/s\n- Traffic intensity: {traffic_intensity} %\n- Renewable share: {renewable_share} %\n\nOutcomes:\n- PM10: {avg_pm10:.1f} µg/m³\n- CO2 index: {avg_co2:.1f}\n- Energy demand: {avg_energy:.1f}\n- Smog risk: {dominant_risk}\n\nInterpretation:\n""" + "\n".join([f"- {i}" for i in interp]) + f"\n\nStudent conclusions:\n{student_note}\n"
        st.download_button("Download classroom report (.md)", data=report, file_name="classroom_report.md", mime="text/markdown")

st.caption("Educational note: synthetic dataset + simplified environmental-engineering logic for classroom reasoning.")
