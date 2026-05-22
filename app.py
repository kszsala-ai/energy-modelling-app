import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------
# Page config + engineering visual style
# ------------------------------
st.set_page_config(
    page_title="Environmental-Energy Decision Laboratory",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0F172A;
        --panel: #111827;
        --panel2: #1F2937;
        --text: #E5E7EB;
        --muted: #9CA3AF;
        --steel: #334155;
        --orange: #C2410C;
        --green: #065F46;
        --yellow: #92400E;
        --red: #991B1B;
        --blue: #1D4ED8;
    }
    .stApp {background: linear-gradient(180deg, #0B1220 0%, #111827 100%); color: var(--text);}
    .block-container {padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 1350px;}
    h1,h2,h3 {color: #E2E8F0 !important; letter-spacing: 0.2px;}
    .lab-card {
        background: rgba(31,41,55,0.86);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .soft {color: #94A3B8; font-size: 0.95rem;}
    .risk-box {padding: 0.9rem 1rem; border-radius: 10px; border-left: 7px solid; margin-top: 0.4rem;}
    .intro-box {padding: 0.9rem 1rem; border-radius: 10px; border: 1px solid #334155; background: rgba(30,41,59,0.65);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# Core functions
# ------------------------------
def heating_from_temperature(temp_c: float) -> float:
    return max(5.0, min(100.0, 100 - ((temp_c + 15) * 2.8)))


def calculate_heating_intensity(temperature: float) -> float:
    """Backward-compatible alias for merged variants."""
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
        "low": ("#064E3B", "#6EE7B7"),
        "moderate": ("#78350F", "#FCD34D"),
        "high": ("#7C2D12", "#FDBA74"),
        "alarm": ("#7F1D1D", "#FCA5A5"),
    }
    return mapper.get(str(risk).lower(), ("#334155", "#E2E8F0"))


# ------------------------------
# Data loading + validation
# ------------------------------
DATA_FILE = "smog_energy_dataset.csv"
required_columns = [
    "date", "year", "month", "temperature", "wind_speed", "heating_intensity",
    "traffic_intensity", "renewable_share", "energy_demand", "local_emission",
    "dispersion_factor", "PM10", "CO2_emission", "smog_risk"
]

try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"Dataset file '{DATA_FILE}' was not found. Place it next to app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

missing = [col for col in required_columns if col not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(missing)}")
    st.stop()


# ------------------------------
# Sidebar scenario controls
# ------------------------------
predefined = {
    "Custom": {"temperature": 0, "wind_speed": 3.0, "traffic_intensity": 50, "renewable_share": 25},
    "Winter smog episode": {"temperature": -8, "wind_speed": 1.2, "traffic_intensity": 78, "renewable_share": 12},
    "Summer clean air": {"temperature": 22, "wind_speed": 6.5, "traffic_intensity": 35, "renewable_share": 35},
    "Energy transition": {"temperature": 6, "wind_speed": 4.0, "traffic_intensity": 45, "renewable_share": 55},
    "High traffic urban day": {"temperature": 4, "wind_speed": 2.0, "traffic_intensity": 92, "renewable_share": 20},
}

st.sidebar.markdown("## ⚙️ Scenario Controls")
scenario_name = st.sidebar.selectbox("Scenario mode", list(predefined.keys()))
defaults = predefined[scenario_name]

temperature = st.sidebar.slider("Outdoor temperature [°C]", -15, 25, int(defaults["temperature"]))
wind_speed = st.sidebar.slider("Wind speed [m/s]", 0.0, 12.0, float(defaults["wind_speed"]), 0.1)
traffic_intensity = st.sidebar.slider("Traffic intensity [%]", 0, 100, int(defaults["traffic_intensity"]))
renewable_share = st.sidebar.slider("Renewable energy share [%]", 0, 60, int(defaults["renewable_share"]))

calculated_heating = calculate_heating_intensity(float(temperature))
st.sidebar.metric("Calculated heating intensity [%]", f"{calculated_heating:.1f}")
st.sidebar.caption("Heating is auto-derived from temperature (simplified engineering rule).")

analyze = st.sidebar.button("Analyze Decision Scenario", type="primary")
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if analyze:
    st.session_state.run_analysis = True

filtered_df = pd.DataFrame(columns=df.columns)
if st.session_state.run_analysis:
    filtered_df = get_nearest_scenarios(
        df, float(temperature), float(wind_speed), float(traffic_intensity), float(renewable_share), 50
    )


# ------------------------------
# Top navigation tabs
# ------------------------------
st.title("Environmental-Energy Decision Support Laboratory")
st.caption("Educational digital twin for first-year Energy Engineering: reasoning under uncertainty, not deterministic forecasting.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Introduction",
    "Dataset & Variables",
    "Environmental Relationships",
    "Decision Laboratory",
    "Model Interpretation",
    "Student Tasks",
])

with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### 🧭 What is this laboratory?")
        st.markdown(
            """
            <div class="intro-box">
            <b>Digital twin (simplified):</b> a virtual teaching model of an environmental-energy system.<br>
            Students adjust conditions and observe likely system responses based on similar synthetic scenarios.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="intro-box">
            <b>Uncertainty-aware modelling:</b> the app does not predict an exact future value.
            It estimates outcomes from the most similar cases in an educational dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("### 🧪 Learning focus")
        st.markdown("- Environmental-energy interactions")
        st.markdown("- Cause-effect reasoning")
        st.markdown("- Practical decision consequences")
        st.markdown("- Smog and emission mitigation thinking")
        st.info("Dataset is synthetic and educational. Model is simplified for classroom interpretation.")

with tab2:
    st.markdown("### Variable map")
    c_in, c_model, c_out = st.columns(3)
    with c_in:
        st.markdown("#### Input variables")
        st.markdown("- **Outdoor temperature [°C]**: ambient condition; colder air raises heating demand.")
        st.markdown("- **Wind speed [m/s]**: atmospheric mixing; higher wind improves dispersion.")
        st.markdown("- **Traffic intensity [%]**: transport-related pollutant pressure.")
        st.markdown("- **Renewable energy share [%]**: lower-emission part of energy mix.")
    with c_model:
        st.markdown("#### Model variables")
        st.markdown("- **Heating intensity [%]**: calculated from temperature.")
        st.markdown("- **Local emission index [-]**: synthetic source-emission pressure.")
        st.markdown("- **Dispersion factor [-]**: synthetic atmospheric transport efficiency.")
    with c_out:
        st.markdown("#### Output variables")
        st.markdown("- **PM10 concentration [µg/m³]**: particulate pollution burden.")
        st.markdown("- **CO₂ emission index [-]**: climate-pressure indicator.")
        st.markdown("- **Energy demand index [0–100]**: system demand intensity.")
        st.markdown("- **Smog risk category**: low/moderate/high/alarm.")

    st.markdown("### Dataset preview & distributions")
    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.dataframe(df.head(12), use_container_width=True)
        st.dataframe(df[["temperature", "wind_speed", "traffic_intensity", "renewable_share", "PM10", "CO2_emission"]].describe().round(2), use_container_width=True)
    with p2:
        st.plotly_chart(px.histogram(df, x="PM10", nbins=30, title="PM10 distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="CO2_emission", nbins=30, title="CO₂ emission index distribution"), use_container_width=True)

with tab3:
    st.markdown("### A. Temperature vs Heating Demand")
    st.plotly_chart(
        px.scatter(df, x="temperature", y="heating_intensity", color_discrete_sequence=["#60A5FA"], title="Colder weather tends to increase heating demand"),
        use_container_width=True,
    )
    st.caption("Engineering interpretation: in this synthetic system, lower outdoor temperature increases heating intensity and fuel-related emission pressure.")

    st.markdown("### B. Wind Speed vs PM10")
    st.plotly_chart(
        px.scatter(df, x="wind_speed", y="PM10", color_discrete_sequence=["#F59E0B"], title="Wind supports pollutant dispersion"),
        use_container_width=True,
    )
    st.caption("Engineering interpretation: weak wind conditions often coincide with PM10 accumulation near ground level.")

    st.markdown("### C. Renewable Share vs CO₂")
    st.plotly_chart(
        px.scatter(df, x="renewable_share", y="CO2_emission", color_discrete_sequence=["#34D399"], title="Renewables and CO₂ relationship"),
        use_container_width=True,
    )
    st.caption("Engineering interpretation: larger renewable share is generally associated with lower CO₂ emission index.")

    st.markdown("### D. Traffic Intensity vs PM10")
    st.plotly_chart(
        px.scatter(df, x="traffic_intensity", y="PM10", color_discrete_sequence=["#FB7185"], title="Traffic contribution to PM10"),
        use_container_width=True,
    )
    st.caption("Engineering interpretation: traffic pressure contributes to particulate burden, especially with poor dispersion conditions.")

with tab4:
    st.markdown("### Decision consequences")
    if not st.session_state.run_analysis:
        st.info("Set a scenario in the sidebar and click **Analyze Decision Scenario**.")
    else:
        avg_pm10 = filtered_df["PM10"].mean()
        avg_co2 = filtered_df["CO2_emission"].mean()
        avg_energy = filtered_df["energy_demand"].mean()
        dominant_risk = filtered_df["smog_risk"].mode().iloc[0] if not filtered_df["smog_risk"].mode().empty else "unknown"
        smog_alert_days = int((filtered_df["PM10"] > 100).sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PM10 concentration [µg/m³]", f"{avg_pm10:.1f}")
        m2.metric("CO₂ emission index [-]", f"{avg_co2:.1f}")
        m3.metric("Energy demand index [0–100]", f"{avg_energy:.1f}")
        m4.metric("Smog risk category", str(dominant_risk).capitalize())
        m5.metric("Estimated winter smog-alert days [days/year]", f"{smog_alert_days}")

        bg, fg = risk_style(dominant_risk)
        st.markdown(
            f"""
            <div class="risk-box" style="background:{bg}; border-color:{fg}; color:{fg};">
            <b>Engineering decision summary:</b> Scenario outcomes are estimated from the 50 most similar synthetic cases.
            This seasonal estimate suggests a <b>{str(dominant_risk).capitalize()}</b> smog-risk regime.
            </div>
            """,
            unsafe_allow_html=True,
        )

        short = []
        if temperature <= 0:
            short.append("Low temperature increased heating intensity and emission pressure.")
        if wind_speed <= 2.5:
            short.append("Low wind speed reduced pollutant dispersion and increased PM10 accumulation risk.")
        if traffic_intensity >= 70:
            short.append("High traffic intensity strengthened PM10 load.")
        if renewable_share >= 40:
            short.append("Higher renewable share helped reduce CO₂-related pressure.")
        if not short:
            short.append("Inputs are moderate; use scenario presets to compare stronger environmental responses.")

        st.markdown("#### Concise interpretation")
        for line in short:
            st.markdown(f"- {line}")
        st.caption("Estimated winter smog-alert days are simplified seasonal indicators derived from nearest synthetic scenarios.")

with tab5:
    st.markdown("### How the uncertainty-aware model works")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("<div class='lab-card'><b>1) Inputs</b><br><span class='soft'>Temperature, wind, traffic, renewables</span></div>", unsafe_allow_html=True)
    s2.markdown("<div class='lab-card'><b>2) Heating</b><br><span class='soft'>Automatic heating calculation</span></div>", unsafe_allow_html=True)
    s3.markdown("<div class='lab-card'><b>3) Similar cases</b><br><span class='soft'>50 nearest synthetic scenarios</span></div>", unsafe_allow_html=True)
    s4.markdown("<div class='lab-card'><b>4) Response</b><br><span class='soft'>PM10, CO₂, demand, risk interpretation</span></div>", unsafe_allow_html=True)

    st.markdown("#### Conceptual workflow")
    st.markdown("**Inputs** → **Heating calculation** → **Nearest-scenario search** → **Environmental response estimation** → **Interpretation**")
    st.info("Environmental systems are variable; uncertainty appears because similar conditions can still produce a range of outcomes.")

with tab6:
    st.markdown("### Classroom tasks (interactive)")
    if not st.session_state.run_analysis:
        st.warning("Run **Analyze Decision Scenario** first to evaluate tasks.")
    else:
        avg_pm10 = filtered_df["PM10"].mean()
        avg_co2 = filtered_df["CO2_emission"].mean()
        avg_energy = filtered_df["energy_demand"].mean()
        dominant_risk = filtered_df["smog_risk"].mode().iloc[0] if not filtered_df["smog_risk"].mode().empty else "unknown"

        t1 = st.checkbox("TASK 1: Reduce PM10 below 35 µg/m³ with renewable share ≤ 40% and energy demand ≤ 70")
        if t1:
            condition = (avg_pm10 < 35) and (renewable_share <= 40) and (avg_energy <= 70)
            if condition:
                st.success("Task 1 completed: scenario meets PM10, renewable-share, and energy-demand constraints.")
            else:
                st.warning("Task 1 not met. Hint: increase wind or renewable share while controlling traffic and temperature impact.")

        t2 = st.checkbox("TASK 2: Reduce CO₂ emissions without increasing smog risk")
        if t2:
            condition = (avg_co2 < df["CO2_emission"].median()) and (str(dominant_risk).lower() in ["low", "moderate"])
            if condition:
                st.success("Task 2 completed: lower CO₂ with stable/non-high smog risk.")
            else:
                st.warning("Task 2 not met. Hint: push renewable share upward and avoid low-wind, high-traffic combinations.")

        t3 = st.checkbox("TASK 3: Find conditions producing HIGH smog risk")
        if t3:
            condition = str(dominant_risk).lower() == "high"
            if condition:
                st.success("Task 3 completed: this scenario produces HIGH smog risk.")
            else:
                st.warning("Task 3 not met. Hint: try colder temperature, weaker wind, and higher traffic.")

st.markdown("---")
st.caption("Educational note: this laboratory uses synthetic data and simplified relationships for teaching systems thinking.")
