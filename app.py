import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Page configuration + lightweight styling
# ------------------------------
st.set_page_config(page_title="Environmental-Energy Digital Twin", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .subtitle {font-size: 1.05rem; color: #4A5568; margin-bottom: 1rem;}
    .section-gap {margin-top: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 Educational Environmental-Energy Digital Twin")
st.markdown(
    '<div class="subtitle">A simplified decision-support simulator for first-year Energy Engineering students. '
    "Change environmental conditions and observe system-level responses under uncertainty.</div>",
    unsafe_allow_html=True,
)




# Centralized heating formula to keep merge conflict resolution simple and consistent.
def heating_from_temperature(temp_c: float) -> float:
    return max(5.0, min(100.0, 100 - ((temp_c + 15) * 2.8)))


def calculate_heating_intensity(temperature: float) -> float:
    """Backward-compatible alias for heating formula used in some merged/local variants."""
    return heating_from_temperature(float(temperature))

if not callable(calculate_heating_intensity):
    raise RuntimeError("calculate_heating_intensity is not callable. Resolve merge conflicts in app.py.")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load dataset and parse date column safely."""
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


def get_nearest_scenarios(data: pd.DataFrame, t: float, w: float, tr: float, r: float, top_n: int = 50) -> pd.DataFrame:
    """Find most similar scenarios using weighted Manhattan distance."""
    scenario_heating = heating_from_temperature(t)

    # Normalize differences by feature ranges to make components comparable.
    dist = (
        (data["temperature"].sub(t).abs() / 40.0)
        + (data["wind_speed"].sub(w).abs() / 12.0)
        + (data["traffic_intensity"].sub(tr).abs() / 100.0)
        + (data["renewable_share"].sub(r).abs() / 60.0)
        + (data["heating_intensity"].sub(scenario_heating).abs() / 100.0)
    )

    out = data.copy()
    out["similarity_distance"] = dist
    out = out.nsmallest(min(top_n, len(out)), "similarity_distance").copy()
    return out


DATA_FILE = "smog_energy_dataset.csv"
required_columns = [
    "date",
    "year",
    "month",
    "temperature",
    "wind_speed",
    "heating_intensity",
    "traffic_intensity",
    "renewable_share",
    "energy_demand",
    "local_emission",
    "dispersion_factor",
    "PM10",
    "CO2_emission",
    "smog_risk",
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
# Sidebar controls
# ------------------------------
predefined = {
    "Custom": {"temperature": 0, "wind_speed": 3.0, "traffic_intensity": 50, "renewable_share": 25},
    "Winter smog episode": {"temperature": -8, "wind_speed": 1.2, "traffic_intensity": 78, "renewable_share": 12},
    "Summer clean air": {"temperature": 22, "wind_speed": 6.5, "traffic_intensity": 35, "renewable_share": 35},
    "Energy transition": {"temperature": 6, "wind_speed": 4.0, "traffic_intensity": 45, "renewable_share": 55},
    "High traffic urban day": {"temperature": 4, "wind_speed": 2.0, "traffic_intensity": 92, "renewable_share": 20},
}

st.sidebar.header("Scenario Builder")
scenario_name = st.sidebar.selectbox("Educational scenario", list(predefined.keys()))
defaults = predefined[scenario_name]

temperature = st.sidebar.slider("Outdoor temperature [°C]", -15, 25, int(defaults["temperature"]))
wind_speed = st.sidebar.slider("Wind speed [m/s]", 0.0, 12.0, float(defaults["wind_speed"]), 0.1)
traffic_intensity = st.sidebar.slider("Traffic intensity [%]", 0, 100, int(defaults["traffic_intensity"]))
renewable_share = st.sidebar.slider("Renewable energy share [%]", 0, 60, int(defaults["renewable_share"]))

# Calculate heating directly from temperature to avoid runtime dependency issues.
calculated_heating = calculate_heating_intensity(float(temperature))
st.sidebar.metric("Calculated heating intensity [%]", f"{calculated_heating:.1f}")
st.sidebar.caption("Heating is automatically derived from temperature to avoid unrealistic system states.")

analyze = st.sidebar.button("Analyze Scenario", type="primary")

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if analyze:
    st.session_state.run_analysis = True

# Always produce similar scenarios once user starts analysis.
filtered_df = pd.DataFrame(columns=df.columns)
if st.session_state.run_analysis:
    filtered_df = get_nearest_scenarios(
        df,
        t=float(temperature),
        w=float(wind_speed),
        tr=float(traffic_intensity),
        r=float(renewable_share),
        top_n=50,
    )

if not st.session_state.run_analysis:
    st.info("Choose inputs in the sidebar and click **Analyze Scenario** to run the digital twin.")
else:
    avg_pm10 = filtered_df["PM10"].mean()
    avg_co2 = filtered_df["CO2_emission"].mean()
    avg_energy = filtered_df["energy_demand"].mean()
    dominant_risk = filtered_df["smog_risk"].mode().iloc[0] if not filtered_df["smog_risk"].mode().empty else "unknown"
    smog_alert_days = int((filtered_df["PM10"] > 100).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg PM10 [µg/m³]", f"{avg_pm10:.1f}")
    k2.metric("Avg CO₂ emission index", f"{avg_co2:.1f}")
    k3.metric("Avg energy demand", f"{avg_energy:.1f}")
    k4.metric("Dominant smog risk", str(dominant_risk).capitalize())
    k5.metric("Smog alert days", f"{smog_alert_days} / {len(filtered_df)}")

    risk_colors = {
        "low": ("#E6FFFA", "#065F46"),
        "moderate": ("#FFF7D6", "#92400E"),
        "high": ("#FFEDD5", "#9A3412"),
        "alarm": ("#FEE2E2", "#991B1B"),
    }
    bg, fg = risk_colors.get(str(dominant_risk).lower(), ("#E5E7EB", "#1F2937"))
    st.markdown(
        f"""
        <div style="margin-top:0.8rem; margin-bottom:0.8rem; padding: 1rem 1.1rem; border-radius: 0.7rem;
                    background-color: {bg}; color: {fg}; border-left: 8px solid {fg};">
            <b>System smog status:</b> {str(dominant_risk).capitalize()}<br>
            This estimate is based on the <b>50 most similar synthetic scenarios</b>, not exact matching.
            It reflects uncertainty-aware environmental engineering reasoning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Scenario Similarity Diagnostics")
    st.caption("Lower distance = stronger similarity to selected input conditions.")
    st.plotly_chart(
        px.histogram(filtered_df, x="similarity_distance", nbins=20, title="Distribution of similarity distances (50 nearest scenarios)"),
        use_container_width=True,
    )

    st.markdown("### Visual Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.scatter(filtered_df, x="temperature", y="PM10", color="smog_risk", size="traffic_intensity", title="A. Temperature vs PM10"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.scatter(filtered_df, x="wind_speed", y="PM10", color="smog_risk", size="heating_intensity", title="B. Wind speed vs PM10"),
            use_container_width=True,
        )

    monthly_pm10 = filtered_df.groupby("month", as_index=False)["PM10"].mean().sort_values("month")
    st.plotly_chart(
        px.line(monthly_pm10, x="month", y="PM10", markers=True, title="C. Monthly average PM10"),
        use_container_width=True,
    )

    risk_pm10 = filtered_df.groupby("smog_risk", as_index=False)["PM10"].mean()
    st.plotly_chart(
        px.bar(risk_pm10, x="smog_risk", y="PM10", color="smog_risk", title="D. Average PM10 by smog risk category"),
        use_container_width=True,
    )

    corr_cols = ["temperature", "wind_speed", "heating_intensity", "traffic_intensity", "renewable_share", "PM10", "CO2_emission"]
    corr = filtered_df[corr_cols].corr(numeric_only=True)
    fig_heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0))
    fig_heat.update_layout(title="E. Correlation heatmap", height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Engineering Interpretation Report")
    lines = []
    if temperature <= 0:
        lines.append(f"- **Thermal driver:** Outdoor temperature ({temperature}°C) increased modeled heating intensity to **{calculated_heating:.1f}%**, elevating combustion pressure.")
    else:
        lines.append(f"- **Thermal driver:** Moderate temperature ({temperature}°C) reduced modeled heating intensity to **{calculated_heating:.1f}%**.")
    if wind_speed <= 2.5:
        lines.append(f"- **Dispersion driver:** Weak wind ({wind_speed:.1f} m/s) limited atmospheric mixing and favored PM10 accumulation.")
    elif wind_speed >= 6:
        lines.append(f"- **Dispersion driver:** Stronger wind ({wind_speed:.1f} m/s) improved pollutant dispersion and reduced stagnation risk.")
    if renewable_share >= 40:
        lines.append(f"- **Energy transition driver:** High renewable share ({renewable_share}%) lowered local-emission intensity in similar scenarios.")
    else:
        lines.append(f"- **Energy transition driver:** Renewable share ({renewable_share}%) indicates partial decarbonization potential; higher values generally improve outcomes.")
    if traffic_intensity >= 70:
        lines.append(f"- **Transport driver:** High traffic intensity ({traffic_intensity}%) was associated with increased PM10 burden in nearest analogues.")
    else:
        lines.append(f"- **Transport driver:** Traffic intensity ({traffic_intensity}%) remained a relevant but moderate PM10 contributor.")
    lines.append(f"- **System summary:** Across the 50 nearest scenarios, mean PM10 = **{avg_pm10:.1f} µg/m³**, mean CO₂ index = **{avg_co2:.1f}**, smog-alert days = **{smog_alert_days}**.")
    st.markdown("\n".join(lines))

with st.expander("How does this environmental-energy model work?", expanded=False):
    st.markdown(
        """
### A. What data are used?
This app uses a **synthetic environmental-energy dataset** created for education. It represents interactions between:
- weather conditions,
- heating demand,
- emissions,
- air quality,
- renewable energy transition.

### B. Input variables (student-controlled)
- **Outdoor temperature [°C]**: ambient atmospheric condition; lower values raise heating demand.
- **Wind speed [m/s]**: proxy for atmospheric mixing; higher values typically improve pollutant dispersion.
- **Traffic intensity [%]**: proxy for transport-related emissions and urban PM contributions.
- **Renewable energy share [%]**: contribution of lower-emission energy sources in the local energy mix.

### C. Calculated variables (model/system outputs)
- **Heating intensity [%]**: automatically derived from temperature using a simplified engineering relationship.
- **Local emission index [-]**: synthetic indicator of near-source emissions.
- **PM10 concentration [µg/m³]**: particulate matter concentration used for air-quality interpretation.
- **CO2 emission index [-]**: synthetic greenhouse-gas burden indicator.
- **Energy demand [index]**: synthetic total demand pressure.
- **Smog risk category [-]**: categorical risk class (low/moderate/high/alarm).

### Privacy and telemetry note
Streamlit may collect anonymous usage statistics at the platform level. If needed, you can disable it in:
`~/.streamlit/config.toml` with:
```toml
[browser]
gatherUsageStats = false
```

### D. How the model works under uncertainty
The app does **not** predict exact future values. Instead, it finds the **50 most similar scenarios** (nearest neighbors)
in the synthetic dataset based on selected conditions and calculated heating demand. This mimics uncertainty-aware,
probabilistic reasoning used in real environmental-energy decision-support systems.
        """
    )

st.markdown("### Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("### Download filtered scenarios")
if st.session_state.run_analysis and not filtered_df.empty:
    st.download_button(
        label="Download 50 nearest scenarios as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="nearest_smog_energy_scenarios.csv",
        mime="text/csv",
    )
else:
    st.caption("Run scenario analysis to enable filtered dataset download.")

show_stats = st.checkbox("Show raw statistics")
if show_stats:
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
