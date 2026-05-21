import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------
# Page configuration and styling
# ------------------------------
st.set_page_config(
    page_title="Environmental-Energy Lab",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Interactive Environmental-Energy System Simulator")
st.markdown(
    """
This educational app helps first-year Energy Engineering students explore how
**temperature, heating, wind, traffic, and renewable energy share** influence
**PM10 concentration** and **CO₂ emissions**.
Use the sidebar to build a scenario, then analyze similar historical synthetic records.
"""
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(path)
    # Parse dates safely; invalid entries become NaT.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ------------------------------
# Data loading
# ------------------------------
DATA_FILE = "smog_energy_dataset.csv"

try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Dataset file '{DATA_FILE}' was not found. Place it in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

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

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()


# ------------------------------
# Sidebar: scenario controls
# ------------------------------
st.sidebar.header("Scenario Inputs")
st.sidebar.markdown("Adjust parameters and click **Analyze Scenario**.")

temperature = st.sidebar.slider("Outdoor temperature (°C)", -15, 25, 0)
wind_speed = st.sidebar.slider("Wind speed (m/s)", 0.0, 12.0, 3.0, 0.1)
traffic_intensity = st.sidebar.slider("Traffic intensity", 0, 100, 50)
renewable_share = st.sidebar.slider("Renewable energy share (%)", 0, 60, 25)
heating_intensity = st.sidebar.slider("Heating intensity", 0, 100, 40)

analyze = st.sidebar.button("Analyze Scenario", type="primary")

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if analyze:
    st.session_state.run_analysis = True


# ------------------------------
# Filtering logic
# ------------------------------
if st.session_state.run_analysis:
    filtered_df = df[
        (df["temperature"].between(temperature - 4, temperature + 4))
        & (df["wind_speed"].between(wind_speed - 2, wind_speed + 2))
        & (df["traffic_intensity"].between(traffic_intensity - 20, traffic_intensity + 20))
        & (df["renewable_share"].between(renewable_share - 10, renewable_share + 10))
        & (df["heating_intensity"].between(heating_intensity - 20, heating_intensity + 20))
    ].copy()
else:
    filtered_df = pd.DataFrame(columns=df.columns)


# ------------------------------
# Main content
# ------------------------------
if not st.session_state.run_analysis:
    st.info("Select inputs in the sidebar and click **Analyze Scenario** to start.")
elif filtered_df.empty:
    st.warning(
        "No similar scenarios found for the selected ranges. Try broader or different values."
    )
else:
    # ------------------------------
    # KPI Section
    # ------------------------------
    avg_pm10 = filtered_df["PM10"].mean()
    avg_co2 = filtered_df["CO2_emission"].mean()
    avg_energy = filtered_df["energy_demand"].mean()
    dominant_risk = filtered_df["smog_risk"].mode().iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average PM10", f"{avg_pm10:.2f}")
    k2.metric("Average CO₂ emission", f"{avg_co2:.2f}")
    k3.metric("Average energy demand", f"{avg_energy:.2f}")
    k4.metric("Dominant smog risk", f"{dominant_risk}")

    # ------------------------------
    # Smog risk warning box
    # ------------------------------
    risk_colors = {
        "low": ("#D4EDDA", "#155724"),
        "moderate": ("#FFF3CD", "#856404"),
        "high": ("#FFE8CC", "#8A4B08"),
        "alarm": ("#F8D7DA", "#721C24"),
    }
    bg, fg = risk_colors.get(str(dominant_risk).lower(), ("#E2E3E5", "#383D41"))

    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.6rem; background-color: {bg}; color: {fg}; border: 1px solid {fg};">
            <b>Smog risk status:</b> {dominant_risk.capitalize()}<br>
            Interpretation: The selected scenario is currently associated with a <b>{dominant_risk}</b> smog risk level.
            Combine this with PM10 and CO₂ indicators below to assess environmental impact.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Visualizations")

    c1, c2 = st.columns(2)

    with c1:
        fig_temp_pm10 = px.scatter(
            filtered_df,
            x="temperature",
            y="PM10",
            color="smog_risk",
            title="A. Temperature vs PM10",
            hover_data=["wind_speed", "heating_intensity", "traffic_intensity"],
        )
        st.plotly_chart(fig_temp_pm10, use_container_width=True)

    with c2:
        fig_wind_pm10 = px.scatter(
            filtered_df,
            x="wind_speed",
            y="PM10",
            color="smog_risk",
            title="B. Wind Speed vs PM10",
            hover_data=["temperature", "heating_intensity", "traffic_intensity"],
        )
        st.plotly_chart(fig_wind_pm10, use_container_width=True)

    # C. Monthly average PM10
    monthly_pm10 = (
        filtered_df.groupby("month", as_index=False)["PM10"].mean().sort_values("month")
    )
    fig_monthly = px.line(
        monthly_pm10,
        x="month",
        y="PM10",
        markers=True,
        title="C. Monthly Average PM10",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # D. Average PM10 by smog risk
    risk_pm10 = filtered_df.groupby("smog_risk", as_index=False)["PM10"].mean()
    fig_risk = px.bar(
        risk_pm10,
        x="smog_risk",
        y="PM10",
        color="smog_risk",
        title="D. Average PM10 by Smog Risk Category",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # E. Correlation heatmap
    corr_cols = [
        "temperature",
        "wind_speed",
        "heating_intensity",
        "traffic_intensity",
        "renewable_share",
        "PM10",
        "CO2_emission",
    ]
    corr_matrix = filtered_df[corr_cols].corr(numeric_only=True)

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Correlation"),
        )
    )
    heatmap.update_layout(title="E. Correlation Heatmap", height=550)
    st.plotly_chart(heatmap, use_container_width=True)

    # ------------------------------
    # Automatic interpretation
    # ------------------------------
    st.markdown("### Automatic Interpretation")

    interpretation_points = []

    if temperature < 0:
        interpretation_points.append(
            "Low temperature likely increased heating demand, which can raise local PM10 concentration."
        )
    elif temperature > 15:
        interpretation_points.append(
            "Milder temperature conditions are usually linked with lower heating pressure and potentially lower PM10 from heating sources."
        )

    if wind_speed >= 6:
        interpretation_points.append(
            "Higher wind speed improved pollutant dispersion, which often helps reduce local PM10 accumulation."
        )
    elif wind_speed <= 2:
        interpretation_points.append(
            "Low wind speed limited dispersion, so pollutants can accumulate more easily near the ground."
        )

    if renewable_share >= 40:
        interpretation_points.append(
            "A higher renewable energy share is associated with lower local emissions and can support CO₂ reduction."
        )
    elif renewable_share <= 15:
        interpretation_points.append(
            "A low renewable share suggests stronger dependence on conventional fuels, which can increase emissions."
        )

    if traffic_intensity >= 70:
        interpretation_points.append(
            "High traffic intensity likely contributed to higher PM10 levels through transport-related emissions and resuspension."
        )
    elif traffic_intensity <= 30:
        interpretation_points.append(
            "Lower traffic intensity reduces one of the key urban PM10 sources."
        )

    if heating_intensity >= 70:
        interpretation_points.append(
            "High heating intensity can significantly increase combustion-related particulate emissions during colder periods."
        )

    if not interpretation_points:
        interpretation_points.append(
            "This scenario is balanced across major inputs; compare charts to identify which variable still has the strongest influence."
        )

    for point in interpretation_points:
        st.markdown(f"- {point}")

# ------------------------------
# Educational and extra sections
# ------------------------------
with st.expander("How does the model work?"):
    st.write(
        """
        This app uses an **educational synthetic dataset** to simulate relationships between
        environmental and energy-system variables.

        - It is **not** a full physical or regulatory model.
        - It simplifies cause-effect links (e.g., heating, traffic, and weather effects on PM10).
        - Its purpose is to help students understand interactions between variables and practice
          scenario-based reasoning.
        """
    )

st.markdown("### Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("### Download Filtered Dataset")
if not filtered_df.empty:
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered scenarios as CSV",
        data=csv_data,
        file_name="filtered_smog_energy_scenarios.csv",
        mime="text/csv",
    )
else:
    st.caption("Run analysis with matching scenarios to enable download.")

show_stats = st.checkbox("Show raw statistics")
if show_stats:
    st.markdown("### Raw Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
