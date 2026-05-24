import pandas as pd
import plotly.express as px
import streamlit as st
import re
import smtplib
from datetime import datetime
from email.message import EmailMessage
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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




def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


def _fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_report_pdf(context: dict) -> bytes:
    """Generate concise 2-page educational scenario summary PDF."""
    out = BytesIO()
    with PdfPages(out) as pdf:
        # PAGE 1: scenario summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        y = 0.96
        ax.text(0.5, y, "Environmental-Energy Scenario Summary", ha="center", va="top", fontsize=18, fontweight="bold")
        y -= 0.05

        ax.text(0.03, y, "Selected Environmental Conditions", fontsize=13, fontweight="bold")
        y -= 0.03
        for line in [
            f"Outdoor temperature: {context['temperature']} °C",
            f"Wind speed: {context['wind_speed']} m/s",
            f"Traffic intensity: {context['traffic_intensity']} %",
            f"Renewable energy share: {context['renewable_share']} %",
            f"Heating intensity: {context['heating']:.1f} %",
        ]:
            ax.text(0.05, y, f"• {line}", fontsize=10)
            y -= 0.024

        y -= 0.02
        ax.text(0.03, y, "Calculated Indicators", fontsize=13, fontweight="bold")
        y -= 0.03
        for line in [
            f"Estimated PM10 concentration: {context['avg_pm10']:.1f} µg/m³",
            f"CO emission index: {context['avg_co2']:.1f}",
            f"Energy demand index: {context['avg_energy']:.1f}",
            f"Smog risk category: {str(context['risk']).capitalize()}",
        ]:
            ax.text(0.05, y, f"• {line}", fontsize=10)
            y -= 0.024

        y -= 0.02
        ax.text(0.03, y, "Student Tasks", fontsize=13, fontweight="bold")
        y -= 0.03
        for row in context['task_rows']:
            ax.text(0.05, y, f"• {row[0]}: {row[1]}", fontsize=10)
            y -= 0.024

        y -= 0.02
        ax.text(0.03, y, "Student Interpretation", fontsize=13, fontweight="bold")
        y -= 0.03
        student_text = context.get('student_interpretation', '').strip() or 'No student interpretation provided.'
        ax.text(0.05, y, student_text, fontsize=10, wrap=True)

        ax.text(0.03, 0.02, "Educational environmental-energy simulator | Synthetic dataset for teaching purposes only", fontsize=8)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # PAGE 2: rule-based interpretation
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        y = 0.96
        ax.text(0.5, y, "Engineering Interpretation", ha='center', va='top', fontsize=18, fontweight='bold')
        y -= 0.06
        for bullet in context['rule_based_points'][:6]:
            ax.text(0.05, y, f"• {bullet}", fontsize=11)
            y -= 0.045

        ax.text(0.03, 0.02, "Educational environmental-energy simulator | Synthetic dataset for teaching purposes only", fontsize=8)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    out.seek(0)
    return out.read()


def send_report_email(receiver: str, pdf_bytes: bytes) -> tuple[bool, str]:
    import os
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_FROM", user or "")
    if not all([host, user, pwd, sender]):
        return False, "Email sending unavailable: set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM environment variables."
    try:
        msg = EmailMessage()
        msg["Subject"] = "Environmental-Energy System Analysis Report"
        msg["From"] = sender
        msg["To"] = receiver
        msg.set_content("Attached: generated educational environmental-energy report.")
        msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename="environmental_report.pdf")
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        return True, "Report sent successfully."
    except Exception as exc:
        return False, f"Email sending failed: {exc}"


def safe_pct_change(ref: float, new: float) -> float:
    if ref == 0:
        return 0.0
    return ((new - ref) / abs(ref)) * 100.0


def validate_interpretation(text: str) -> tuple[bool, str]:
    keywords = {"wind", "heating", "emission", "emissions", "renewable", "pm10", "dispersion", "traffic", "co2"}
    words = [w.strip(".,;:!?()[]{}\"\'").lower() for w in (text or "").split() if w.strip()]
    wc = len([w for w in words if w])
    hits = sum(1 for w in words if w in keywords)
    if wc < 20:
        return False, "Interpretation is too short (minimum 20 words)."
    if hits < 2:
        return False, "Interpretation should reference environmental mechanisms (e.g., wind, heating, emissions, renewable, PM10, dispersion)."
    return True, "Interpretation quality check passed."


def short_env_check(text: str, min_words: int = 12) -> tuple[bool, str]:
    terms = {"pm10", "wind", "heating", "traffic", "renewable", "emission", "dispersion", "temperature", "co2"}
    words = [w.strip(".,;:!?()[]{}\"\'").lower() for w in (text or "").split() if w.strip()]
    if len(words) < min_words:
        return False, f"Please provide at least {min_words} words."
    if not any(w in terms for w in words):
        return False, "Use environmental engineering terms (e.g., wind, heating, PM10, emissions, renewable, dispersion)."
    return True, "Good: interpretation includes engineering context."
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

    st.markdown("#### Introductory tasks (BASIC)")
    intro_note = st.text_area("Identify the main factors influencing PM10 and explain why wind speed affects accumulation.", key="intro_note")
    if st.button("Check Introduction Task"):
        ok, msg = short_env_check(intro_note, min_words=14)
        st.success("✅ Good introductory systems reasoning.") if ok else st.warning(f"⚠ {msg}")
    with st.expander("Hint / example answer"):
        st.markdown("PM10 is shaped by meteorology (wind, temperature), emission sources (traffic, heating), and energy mix (renewable share). Low wind reduces dispersion.")

with tab2:
    st.markdown("### Variables")
    a,b,c = st.columns(3)
    a.markdown("**Inputs:** temperature, wind, traffic, renewable share")
    b.markdown("**Model:** heating intensity, local emission, dispersion")
    c.markdown("**Outputs:** PM10, CO₂ index, demand, smog risk")
    st.dataframe(df.head(10), use_container_width=True)
    stats_df = df[["temperature","wind_speed","traffic_intensity","renewable_share","PM10","CO2_emission","energy_demand"]].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("#### EDA tasks")
    basic_answer = st.text_input("Which variable has the highest average value?")
    variability_answer = st.text_input("Which variable shows the highest variability (std)?")
    if st.button("Check Dataset Tasks"):
        means = stats_df.loc["mean"]
        stds = stats_df.loc["std"]
        mvar = means.idxmax().lower()
        svar = stds.idxmax().lower()
        score = 0
        score += 1 if mvar.split("_")[0] in basic_answer.lower() else 0
        score += 1 if svar.split("_")[0] in variability_answer.lower() else 0
        if score == 2:
            st.success("✅ Correctly identified central tendency and variability leaders.")
        else:
            st.warning(f"⚠ Score {score}/2. Re-check mean and std rows in the statistics table.")

    st.markdown("#### Interactive statistics tools")
    stat_var = st.selectbox("Select variable for distribution tools", ["PM10","CO2_emission","energy_demand","temperature","wind_speed","traffic_intensity","renewable_share"], key="stat_var")
    cstat1, cstat2 = st.columns(2)
    with cstat1:
        st.plotly_chart(px.histogram(df, x=stat_var, nbins=25, title=f"Histogram: {stat_var}"), use_container_width=True)
    with cstat2:
        st.plotly_chart(px.box(df, y=stat_var, title=f"Boxplot: {stat_var}"), use_container_width=True)

    stats_note = st.text_area("Interpret what these statistics and distributions mean for environmental behavior.", key="stats_note")

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

    st.markdown("#### Correlation analysis task")
    corr_cols = ["temperature","wind_speed","traffic_intensity","renewable_share","heating_intensity","PM10","CO2_emission","energy_demand"]
    corr = df[corr_cols].corr(method="spearman", numeric_only=True)

    heat = px.imshow(
        corr.round(2),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Spearman Correlation Heatmap",
    )
    heat.update_layout(height=650, coloraxis_colorbar=dict(title="Spearman r"))
    st.plotly_chart(heat, use_container_width=True)

    st.info("Correlation legend: Strong |r| > 0.7, Moderate 0.4–0.7, Weak < 0.4.")

    corr_no_diag = corr.where(~pd.DataFrame([[i == j for j in corr.columns] for i in corr.index], index=corr.index, columns=corr.columns))
    max_pos = corr_no_diag.stack().idxmax()
    max_neg = corr_no_diag.stack().idxmin()
    max_pos_val = corr_no_diag.stack().max()
    max_neg_val = corr_no_diag.stack().min()

    auto_lines = []
    if abs(corr.loc["heating_intensity", "PM10"]) > 0.7 and corr.loc["heating_intensity", "PM10"] > 0:
        auto_lines.append("Heating intensity shows strong positive association with PM10.")
    if abs(corr.loc["temperature", "energy_demand"]) > 0.7 and corr.loc["temperature", "energy_demand"] < 0:
        auto_lines.append("Temperature is strongly negatively associated with energy demand.")
    if not auto_lines:
        auto_lines.append("The heatmap shows mixed relationships; interpret correlations together with physical mechanisms.")
    for line in auto_lines:
        st.markdown(f"- {line}")

    st.markdown("**Student task:** Identify strongest positive and strongest negative correlation.")
    pos_ans = st.text_input("Strongest positive pair (e.g., heating_intensity-PM10)", key="corr_pos_ans")
    neg_ans = st.text_input("Strongest negative pair (e.g., temperature-energy_demand)", key="corr_neg_ans")
    rel_note = st.text_area("Explain one environmental mechanism visible in the heatmap.", key="rel_note")
    if st.button("Check Relationship Task"):
        ok, msg = short_env_check(rel_note, min_words=16)
        pos_ok = all(token in pos_ans.lower() for token in [max_pos[0].lower().split('_')[0], max_pos[1].lower().split('_')[0]])
        neg_ok = all(token in neg_ans.lower() for token in [max_neg[0].lower().split('_')[0], max_neg[1].lower().split('_')[0]])
        if ok and pos_ok and neg_ok:
            st.success(f"✅ Correct. Strongest positive: {max_pos[0]} vs {max_pos[1]} (r={max_pos_val:.2f}); strongest negative: {max_neg[0]} vs {max_neg[1]} (r={max_neg_val:.2f}).")
        elif not ok:
            st.warning(f"⚠ {msg}")
        else:
            st.warning(f"⚠ Re-check the extreme pairs. Current extremes are near r={max_pos_val:.2f} and r={max_neg_val:.2f}.")

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

        st.markdown("#### Decision-laboratory tasks")
        if st.button("Check Decision Task A: PM10<50 and demand<70"):
            st.success("✅ Constraint task achieved.") if (avg_pm10 < 50 and avg_energy < 70) else st.warning("⚠ Not achieved yet. Tune temperature/wind/traffic/renewables.")
        if st.button("Check Decision Task B: Exceedance<10 without CO₂ increase"):
            baseline_co2 = df["CO2_emission"].median()
            st.success("✅ Trade-off target achieved.") if (pm10_exceed_days < 10 and avg_co2 <= baseline_co2) else st.warning("⚠ Trade-off condition not yet met.")
        st.info("Engineering note: improving one indicator does not automatically optimize the entire environmental-energy system.")

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

    st.markdown("#### Sensitivity analysis laboratory")
    base = {"temperature": temperature, "wind_speed": wind_speed, "traffic_intensity": traffic_intensity, "renewable_share": renewable_share}
    param = st.selectbox("Change one parameter", ["temperature", "wind_speed", "traffic_intensity", "renewable_share"], key="sens_param")
    delta = st.slider("Delta", -20.0, 20.0, 2.0, 0.5, key="sens_delta")
    changed = base.copy()
    changed[param] = max(min(changed[param] + delta, {"temperature":25,"wind_speed":12,"traffic_intensity":100,"renewable_share":60}[param]), {"temperature":-15,"wind_speed":0,"traffic_intensity":0,"renewable_share":0}[param])
    base_df = get_nearest_scenarios(df, float(base["temperature"]), float(base["wind_speed"]), float(base["traffic_intensity"]), float(base["renewable_share"]), 50)
    ch_df = get_nearest_scenarios(df, float(changed["temperature"]), float(changed["wind_speed"]), float(changed["traffic_intensity"]), float(changed["renewable_share"]), 50)
    b_pm10, c_pm10 = base_df["PM10"].mean(), ch_df["PM10"].mean()
    b_co2, c_co2 = base_df["CO2_emission"].mean(), ch_df["CO2_emission"].mean()
    b_dem, c_dem = base_df["energy_demand"].mean(), ch_df["energy_demand"].mean()
    delta_tbl = pd.DataFrame({"Variable":["PM10","CO₂","Demand"],"Change":[f"{safe_pct_change(b_pm10,c_pm10):+.1f}%",f"{safe_pct_change(b_co2,c_co2):+.1f}%",f"{safe_pct_change(b_dem,c_dem):+.1f}%"]})
    st.dataframe(delta_tbl, use_container_width=True, hide_index=True)
    sens_note = st.text_area("Which indicator was most sensitive to this parameter change and why?", key="sens_note")

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

        pm10_diff = safe_pct_change(ls["PM10"], rs["PM10"])
        co2_diff = safe_pct_change(ls["CO2"], rs["CO2"])
        energy_diff = safe_pct_change(ls["energy"], rs["energy"])
        exceed_diff = rs["exceed"] - ls["exceed"]

        st.markdown("##### Comparison diagnostics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM10 difference [%]", f"{pm10_diff:+.1f}%")
        d2.metric("CO₂ difference [%]", f"{co2_diff:+.1f}%")
        d3.metric("Demand difference [%]", f"{energy_diff:+.1f}%")
        d4.metric("Exceedance-day difference", f"{exceed_diff:+d}")

        score = 0
        score += 1 if rs["PM10"] < 50 else 0
        score += 1 if rs["CO2"] < ls["CO2"] else 0
        score += 1 if rs["energy"] <= 70 else 0
        score += 1 if rs["exceed"] < ls["exceed"] else 0
        score += 1 if right_vals["renewable_share"] > left_vals["renewable_share"] else 0
        st.metric("Scenario sustainability score (RIGHT vs LEFT)", f"{score} / 5")

        preferred = st.radio("Which scenario is environmentally preferable?", [f"LEFT: {left_name}", f"RIGHT: {right_name}"], horizontal=True)
        adv_note = st.text_area("What does this comparison show about renewable transition and air quality?", key="adv_note")

        if st.button("Check ADVANCED task"):
            valid_text, valid_msg = validate_interpretation(adv_note)
            better_right = score >= 3
            student_right = preferred.startswith("RIGHT")
            if not valid_text:
                st.warning(f"⚠ {valid_msg}")
            elif (better_right and student_right) or ((not better_right) and (not student_right)):
                st.success("✅ Task completed: your choice and interpretation are consistent with multi-criteria sustainability assessment.")
                if "wind" in adv_note.lower() or "dispersion" in adv_note.lower():
                    st.info("Engineering feedback: your interpretation correctly considered meteorological dispersion effects.")
                elif "heating" in adv_note.lower() or "pm10" in adv_note.lower():
                    st.info("Engineering feedback: good link between heating demand and PM10 response.")
                else:
                    st.info("Engineering feedback: expand mechanistic explanation (wind, heating, emissions, renewables).")
            else:
                st.warning("⚠ Scenario choice does not match multi-criteria evidence. Reassess PM10, CO₂, demand, exceedance days, and renewable share together.")
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

        st.markdown("#### Final laboratory summary")
        final_conclusion = st.text_area("Write final engineering conclusions (environmental interpretation + sustainability assessment).", key="final_conclusion")

        st.markdown("#### Generate Environmental Report")
        eng_sentence = (
            "Low wind speed and increased heating demand contributed to elevated PM10 concentration."
            if wind_speed <= 2.5 and calculated_heating >= 60
            else "Estimated environmental response reflects interacting weather, traffic, heating, and renewable-energy conditions."
        )


        task_rows = [
            ["Basic: low PM10", "Completed" if avg_pm10 < 35 else "Not completed", basic_note or "(no student note)"],
            ["Intermediate: PM10<50 and demand<=70", "Completed" if (avg_pm10 < 50 and avg_energy <= 70) else "Not completed", inter_note or "(no student note)"],
            ["Advanced: renewable transition comparison", "Completed" if (score >= 3 and validate_interpretation(adv_note)[0]) else "Not completed", adv_note or "(no student note)"],
        ]

        rule_based_points = []
        if wind_speed <= 2.5:
            rule_based_points.append("Low wind speed increased pollutant accumulation tendency.")
        elif wind_speed >= 6:
            rule_based_points.append("Higher wind speed improved atmospheric dispersion conditions.")
        if calculated_heating >= 60:
            rule_based_points.append("Increased heating demand intensified PM10 formation risk.")
        if renewable_share >= 40:
            rule_based_points.append("Higher renewable-energy share reduced conventional-emission burden.")
        if traffic_intensity >= 70:
            rule_based_points.append("Traffic intensity contributed to elevated environmental pressure.")
        if temperature <= 0:
            rule_based_points.append("Low outdoor temperature increased heating-related environmental burden.")
        if not rule_based_points:
            rule_based_points.append("Estimated response reflects balanced environmental and energy-system conditions.")

        pdf_context = {
            "date_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "scenario_name": scenario_name,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "traffic_intensity": traffic_intensity,
            "renewable_share": renewable_share,
            "heating": calculated_heating,
            "avg_pm10": avg_pm10,
            "avg_co2": avg_co2,
            "avg_energy": avg_energy,
            "risk": dominant_risk,
            "pm10_exceed": pm10_exceed_days,
            "eng_sentence": eng_sentence,
            "student_interpretation": "BASIC: " + (basic_note or "") + "\nINTERMEDIATE: " + (inter_note or "") + "\nADVANCED: " + (adv_note or ""),
            "rule_based_points": rule_based_points,
            "task_rows": task_rows,
            "auto_lines": auto_lines,
        }
        pdf_bytes = build_report_pdf(pdf_context)
        st.download_button("Download Environmental Report (PDF)", data=pdf_bytes, file_name="environmental_report.pdf", mime="application/pdf")

        st.markdown("##### Send report by email")
        mail_to = st.text_input("Recipient email", placeholder="student@example.com")
        if st.button("Send Report to Email"):
            if not is_valid_email(mail_to):
                st.warning("Please enter a valid email address.")
            else:
                ok, msg = send_report_email(mail_to, pdf_bytes)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

st.caption("Educational note: synthetic dataset + simplified environmental-engineering logic for classroom reasoning.")
