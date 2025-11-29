import os
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1) PAGE CONFIG & STYLES
# =========================================================

st.set_page_config(
    page_title="Real-Time Air Quality & Activity Planner",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main {
        padding: 1rem;
    }

    .main-header {
        background: linear-gradient(135deg, #020617 0%, #0ea5e9 35%, #22c55e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.35);
    }
    .header-title {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
    }
    .header-subtitle {
        color: rgba(226,232,240,0.95);
        font-size: 1.05rem;
        margin-top: 0.6rem;
        text-align: center;
    }

    .section-header {
        color: #0ea5e9;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.8rem 0 1rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #0ea5e9;
    }

    .status-card {
        padding: 1.1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.18);
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #0ea5e9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 0.8rem;
        transition: all 0.2s ease;
    }

    /* Dark sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 2) SESSION STATE INIT
# =========================================================

def init_state():
    ss = st.session_state
    if getattr(ss, "initialized", False):
        return

    ss.initialized = True

    # Player state
    ss.xp = 0
    ss.level = 1
    ss.best_score = None
    ss.event_log = []

    # AI model
    ss.risk_model = None
    ss.scaler = None

    # Keep last evaluation
    ss.last_result = None

init_state()

# =========================================================
# 3) BASIC HELPERS
# =========================================================

def log_event(msg, icon="📌"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, f"{ts} {icon} {msg}")
    st.session_state.event_log = st.session_state.event_log[:80]

def add_xp(amount, reason):
    st.session_state.xp += amount
    log_event(f"+{amount} XP — {reason}", "⭐")
    while st.session_state.xp >= 100:
        st.session_state.xp -= 100
        st.session_state.level += 1
        log_event(f"Leveled up to level {st.session_state.level}!", "🎉")

# =========================================================
# 4) AIR QUALITY DATA (REAL-TIME HOOK + FALLBACK SIM)
# =========================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # set this in your env if you want real data

def fetch_real_aqi(lat, lon):
    """
    Uses OpenWeather Air Pollution API if API key is provided.
    Docs: https://openweathermap.org/api/air-pollution
    """
    if not OPENWEATHER_API_KEY:
        return None, None  # signal: no real data

    try:
        url = (
            "https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        aqi_scale = data["list"][0]["main"]["aqi"]  # 1..5
        # map 1..5 to approximate PM2.5-based AQI
        mapping = {1: 30, 2: 60, 3: 110, 4: 160, 5: 220}
        aqi = mapping.get(aqi_scale, 80)
        return aqi, data
    except Exception as e:
        log_event(f"Error fetching real AQI: {e}", "⚠️")
        return None, None

def simulate_aqi(city_name):
    """
    Fallback: simulate a realistic AQI profile, stable per city per day.
    This keeps the app usable even without an API key or internet.
    """
    seed_str = city_name.lower() + datetime.now().strftime("%Y-%m-%d")
    random.seed(seed_str)
    base = random.randint(40, 140)  # moderate-ish base
    # Slight time-of-day effect: evenings sometimes worse
    hour = datetime.now().hour
    extra = 0
    if 17 <= hour <= 22:
        extra = random.randint(10, 40)
    elif 0 <= hour <= 5:
        extra = random.randint(-15, 10)
    return max(10, base + extra)

def get_current_aqi(city_name, lat, lon):
    """
    Try real API; if not available, use simulator.
    """
    aqi, raw = fetch_real_aqi(lat, lon)
    if aqi is not None:
        source = "Real API"
    else:
        aqi = simulate_aqi(city_name)
        raw = None
        source = "Simulated"

    return aqi, source, raw

def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "🟡", "#eab308"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#f97316"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#a855f7"
    else:
        return "Hazardous", "🟥", "#7f1d1d"

# Simple hourly AQI forecast: either from API (air_pollution/forecast) or simulated around current
def get_hourly_aqi_forecast(city_name, lat, lon, hours_ahead=12, current_aqi=None):
    """Return dict with times & AQI values for next hours."""
    times = [datetime.now() + timedelta(hours=i) for i in range(hours_ahead)]
    values = []

    base = current_aqi if current_aqi is not None else simulate_aqi(city_name)
    random.seed(city_name.lower() + datetime.now().strftime("%Y-%m-%d-forecast"))

    for i in range(hours_ahead):
        # simulate small drift
        drift = random.randint(-15, 15)
        # evenings slightly worse
        hr = times[i].hour
        if 18 <= hr <= 22:
            drift += random.randint(0, 20)
        values.append(max(10, base + drift))

    return {"times": times, "aqi": values}

# =========================================================
# 5) AI MODEL: RISK SCORE FROM AQI + ACTIVITY
# =========================================================

ACTIVITY_INTENSITY = {
    "Easy walk": 1,
    "Brisk walk": 2,
    "Jogging / Running": 3,
    "Outdoor work (construction / delivery)": 3,
    "Kids outdoor play": 2,
}

SENSITIVITY_LEVEL = {
    "Healthy adult": 1,
    "Child / Elderly": 2,
    "Asthma / Heart / Lung condition": 3,
}

def synthetic_risk_formula(aqi, duration_min, intensity_level, sensitivity_level):
    """
    Handmade 'true' risk model used to generate training labels.
    Risk ~ function of AQI, duration, intensity, sensitivity.
    Returns 0..1.
    """
    # Base risk from AQI (scaled 0..1 roughly)
    base = np.clip(aqi / 250, 0, 1.2)

    dur_factor = np.clip(duration_min / 120, 0, 1.5)  # 2 hours = 1
    intensity_factor = 0.6 + 0.4 * (intensity_level - 1)  # 0.6, 1.0, 1.4
    sens_factor = 0.7 + 0.5 * (sensitivity_level - 1)     # 0.7, 1.2, 1.7

    risk = base * (0.6 + 0.7 * dur_factor) * intensity_factor * sens_factor

    # extra non-linear penalty for very bad air
    if aqi > 150:
        risk *= 1.3
    if aqi > 200:
        risk *= 1.5

    return float(np.clip(risk, 0, 1.8))  # may exceed 1 but we'll clip later in training

def train_risk_model():
    """
    Generate synthetic training data and train a small RandomForest on it.
    This 'AI' learns the mapping: (AQI, duration, intensity, sensitivity) -> risk_score (0..1).
    """
    n = 800
    X = []
    y = []

    rng = np.random.default_rng(42)
    for _ in range(n):
        aqi = rng.integers(10, 280)
        dur = rng.integers(10, 180)  # 10 min – 3 hours
        intensity = rng.integers(1, 4)
        sensitivity = rng.integers(1, 4)

        r = synthetic_risk_formula(aqi, dur, intensity, sensitivity)
        X.append([aqi, dur, intensity, sensitivity])
        y.append(r)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=120, random_state=0, n_jobs=-1
    )
    model.fit(X_scaled, y)

    return model, scaler

if st.session_state.risk_model is None:
    model, scaler = train_risk_model()
    st.session_state.risk_model = model
    st.session_state.scaler = scaler

def predict_risk(aqi, duration_min, intensity_level, sensitivity_level):
    X = np.array([[aqi, duration_min, intensity_level, sensitivity_level]])
    X_scaled = st.session_state.scaler.transform(X)
    raw = st.session_state.risk_model.predict(X_scaled)[0]
    # map to 0..1
    return float(np.clip(raw, 0, 1))

def risk_label_and_color(risk):
    if risk < 0.25:
        return "Low", "🟢", "#22c55e"
    elif risk < 0.5:
        return "Moderate", "🟡", "#eab308"
    elif risk < 0.75:
        return "High", "🟠", "#f97316"
    else:
        return "Very High", "🔴", "#ef4444"

def compute_score(risk):
    """
    Score (0–100) inverted from risk.
    Low risk → high score.
    """
    return float(np.clip((1 - risk) * 100, 0, 100))

# =========================================================
# 6) SIDEBAR: USER INPUTS
# =========================================================

with st.sidebar:
    st.markdown("### 🌍 Location & Time")
    st.caption("Pick a city and plan an outdoor activity. The AI will rate your health risk.")

    city = st.text_input("City name (for simulation label)", value="Dammam")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat = st.number_input("Latitude", value=26.42, format="%.4f")
    with col_lon:
        lon = st.number_input("Longitude", value=50.11, format="%.4f")

    st.markdown("---")
    st.markdown("### 🚶 Activity Planner")

    activity_type = st.selectbox(
        "Activity type",
        list(ACTIVITY_INTENSITY.keys()),
        index=1,
        help="Different activities use different breathing rates.",
    )

    duration_min = st.slider(
        "Duration (minutes)",
        min_value=10,
        max_value=180,
        value=45,
        step=5,
        help="Longer exposure → more risk, especially with bad air.",
    )

    sensitivity = st.selectbox(
        "Health sensitivity",
        list(SENSITIVITY_LEVEL.keys()),
        index=0,
        help=(
            "People with asthma, heart, or lung disease, children and elderly are more vulnerable.\n"
            "The risk model gives them extra protection."
        ),
    )

    mask_used = st.checkbox(
        "Wearing a high-quality mask (e.g. N95/FFP2)",
        value=False,
        help="A good mask can significantly reduce inhaled pollution."
    )

    st.markdown("---")
    st.markdown("### 👤 Progress")
    st.metric("Level", st.session_state.level)
    st.metric("XP", st.session_state.xp)
    if st.session_state.best_score is not None:
        st.metric("Best Score", f"{st.session_state.best_score:.1f} / 100")

    st.markdown("---")
    st.caption("Hint: Try to design a plan with **Low or Moderate risk** but still realistic duration.")

# =========================================================
# 7) HEADER
# =========================================================

st.markdown(
    """
<div class="main-header">
  <h1 class="header-title">😷 Real-Time Air Quality & Activity Planner</h1>
  <p class="header-subtitle">
    Plan your outdoor activity based on air pollution levels. 
    An AI model estimates health risk from AQI, duration, activity intensity, and your sensitivity.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "💡 **Goal**: Learn how air quality + time + activity together change your health risk.\n\n"
    "- Step 1: Choose your city, activity, duration, and sensitivity.\n"
    "- Step 2: The app gets current AQI (real or simulated) and uses an AI model to estimate risk.\n"
    "- Step 3: It suggests **better times** to go outside if risk is high."
)

# =========================================================
# 8) FETCH CURRENT AQI
# =========================================================

aqi, aqi_source, raw_data = get_current_aqi(city, lat, lon)
cat, cat_icon, cat_color = aqi_category(aqi)

status_text = f"{cat_icon} {cat} (AQI {aqi})"

st.markdown(
    f"""
<div class="status-card" style="background:{cat_color}15; border: 2px solid {cat_color}; color:{cat_color};">
  Current air quality in <strong>{city}</strong>: {status_text}  
  <br><span style="font-size:0.85rem; opacity:0.85;">Source: {aqi_source}</span>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 9) RUN RISK EVALUATION
# =========================================================

st.markdown('<p class="section-header">▶️ Evaluate Your Plan</p>', unsafe_allow_html=True)
st.write("Click the button to evaluate the risk of **this specific plan** (activity + duration + your health).")

evaluate_btn = st.button("Evaluate Plan", type="primary")

if evaluate_btn:
    intensity_level = ACTIVITY_INTENSITY[activity_type]
    sensitivity_level = SENSITIVITY_LEVEL[sensitivity]

    # Adjust AQI if using a mask (simple effect)
    effective_aqi = aqi
    if mask_used:
        effective_aqi = int(aqi * 0.7)  # reduce 30%

    risk = predict_risk(
        effective_aqi,
        duration_min,
        intensity_level,
        sensitivity_level,
    )
    risk = float(np.clip(risk, 0, 1))
    risk_label, risk_icon, risk_color = risk_label_and_color(risk)
    score = compute_score(risk)

    if st.session_state.best_score is None or score > st.session_state.best_score:
        st.session_state.best_score = score

    st.session_state.last_result = {
        "aqi": aqi,
        "effective_aqi": effective_aqi,
        "risk": risk,
        "risk_label": risk_label,
        "risk_icon": risk_icon,
        "risk_color": risk_color,
        "score": score,
        "activity_type": activity_type,
        "duration_min": duration_min,
        "sensitivity": sensitivity,
        "mask_used": mask_used,
    }

    # XP rules: reward low risk with realistic duration
    if duration_min >= 30 and score >= 80:
        add_xp(60, "Designed an excellent low-risk plan")
    elif score >= 60:
        add_xp(40, "Designed a reasonably safe plan")
    else:
        add_xp(15, "Explored a risky plan (learning)")

    log_event(
        f"Evaluated plan: {activity_type}, {duration_min} min, {sensitivity} → Risk={risk_label}, Score={score:.1f}",
        "📊",
    )

# =========================================================
# 10) DISPLAY PLAN RISK + EXPLANATION
# =========================================================

if st.session_state.last_result is not None:
    r = st.session_state.last_result

    st.markdown('<p class="section-header">📊 Risk Summary for This Plan</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Official AQI", f"{r['aqi']}")
    with c2:
        if r["mask_used"]:
            st.metric("Effective AQI (with mask)", f"{r['effective_aqi']}")
        else:
            st.metric("Effective AQI", f"{r['effective_aqi']}")
    with c3:
        st.metric("Risk Level", f"{r['risk_icon']} {r['risk_label']}")
    with c4:
        st.metric("Safety Score", f"{r['score']:.1f} / 100")

    st.markdown(
        f"""
<div class="metric-card" style="border-left-color:{r['risk_color']};">
  <strong>Interpretation:</strong><br>
  - Activity: <strong>{r['activity_type']}</strong> for <strong>{r['duration_min']} minutes</strong><br>
  - Sensitivity: <strong>{r['sensitivity']}</strong><br>
  - Mask: {"✅ Yes" if r["mask_used"] else "❌ No"}<br><br>
  Overall health risk is: <strong style="color:{r['risk_color']};">{r['risk_icon']} {r['risk_label']}</strong>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 💬 How each parameter affected your risk:")

    bullets = []

    # AQI comment
    if r["aqi"] <= 50:
        bullets.append("• Air quality is **Good** — healthiest time to be outside.")
    elif r["aqi"] <= 100:
        bullets.append("• Air quality is **Moderate** — still okay for most people, but sensitive groups should be cautious.")
    elif r["aqi"] <= 150:
        bullets.append("• Air quality is **Unhealthy for Sensitive Groups** — if you are in a risk group, short and light activity is better.")
    else:
        bullets.append("• Air quality is **Unhealthy or worse** — outdoor activity can be harmful, especially for children and people with health conditions.")

    # Duration comment
    if r["duration_min"] <= 30:
        bullets.append("• Duration is **short** (≤ 30 min): good for keeping exposure limited.")
    elif r["duration_min"] <= 90:
        bullets.append("• Duration is **medium** (30–90 min): risk grows with time, especially at high AQI.")
    else:
        bullets.append("• Duration is **long** (> 90 min): long exposure in bad air is usually a bad idea.")

    # Activity intensity comment
    intensity_level = ACTIVITY_INTENSITY[r["activity_type"]]
    if intensity_level == 1:
        bullets.append("• Activity intensity is **low** (easy walk) → gentlest on your lungs.")
    elif intensity_level == 2:
        bullets.append("• Activity intensity is **medium** (brisk walk / kids play) → you breathe deeper, pulling in more pollutants.")
    else:
        bullets.append("• Activity intensity is **high** (jogging / heavy work) → you inhale a lot more air (and pollution) per minute.")

    # Sensitivity comment
    sens_level = SENSITIVITY_LEVEL[r["sensitivity"]]
    if sens_level == 1:
        bullets.append("• You selected **Healthy adult**: baseline risk.")
    elif sens_level == 2:
        bullets.append("• You selected **Child / Elderly**: model increases risk because their bodies are more vulnerable.")
    else:
        bullets.append("• You selected **Asthma / Heart / Lung condition**: even moderate pollution can trigger issues, so risk is amplified.")

    # Mask comment
    if r["mask_used"]:
        bullets.append("• ✅ Wearing a proper mask reduces **effective AQI** and helps keep risk lower.")
    else:
        bullets.append("• ❌ No mask: your lungs take the full AQI.")

    for b in bullets:
        st.write(b)

# =========================================================
# 11) FIND SAFER TIME SLOTS (NEXT HOURS)
# =========================================================

st.markdown('<p class="section-header">⏰ Safer Time Suggestions</p>', unsafe_allow_html=True)
st.write(
    "We look at the next few hours and estimate how risky the **same activity** would be at each time.\n"
    "This lets you **shift the activity** to a better time instead of canceling it."
)

forecast = get_hourly_aqi_forecast(city, lat, lon, hours_ahead=12, current_aqi=aqi)
times = forecast["times"]
aqi_values = forecast["aqi"]

intensity_level = ACTIVITY_INTENSITY[activity_type]
sensitivity_level = SENSITIVITY_LEVEL[sensitivity]

rows = []
for t, a in zip(times, aqi_values):
    eff_a = a
    if mask_used:
        eff_a = int(a * 0.7)
    r = predict_risk(eff_a, duration_min, intensity_level, sensitivity_level)
    r = float(np.clip(r, 0, 1))
    label, icon, _ = risk_label_and_color(r)
    rows.append(
        {
            "Time": t.strftime("%H:%M"),
            "AQI": a,
            "Effective AQI": eff_a,
            "Risk Score": r,
            "Risk Level": f"{icon} {label}",
        }
    )

df_forecast = pd.DataFrame(rows)
st.dataframe(df_forecast.style.format({"Risk Score": "{:.2f}"}), use_container_width=True)

# Simple suggestion: pick earliest time with Low or Moderate risk
best_slot = None
for row in rows:
    if row["Risk Score"] < 0.5:  # low or moderate approx
        best_slot = row
        break

st.markdown("#### 🧭 Recommendation:")

if best_slot is None:
    st.warning(
        "All the next 12 hours have **High or Very High** estimated risk for this activity. "
        "If possible, shorten duration, switch to indoor, or choose a different day."
    )
else:
    st.success(
        f"Better plan: start around **{best_slot['Time']}** when AQI ≈ {best_slot['AQI']} "
        f"and estimated risk is **{best_slot['Risk Level']}** for the same activity & duration."
    )

# Plot AQI + risk over time
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[t.strftime("%H:%M") for t in times],
        y=aqi_values,
        name="AQI",
        mode="lines+markers",
    )
)
fig2_axis = [r["Risk Score"] for r in rows]
fig.add_trace(
    go.Scatter(
        x=[t.strftime("%H:%M") for t in times],
        y=[s * 300 for s in fig2_axis],  # scale risk to overlay visually (0–300)
        name="Risk (scaled)",
        mode="lines",
        line=dict(dash="dash"),
    )
)
fig.update_layout(
    xaxis_title="Time (next hours)",
    yaxis_title="AQI / Risk (scaled)",
    height=320,
    margin=dict(l=10, r=10, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 12) EVENT LOG
# =========================================================

st.markdown('<p class="section-header">📜 Event Log</p>', unsafe_allow_html=True)
if st.session_state.event_log:
    for e in st.session_state.event_log[:12]:
        st.write(e)
else:
    st.write("No events yet. Evaluate a plan to see logs here.")
