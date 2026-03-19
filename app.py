"""
app.py — NovaPulse AI · Contactless Health Monitoring Dashboard
Streamlit entry point: webcam capture → rPPG processing → live dashboard + AI insights.

Performance optimizations:
  - UI updates throttled to ~10 fps (not every frame)
  - Plotly chart updates every 2 seconds (not every frame)
  - Frame processing skips when buffer is ahead
  - Unique keys on all Streamlit elements to avoid duplicate-ID crashes
"""

import time
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import deque

from bpm_engine import (
    detect_face_landmarks,
    get_forehead_roi,
    calc_bpm,
    calc_stress,
    calc_blink_rate,
    get_ear,
    calc_cni,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    EAR_THRESHOLD,
)
from ai_advisor import get_health_insight

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaPulse AI",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS injection ────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="stApp"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #060C14 !important;
    color: #E2E8F0;
}
header[data-testid="stHeader"],
footer, #MainMenu { display: none !important; }
section[data-testid="stSidebar"] { background-color: #0D1B2A !important; }
div[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }

.np-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 0 12px;
}
.np-brand {
    font-size: 28px; font-weight: 600; color: #E2E8F0;
    display: flex; align-items: center; gap: 10px;
}
.np-brand span { color: #00D9AA; }
.np-tagline { font-size: 13px; color: #94A3B8; margin-top: 2px; font-weight: 400; }
.np-status {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
}
.np-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.np-dot-live { background: #00D9AA; box-shadow: 0 0 6px #00D9AA; animation: livePulse 2s ease-in-out infinite; }
.np-dot-ready { background: #4B5563; }
@keyframes livePulse {
    0%, 100% { box-shadow: 0 0 6px #00D9AA; }
    50% { box-shadow: 0 0 14px #00D9AA, 0 0 28px rgba(0,217,170,0.3); }
}
.np-divider {
    height: 1px;
    background: linear-gradient(90deg, #00D9AA40, transparent);
    margin-bottom: 16px;
}

.metric-card {
    background: #111827;
    border: 1px solid rgba(0, 217, 170, 0.15);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.3s, box-shadow 0.3s;
    margin-bottom: 8px;
}
.metric-card:hover {
    border-color: rgba(0, 217, 170, 0.35);
    box-shadow: 0 0 20px rgba(0, 217, 170, 0.06);
}
.card-label {
    font-size: 12px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 1.5px; color: #94A3B8; margin-bottom: 6px;
}
.card-value { font-size: 52px; font-weight: 300; color: #00D9AA; line-height: 1.1; }
.card-unit { font-size: 14px; font-weight: 400; color: #94A3B8; margin-top: 2px; }
.card-sub { font-size: 11px; color: #4B5563; margin-top: 6px; font-weight: 400; }

.stress-relaxed  { color: #10B981 !important; }
.stress-moderate { color: #F59E0B !important; }
.stress-stressed { color: #EF4444 !important; }

.cni-bar-wrap {
    width: 100%; height: 10px; border-radius: 5px;
    background: #1E293B; margin-top: 10px; overflow: hidden;
}
.cni-bar-fill {
    height: 100%; border-radius: 5px;
    background: linear-gradient(90deg, #EF4444, #F59E0B, #00D9AA);
    transition: width 0.6s ease;
}

.buf-bar-wrap {
    width: 100%; height: 4px; border-radius: 2px;
    background: #1E293B; margin: 8px 0;
}
.buf-bar-fill {
    height: 100%; border-radius: 2px; background: #00D9AA;
    transition: width 0.3s ease;
}
.buf-label { font-size: 12px; color: #94A3B8; text-align: center; }
.instruction-text { font-size: 12px; color: #4B5563; text-align: center; margin-top: 4px; }

.ai-card {
    background: #111827;
    border: 1px solid rgba(0, 217, 170, 0.15);
    border-left: 3px solid #00D9AA;
    border-radius: 12px;
    padding: 16px 18px;
    font-size: 14px; line-height: 1.65;
    color: #E2E8F0;
}
.ai-card-header {
    font-size: 14px; font-weight: 600; color: #E2E8F0;
    margin-bottom: 4px;
}
.ai-card-sub { font-size: 11px; color: #4B5563; margin-bottom: 10px; }
.ai-card-body { font-style: italic; }
.ai-attribution { font-size: 11px; color: #4B5563; margin-top: 10px; }

.pulse-dots { display: flex; gap: 5px; justify-content: center; padding: 10px 0; }
.pulse-dots span {
    width: 8px; height: 8px; border-radius: 50%;
    background: #00D9AA; animation: pulseDot 1.4s infinite ease-in-out both;
}
.pulse-dots span:nth-child(1) { animation-delay: -0.32s; }
.pulse-dots span:nth-child(2) { animation-delay: -0.16s; }
@keyframes pulseDot {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.2); }
}

div[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
.sidebar-section { font-size: 13px; color: #94A3B8; line-height: 1.6; }
.sidebar-section h4 { color: #E2E8F0; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "green_buf": lambda: deque(maxlen=300),
    "bpm_history": lambda: deque(maxlen=60),
    "ear_history": lambda: deque(maxlen=300),
    "blink_count": lambda: 0,
    "ear_consec": lambda: 0,
    "ai_insight": lambda: None,
    "last_ai_call": lambda: 0.0,
    "monitoring": lambda: False,
    "history": lambda: [],
    "frame_ts": lambda: 0,
}
for key, factory in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = factory()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "featherless.ai API Key",
        type="password",
        placeholder="Get free key at featherless.ai",
        key="api_key_input",
    )
    st.markdown("---")
    use_fallback = st.toggle("Use fallback video", value=False)
    video_file = None
    if use_fallback:
        video_file = st.file_uploader("Upload fallback .mp4", type=["mp4"])
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-section">'
        "<h4>About NovaPulse</h4>"
        "NovaPulse uses <b>remote photoplethysmography (rPPG)</b> — detecting tiny "
        "colour changes in your skin caused by blood flow. These micro-fluctuations in "
        "the green channel of your webcam feed are processed to estimate heart rate, "
        "stress, and fatigue — no sensors required."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        '<div class="sidebar-section">'
        "<h4>How it works</h4>"
        "<ol>"
        "<li>📷 Webcam captures face at 30 fps</li>"
        "<li>🟢 Forehead ROI isolated via MediaPipe</li>"
        "<li>📈 Green-channel signal extracted &amp; filtered</li>"
        "<li>💓 Peak detection → BPM, HRV, blink rate</li>"
        "<li>🤖 AI interprets vitals via featherless.ai</li>"
        "</ol></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-section">'
        "<h4>featherless.ai Setup</h4>"
        "<ol>"
        "<li>Visit <b>featherless.ai</b></li>"
        "<li>Sign up for a free account</li>"
        "<li>Go to API Keys section</li>"
        "<li>Create a new key</li>"
        "<li>Paste it in the field above</li>"
        "</ol>"
        "<p style='color:#4B5563;font-size:11px'>Free tier: 100 req/day · Model: Llama 3.1 8B</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────
def _stress_class(label):
    return {"Relaxed": "stress-relaxed", "Moderate": "stress-moderate", "Stressed": "stress-stressed"}.get(label, "")


def _cni_label(cni):
    if cni >= 65:
        return "Optimal"
    elif cni >= 40:
        return "Moderate Strain"
    return "High Stress"


def _make_trend_chart(bpm_history):
    fig = go.Figure()
    y = list(bpm_history) if bpm_history else []
    fig.add_trace(
        go.Scatter(
            y=y,
            mode="lines",
            line=dict(color="#00D9AA", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(0,217,170,0.07)",
        )
    )
    fig.update_layout(
        title=dict(text="Heart Rate Trend", font=dict(size=14, color="#94A3B8", family="DM Sans")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,1)",
        yaxis=dict(range=[40, 140], gridcolor="#1E293B", tickfont=dict(color="#4B5563", size=11), title=""),
        xaxis=dict(showticklabels=False, gridcolor="#1E293B", title=""),
        margin=dict(l=40, r=16, t=36, b=20),
        height=240,
        showlegend=False,
    )
    return fig


def _card_html(icon, label, value, unit, sub, extra="", value_style=""):
    style = f' style="{value_style}"' if value_style else ""
    return (
        f'<div class="metric-card">'
        f'<div class="card-label">{icon} {label}</div>'
        f'<div class="card-value"{style}>{value}</div>'
        f'<div class="card-unit">{unit}</div>'
        f"{extra}"
        f'<div class="card-sub">{sub}</div>'
        f"</div>"
    )


# ── Header ───────────────────────────────────────────────────────────────────
is_live = st.session_state.get("monitoring", False)
dot_cls = "np-dot-live" if is_live else "np-dot-ready"
status_txt = "LIVE" if is_live else "READY"

st.markdown(
    f"""
<div class="np-header">
  <div>
    <div class="np-brand">💓 <span>NovaPulse</span> AI</div>
    <div class="np-tagline">Contactless health monitoring · powered by rPPG + featherless.ai</div>
  </div>
  <div class="np-status"><span class="np-dot {dot_cls}"></span> {status_txt}</div>
</div>
<div class="np-divider"></div>
""",
    unsafe_allow_html=True,
)

# ── Control buttons ──────────────────────────────────────────────────────────
ctrl = st.columns([1, 1, 6])
start_btn = ctrl[0].button("▶️ Start Monitoring", key="start_btn")
stop_btn = ctrl[1].button("⬛ Stop Monitoring", key="stop_btn")

# ── Layout placeholders ─────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")
with left_col:
    cam_ph = st.empty()
    buf_ph = st.empty()
    instr_ph = st.empty()
with right_col:
    hr_ph = st.empty()
    stress_ph = st.empty()
    blink_ph = st.empty()
    cni_ph = st.empty()

bot_l, bot_r = st.columns([2, 1], gap="large")
with bot_l:
    chart_ph = st.empty()
with bot_r:
    ai_ph = st.empty()

# Session history
if st.session_state.history:
    with st.expander("📋 Session History", expanded=False):
        for i, s in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(
                f"**Session {len(st.session_state.history) - i}** — "
                f"{s['time']} · BPM {s['bpm']} · {s['stress']} · CNI {s['cni']}"
            )
            if s.get("insight"):
                st.caption(s["insight"])


# ── Default UI ───────────────────────────────────────────────────────────────
def _show_defaults():
    cam_ph.markdown(
        '<div style="background:#111827;border-radius:16px;height:360px;'
        "display:flex;align-items:center;justify-content:center;"
        'color:#4B5563;font-size:14px;">📷 Press ▶️ Start Monitoring to begin</div>',
        unsafe_allow_html=True,
    )
    buf_ph.markdown(
        '<div class="buf-bar-wrap"><div class="buf-bar-fill" style="width:0%"></div></div>'
        '<div class="buf-label">Waiting for signal...</div>',
        unsafe_allow_html=True,
    )
    instr_ph.markdown(
        '<div class="instruction-text">Sit still · Good lighting · Face the camera</div>',
        unsafe_allow_html=True,
    )
    hr_ph.markdown(_card_html("❤️", "HEART RATE", "--", "BPM", "Normal range: 60–100"), unsafe_allow_html=True)
    stress_ph.markdown(_card_html("🧠", "STRESS LEVEL", "--", "", "HRV: -- ms"), unsafe_allow_html=True)
    blink_ph.markdown(_card_html("👁️", "BLINK RATE", "--", "/min", ""), unsafe_allow_html=True)
    cni_bar = '<div class="cni-bar-wrap"><div class="cni-bar-fill" style="width:0%"></div></div>'
    cni_ph.markdown(_card_html("⚡", "CARDIAC NEURO INDEX", "--", "/ 100", "", extra=cni_bar), unsafe_allow_html=True)
    chart_ph.plotly_chart(_make_trend_chart([]), use_container_width=True, key="chart_default")
    ai_ph.markdown(
        '<div class="ai-card">'
        '<div class="ai-card-header">🤖 AI Health Insight</div>'
        '<div class="ai-card-sub">powered by featherless.ai</div>'
        '<div class="pulse-dots"><span></span><span></span><span></span></div>'
        '<div style="text-align:center;color:#4B5563;font-size:12px;">Waiting for vitals…</div>'
        '<div class="ai-attribution">insights by featherless.ai · Llama 3.1</div>'
        "</div>",
        unsafe_allow_html=True,
    )


if not start_btn and not st.session_state.get("monitoring"):
    _show_defaults()


# ── Monitoring loop ──────────────────────────────────────────────────────────
if start_btn or st.session_state.get("monitoring"):
    st.session_state.monitoring = True

    cap = None
    tmp_path = None
    try:
        if use_fallback and video_file is not None:
            import tempfile, os

            tmp_path = os.path.join(tempfile.gettempdir(), "novapulse_fallback.mp4")
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())
            cap = cv2.VideoCapture(tmp_path)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if cap is None or not cap.isOpened():
            cam_ph.markdown(
                '<div class="metric-card" style="border-color:#EF444440;">'
                '<div class="card-label" style="color:#EF4444;">⚠️ WEBCAM ERROR</div>'
                '<div style="font-size:14px;color:#94A3B8;margin-top:8px;">'
                "Could not open webcam. Please check:<br>"
                "• Browser has camera permission<br>"
                "• No other app is using the camera<br>"
                "• Try refreshing the page"
                "</div></div>",
                unsafe_allow_html=True,
            )
            st.session_state.monitoring = False
            st.stop()

        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = fps_raw if fps_raw > 0 else 30.0

        # Tracking variables
        bpm = None
        stress_label = "Measuring..."
        rmssd = 0.0
        blink_rate = 0.0
        fatigue_label = "Measuring..."
        cni = None

        frame_count = 0
        last_ui_update = 0.0
        last_chart_update = 0.0
        chart_counter = 0

        # Use monotonic timestamp for MediaPipe VIDEO mode
        start_ts = int(time.time() * 1000)

        while cap.isOpened():
            if stop_btn:
                break

            ret, frame = cap.read()
            if not ret:
                if use_fallback:
                    break
                continue

            frame_count += 1
            now = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Monotonically increasing timestamp for VIDEO mode
            ts_ms = int(now * 1000) - start_ts + 1
            st.session_state.frame_ts = max(st.session_state.frame_ts + 1, ts_ms)

            # ── Face detection (every frame for data quality) ────────────
            lm = detect_face_landmarks(rgb, st.session_state.frame_ts)

            if lm is not None:
                # Forehead + cheek ROI
                green_val, roi_box = get_forehead_roi(frame, lm, w, h)
                if green_val is not None:
                    st.session_state.green_buf.append(green_val)
                    if roi_box:
                        x1, y1, x2, y2 = roi_box
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 217, 170), 1)

                # EAR
                ear_l = get_ear(lm, LEFT_EYE_IDX, w, h)
                ear_r = get_ear(lm, RIGHT_EYE_IDX, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                st.session_state.ear_history.append(ear_avg)

                if ear_avg < EAR_THRESHOLD:
                    st.session_state.ear_consec += 1
                else:
                    if 2 <= st.session_state.ear_consec <= 6:
                        st.session_state.blink_count += 1
                    st.session_state.ear_consec = 0

            # ── Compute metrics (every 3rd frame to reduce CPU) ──────────
            if frame_count % 3 == 0:
                green_list = list(st.session_state.green_buf)
                ear_list = list(st.session_state.ear_history)

                new_bpm = calc_bpm(green_list, fps)
                if new_bpm is not None:
                    bpm = new_bpm

                new_stress, new_rmssd = calc_stress(green_list, fps)
                if new_stress != "Measuring...":
                    stress_label = new_stress
                    rmssd = new_rmssd

                new_blink, new_fatigue = calc_blink_rate(ear_list, fps)
                if new_fatigue != "Measuring...":
                    blink_rate = new_blink
                    fatigue_label = new_fatigue

                if bpm is not None and rmssd > 0 and blink_rate > 0:
                    cni = calc_cni(bpm, rmssd, blink_rate)

                if bpm is not None:
                    st.session_state.bpm_history.append(bpm)

            # ── AI insight (throttled: once every 15s) ───────────────────
            if cni is not None and (now - st.session_state.last_ai_call) > 15:
                st.session_state.ai_insight = get_health_insight(
                    bpm, stress_label, cni, blink_rate, api_key
                )
                st.session_state.last_ai_call = now
                st.session_state.history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "bpm": bpm,
                    "stress": stress_label,
                    "cni": cni,
                    "insight": st.session_state.ai_insight,
                })

            # ── UI update (throttled to ~8 fps) ──────────────────────────
            if (now - last_ui_update) < 0.12:
                continue
            last_ui_update = now

            # Webcam feed
            cam_ph.image(rgb, channels="RGB", use_container_width=True)

            # Buffer bar
            buf_pct = min(100, int(len(st.session_state.green_buf) / 120 * 100))  # ~4s for "ready"
            buf_text = "Signal ready ✓" if buf_pct >= 100 else f"Collecting signal… {len(st.session_state.green_buf)/fps:.1f}s"
            buf_ph.markdown(
                f'<div class="buf-bar-wrap"><div class="buf-bar-fill" style="width:{buf_pct}%"></div></div>'
                f'<div class="buf-label">{buf_text}</div>',
                unsafe_allow_html=True,
            )
            instr_ph.markdown(
                '<div class="instruction-text">Sit still · Good lighting · Face the camera</div>',
                unsafe_allow_html=True,
            )

            # Metric cards
            bpm_display = f"{bpm}" if bpm else "--"
            hr_ph.markdown(
                _card_html("❤️", "HEART RATE", bpm_display, "BPM", "Normal range: 60–100"),
                unsafe_allow_html=True,
            )

            stress_cls = _stress_class(stress_label)
            s_display = stress_label if stress_label != "Measuring..." else "--"
            hrv_txt = f"HRV: {rmssd:.0f} ms" if rmssd > 0 else "HRV: -- ms"
            stress_ph.markdown(
                _card_html("🧠", "STRESS LEVEL", s_display, "", hrv_txt,
                           value_style=f"font-size:28px;font-weight:500" + (f";color:{'#10B981' if stress_label=='Relaxed' else '#F59E0B' if stress_label=='Moderate' else '#EF4444' if stress_label=='Stressed' else '#00D9AA'}" if s_display != "--" else "")),
                unsafe_allow_html=True,
            )

            b_display = f"{blink_rate}" if fatigue_label != "Measuring..." else "--"
            blink_ph.markdown(
                _card_html("👁️", "BLINK RATE", b_display, "/min", fatigue_label,
                           value_style="font-size:36px"),
                unsafe_allow_html=True,
            )

            cni_display = f"{cni}" if cni is not None else "--"
            cni_pct = cni if cni is not None else 0
            cni_lbl = _cni_label(cni) if cni is not None else ""
            cni_bar = f'<div class="cni-bar-wrap"><div class="cni-bar-fill" style="width:{cni_pct}%"></div></div>'
            cni_ph.markdown(
                _card_html("⚡", "CARDIAC NEURO INDEX", cni_display, "/ 100", cni_lbl, extra=cni_bar),
                unsafe_allow_html=True,
            )

            # ── Chart (throttled: every 2 seconds) ──────────────────────
            if (now - last_chart_update) > 2.0:
                last_chart_update = now
                chart_counter += 1
                chart_ph.plotly_chart(
                    _make_trend_chart(st.session_state.bpm_history),
                    use_container_width=True,
                    key=f"chart_{chart_counter}",
                )

            # AI insight panel
            if st.session_state.ai_insight:
                ai_ph.markdown(
                    '<div class="ai-card">'
                    '<div class="ai-card-header">🤖 AI Health Insight</div>'
                    '<div class="ai-card-sub">powered by featherless.ai</div>'
                    f'<div class="ai-card-body">{st.session_state.ai_insight}</div>'
                    '<div class="ai-attribution">insights by featherless.ai · Llama 3.1</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                ai_ph.markdown(
                    '<div class="ai-card">'
                    '<div class="ai-card-header">🤖 AI Health Insight</div>'
                    '<div class="ai-card-sub">powered by featherless.ai</div>'
                    '<div class="pulse-dots"><span></span><span></span><span></span></div>'
                    '<div style="text-align:center;color:#4B5563;font-size:12px;">Analysing your vitals…</div>'
                    '<div class="ai-attribution">insights by featherless.ai · Llama 3.1</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if cap is not None:
            cap.release()
        st.session_state.monitoring = False
