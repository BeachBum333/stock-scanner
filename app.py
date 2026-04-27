"""
app.py — Stock Strategy Scanner Dashboard
Swing + Day Trading strategies, Options Activity, Alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import io, smtplib
from email.mime.text import MIMEText

from strategies import (run_strategies, STRATEGY_MAP,
                        SWING_STRATEGIES, DAY_STRATEGIES, compute_indicators)
from data_fetcher  import fetch_stock_data, market_is_open
from universes     import get_universe
from options_scanner import render_options_panel, scan_options_for_list
from gamma_wall    import render_gamma_wall

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Strategy Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise session state ───────────────────────────────────────────────
for _k, _v in [("scan_results", []), ("scan_meta", {}), ("raw_data", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white; padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 1rem;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 0.2rem 0 0; opacity: 0.8; font-size: 0.95rem; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sound alert helper (JS) ────────────────────────────────────────────────
def _play_beep():
    st.components.v1.html("""
    <script>
    try {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      [800, 1000, 1200].forEach(function(freq, i) {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.15);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.15 + 0.14);
        osc.start(ctx.currentTime + i * 0.15);
        osc.stop(ctx.currentTime + i * 0.15 + 0.15);
      });
    } catch(e) {}
    </script>
    """, height=0)

# ── Email alert helper ─────────────────────────────────────────────────────
def _send_email_alert(signals: list, smtp_user: str, smtp_pass: str, to_addr: str):
    try:
        body = "📈 Stock Scanner Alerts\n\n"
        for r in signals:
            body += f"  {r['Ticker']}  ${r['Price']}  |  {r['Signal List']}\n"
        msg           = MIMEText(body)
        msg["Subject"]= f"📈 Scanner: {len(signals)} signal(s) found"
        msg["From"]   = smtp_user
        msg["To"]     = to_addr
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(smtp_user, smtp_pass)
            srv.send_message(msg)
        return True
    except Exception as e:
        st.sidebar.error(f"Email error: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Strategy Scanner")

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 Alpaca API Keys")
    default_key    = st.secrets.get("ALPACA_API_KEY", "")    if hasattr(st, "secrets") else ""
    default_secret = st.secrets.get("ALPACA_SECRET_KEY", "") if hasattr(st, "secrets") else ""
    api_key    = st.text_input("API Key",    value=default_key,    type="password", placeholder="PKxxx…")
    secret_key = st.text_input("Secret Key", value=default_secret, type="password", placeholder="xxx…")
    use_alpaca = bool(api_key and secret_key)
    if not use_alpaca:
        st.info("ℹ️ No keys → Yahoo Finance (EOD, free)")

    st.divider()

    # ── Universe ──────────────────────────────────────────────────────────────
    st.markdown("### 🌐 Universe")
    universe_choice = st.radio("Select universe", ["S&P 500", "NASDAQ 100", "Custom"])
    custom_tickers  = ""
    if universe_choice == "Custom":
        custom_tickers = st.text_area("Tickers (comma/space separated)",
                                      placeholder="AAPL NVDA TSLA META …", height=80)
    max_stocks = st.slider("Max stocks to scan", 10, 500, 100, 10)

    st.divider()

    # ── Swing Strategies ──────────────────────────────────────────────────────
    st.markdown("### 📊 Swing Strategies")
    swing_info = {
        "Minervini SEPA":   "Trend Template + VCP",
        "Bonde Momentum":   "Momentum Burst + Episodic Pivot",
        "Island Reversal":  "Bullish Island Reversal",
        "Gap Up/Down":      "Gap with high volume",
        "Failed Breakdown": "Bear trap / support reclaim",
        "Momentum Filter":  "Bull/bear multi-factor",
    }
    selected_swing = [n for n, d in swing_info.items()
                      if st.checkbox(n, value=True, help=d, key=f"sw_{n}")]

    st.divider()

    # ── Day Trading Strategies ────────────────────────────────────────────────
    st.markdown("### ⚡ Day Trading Strategies")
    day_info = {
        "VWAP Trend":          "Price vs rolling 20-day VWAP",
        "Opening Range (ORB)": "Gap + close near HOD/LOD",
        "High Rel Volume":     "Volume spike 3x+ average",
        "Gap & Go":            "Gap continues in gap direction",
        "Intraday Momentum":   "Multi-factor intraday score",
    }
    selected_day = [n for n, d in day_info.items()
                    if st.checkbox(n, value=True, help=d, key=f"dt_{n}")]

    selected_strategies = selected_swing + selected_day

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("### 🎚️ Filters")
    min_price   = st.number_input("Min Price ($)",  value=5.0,    step=1.0)
    min_vol     = st.number_input("Min Avg Volume", value=100000, step=50000, format="%d")
    min_signals = st.slider("Min # Signals", 1, 11, 1)

    st.divider()

    # ── Alert Settings ────────────────────────────────────────────────────────
    st.markdown("### 🔔 Alerts")
    alert_sound = st.toggle("🔊 Sound alert", value=True)
    alert_email = st.toggle("📧 Email alert", value=False)

    smtp_user, smtp_pass, alert_to = "", "", ""
    if alert_email:
        smtp_user = st.text_input("Gmail address", placeholder="you@gmail.com")
        smtp_pass = st.text_input("Gmail App Password", type="password",
                                  help="Generate at myaccount.google.com → Security → App Passwords")
        alert_to  = st.text_input("Send alerts to", value=smtp_user or "")
        st.caption("[How to get Gmail App Password](https://support.google.com/accounts/answer/185833)")

    alert_min_grade = st.selectbox("Alert on grade ≥", ["A", "B", "C"], index=1)

    st.divider()

    # ── Live Refresh ──────────────────────────────────────────────────────────
    st.markdown("### 🔄 Live Refresh")
    live_refresh = st.toggle("Auto-refresh data", value=False,
                             help="Automatically re-fetches and re-scans at the chosen interval")
    refresh_interval = 60
    if live_refresh:
        refresh_interval = st.select_slider(
            "Interval",
            options=[30, 60, 120, 300],
            value=60,
            format_func=lambda s: f"{s}s" if s < 60 else f"{s//60}m",
        )
        if _HAS_AUTOREFRESH:
            st_autorefresh(interval=refresh_interval * 1000, key="live_autorefresh")
            st.caption(f"⏱ Refreshing every {refresh_interval}s")
        else:
            st.warning("Install `streamlit-autorefresh` to enable live refresh.")

    st.divider()

    scan_clicked = st.button("🚀 Run Scan", type="primary", use_container_width=True)
    st.caption("Data: Alpaca / Yahoo Finance  |  ⚠️ Not financial advice")

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

is_open, market_status = market_is_open()
now_str = datetime.now().strftime("%A %b %d %Y  %H:%M:%S")

st.markdown(f"""
<div class="main-header">
  <h1>📈 Stock Strategy Scanner</h1>
  <p>{market_status} &nbsp;|&nbsp; {now_str} &nbsp;|&nbsp;
     {"Alpaca" if use_alpaca else "Yahoo Finance (EOD)"}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SCAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(universe, strategies, alpaca_key, alpaca_secret, p_min_price, p_min_vol):
    prog = st.progress(0, text="Fetching market data…")
    info = st.empty()

    info.info(f"⏳ Downloading data for {len(universe)} stocks…")
    raw = fetch_stock_data(universe, api_key=alpaca_key, secret_key=alpaca_secret,
                           period_days=400, use_alpaca=bool(alpaca_key and alpaca_secret))
    st.session_state.raw_data = raw
    prog.progress(30, text=f"Data for {len(raw)} stocks loaded. Running strategies…")

    filtered_symbols = []
    for sym, df in raw.items():
        try:
            if (df.iloc[-1]["close"] >= p_min_price and
                    df["volume"].rolling(20).mean().iloc[-1] >= p_min_vol):
                filtered_symbols.append(sym)
        except Exception:
            pass

    info.info(f"⚡ {len(filtered_symbols)} stocks passed filters. Scanning…")

    results, done = [], 0
    total = len(filtered_symbols)

    def _scan_one(sym):
        return run_strategies(sym, raw.get(sym), strategies)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_scan_one, s): s for s in filtered_symbols}
        for fut in as_completed(futures):
            done += 1
            prog.progress(int(30 + done / max(total, 1) * 65),
                          text=f"Scanned {done}/{total}…")
            try:
                row = fut.result()
                if row and row["# Signals"] > 0:
                    results.append(row)
            except Exception:
                pass

    prog.progress(100, text="Scan complete ✅")
    info.empty(); prog.empty()
    return results

# ── Trigger ────────────────────────────────────────────────────────────────

if scan_clicked:
    if not selected_strategies:
        st.error("Select at least one strategy.")
    else:
        universe = get_universe(universe_choice, custom_tickers)
        if not universe:
            st.error("No tickers found.")
        else:
            universe = universe[:max_stocks]
            st.session_state.scan_results = run_scan(
                universe, selected_strategies,
                api_key, secret_key, min_price, int(min_vol)
            )
            st.session_state.scan_meta = {
                "universe":   universe_choice,
                "total":      len(universe),
                "strategies": selected_strategies,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
            }

            # ── Alerts ─────────────────────────────────────────────────────
            results_found = st.session_state.scan_results
            grade_map     = {"A": 3, "B": 2, "C": 1}
            min_g         = grade_map.get(alert_min_grade, 2)

            def _row_max_grade(r):
                grades = []
                for sig in r.get("Signals", []):
                    if "[A]" in sig: grades.append(3)
                    elif "[B]" in sig: grades.append(2)
                    elif "[C]" in sig: grades.append(1)
                return max(grades) if grades else 0

            alert_rows = [r for r in results_found if _row_max_grade(r) >= min_g]

            if alert_rows:
                # Toast notifications
                for r in alert_rows[:5]:
                    st.toast(f"📈 {r['Ticker']}  ${r['Price']}  — {r['Signal List']}", icon="🚨")

                # Sound
                if alert_sound:
                    _play_beep()

                # Email
                if alert_email and smtp_user and smtp_pass and alert_to:
                    ok = _send_email_alert(alert_rows, smtp_user, smtp_pass, alert_to)
                    if ok:
                        st.success(f"📧 Email sent to {alert_to} with {len(alert_rows)} signals.")

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results = st.session_state.scan_results
meta    = st.session_state.scan_meta

if not results:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""**How to use:**\n1. Enter Alpaca API keys\n2. Pick universe\n3. Select strategies\n4. Click **Run Scan**""")
    c2.markdown("""**Swing:** Minervini SEPA · Bonde Momentum · Island Reversal · Gap Up/Down · Failed Breakdown · Momentum Filter""")
    c3.markdown("""**Day Trading:** VWAP · ORB · High Rel Vol · Gap & Go · Intraday Momentum\n\n**Options:** Unusual activity · PCR · IV spike""")
    st.stop()

filtered = [r for r in results if r["# Signals"] >= min_signals]

# ── Metrics ────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Scanned",       meta.get("total", 0))
m2.metric("Signals Found", len(results))
m3.metric("After Filter",  len(filtered))
m4.metric("Swing Hits",    sum(1 for r in filtered
                                for s in r["Signals"] if any(k in s for k in SWING_STRATEGIES)))
m5.metric("Day Trade Hits",sum(1 for r in filtered
                                for s in r["Signals"] if any(k in s for k in DAY_STRATEGIES)))
m6.metric("Scan Time",     meta.get("timestamp", "—"))

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_all, tab_entry, tab_day, tab_opts, tab_gamma, tab_chart, tab_detail, tab_guide = st.tabs([
    "📋 All Signals",
    "📍 Entry & Exit",
    "⚡ Day Trading",
    "🎯 Options Activity",
    "🧲 Gamma Wall",
    "📈 Chart",
    "🔍 Strategy Detail",
    "📚 Guide",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — ALL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

with tab_all:
    if not filtered:
        st.warning("No stocks matched. Try lowering Min # Signals.")
    else:
        rows = []
        for r in filtered:
            direction  = r.get("Direction", "—")
            dir_label  = "🟢 BULL" if direction == "BULL" else ("🔴 BEAR" if direction == "BEAR" else "—")
            # Best trade (first in _trades list, highest conviction)
            trades = r.get("_trades", [])
            best   = trades[0] if trades else {}
            rows.append({
                "Dir":       dir_label,
                "Ticker":    r["Ticker"],
                "Price":     r["Price"],
                "Chg %":     r["Chg%"],
                "Rel Vol":   r["RelVol"],
                "RSI":       r["RSI"],
                "Entry":     best.get("entry",    "—"),
                "Stop":      best.get("stop_loss","—"),
                "T1 (1.5R)": best.get("target_1", "—"),
                "T2 (2.5R)": best.get("target_2", "—"),
                "Risk %":    best.get("risk_%",   "—"),
                "# Signals": r["# Signals"],
                "Signals":   r["Signal List"],
            })

        df_disp = pd.DataFrame(rows).sort_values("# Signals", ascending=False)

        def colour_dir(val):
            if "BULL" in str(val): return "color: #16a34a; font-weight: bold"
            if "BEAR" in str(val): return "color: #ef4444; font-weight: bold"
            return ""

        def colour_chg(val):
            if isinstance(val, (int, float)):
                return "color: #16a34a" if val > 0 else "color: #dc2626" if val < 0 else ""
            return ""

        # Format numeric columns only where they are numeric
        num_cols = ["Price", "Chg %", "Rel Vol", "RSI", "Entry", "Stop",
                    "T1 (1.5R)", "T2 (2.5R)", "Risk %"]
        fmt = {}
        for col in num_cols:
            if col in df_disp.columns and pd.api.types.is_numeric_dtype(df_disp[col]):
                if col == "Price":       fmt[col] = "${:.2f}"
                elif col == "Chg %":     fmt[col] = "{:+.2f}%"
                elif col == "Rel Vol":   fmt[col] = "{:.1f}x"
                elif col == "RSI":       fmt[col] = "{:.0f}"
                elif col == "Risk %":    fmt[col] = "{:.2f}%"
                elif col in ("Entry","Stop","T1 (1.5R)","T2 (2.5R)"): fmt[col] = "${:.2f}"

        styler = df_disp.style.map(colour_chg, subset=["Chg %"]).map(colour_dir, subset=["Dir"])
        if fmt:
            styler = styler.format(fmt, na_rep="—")

        st.dataframe(styler, use_container_width=True, height=480)

        csv = io.StringIO()
        df_disp.to_csv(csv, index=False)
        st.download_button("⬇️ Download CSV", csv.getvalue(),
                           file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

        # Signal distribution chart
        strat_counts = {}
        for r in filtered:
            for sig in r["Signals"]:
                n = sig.split("[")[0].strip()
                strat_counts[n] = strat_counts.get(n, 0) + 1

        if strat_counts:
            colours = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6",
                       "#8b5cf6","#06b6d4","#f97316","#84cc16","#ec4899","#14b8a6"]
            fig = go.Figure(go.Bar(
                x=list(strat_counts.keys()), y=list(strat_counts.values()),
                marker_color=colours[:len(strat_counts)]
            ))
            fig.update_layout(template="plotly_dark", height=280,
                              margin=dict(l=10,r=10,t=20,b=10),
                              xaxis_title="Strategy", yaxis_title="# Stocks")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ENTRY & EXIT (Bull/Bear Direction + Trade Setup Cards)
# ══════════════════════════════════════════════════════════════════════════════

_ENTRY_CSS = """
<style>
.trade-card {
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    border-left: 5px solid #6366f1;
}
.bull-card  { border-left-color: #16a34a; background: rgba(22,163,74,0.07); }
.bear-card  { border-left-color: #ef4444; background: rgba(239,68,68,0.07); }
.card-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.2rem; }
.entry-note { font-size: 0.88rem; color: #86efac; margin-top: 0.3rem; }
.exit-note  { font-size: 0.88rem; color: #fca5a5; margin-top: 0.2rem; }
</style>
"""

with tab_entry:
    st.markdown(_ENTRY_CSS, unsafe_allow_html=True)

    if not filtered:
        st.info("Run a scan first to see entry/exit setups.")
    else:
        has_trades = [r for r in filtered if r.get("_trades")]
        if not has_trades:
            st.warning("No trade setups available — ensure at least one strategy is enabled.")
        else:
            # Stock selector
            tickers_with_trades = [r["Ticker"] for r in has_trades]
            sel_ee = st.selectbox("Select stock to view trade setups",
                                  tickers_with_trades, key="ee_sel")

            sel_ee_row = next((r for r in has_trades if r["Ticker"] == sel_ee), None)

            if sel_ee_row:
                direction = sel_ee_row.get("Direction", "—")
                dir_badge = (
                    "🟢 **BULLISH**" if direction == "BULL" else
                    "🔴 **BEARISH**"  if direction == "BEAR" else
                    "⚪ **NEUTRAL**"
                )
                st.markdown(
                    f"## {sel_ee}  &nbsp; ${sel_ee_row['Price']:.2f}"
                    f"  &nbsp;|&nbsp; {sel_ee_row['Chg%']:+.2f}%"
                    f"  &nbsp;|&nbsp; RSI {sel_ee_row['RSI']}"
                    f"  &nbsp;|&nbsp; Dominant Direction: {dir_badge}",
                    unsafe_allow_html=True
                )
                st.caption(f"Signals: {sel_ee_row['Signal List']}")
                st.divider()

                trades = sel_ee_row.get("_trades", [])
                for t in trades:
                    tdir   = t.get("direction", "BULL")
                    card_cls = "bull-card" if tdir == "BULL" else "bear-card"
                    dir_icon = "🟢" if tdir == "BULL" else "🔴"
                    grade    = t.get("grade", "")
                    ttype    = t.get("trade_type", "Swing")
                    strat    = t.get("strategy", "")

                    entry   = t.get("entry")
                    stop    = t.get("stop_loss")
                    t1      = t.get("target_1")
                    t2      = t.get("target_2")
                    t3      = t.get("target_3")
                    risk_p  = t.get("risk_%")
                    rr      = t.get("R:R", "1:2.5")

                    entry_note = t.get("entry_note", "")
                    exit_note  = t.get("exit_note", "")

                    st.markdown(
                        f'<div class="trade-card {card_cls}">'
                        f'<div class="card-title">{dir_icon} {strat} &nbsp;'
                        f'<span style="color:#a78bfa">[{grade}]</span>'
                        f' &nbsp;·&nbsp; <span style="opacity:.7;font-size:.85rem">{ttype}</span></div>'
                        f'<div class="entry-note">📌 {entry_note}</div>'
                        f'<div class="exit-note">🚪 {exit_note}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    def _fmt_price(v):
                        return f"${v:.2f}" if isinstance(v, (int, float)) else "—"
                    c1.metric("Entry",      _fmt_price(entry))
                    c2.metric("Stop Loss",  _fmt_price(stop),
                              delta=f"-{risk_p:.2f}%" if isinstance(risk_p, (int,float)) else None,
                              delta_color="inverse")
                    c3.metric("T1 (1.5R)",  _fmt_price(t1))
                    c4.metric("T2 (2.5R)",  _fmt_price(t2))
                    c5.metric("T3 (4R)",    _fmt_price(t3))
                    c6.metric("R:R",        rr)
                    st.markdown("---")

            # ── Summary table of all stocks ──────────────────────────────────────
            st.markdown("### 📊 All Setups — Summary")
            summary_rows = []
            for r in has_trades:
                for t in r.get("_trades", []):
                    tdir = t.get("direction", "BULL")
                    summary_rows.append({
                        "Dir":      "🟢 BULL" if tdir == "BULL" else "🔴 BEAR",
                        "Ticker":   r["Ticker"],
                        "Strategy": t.get("strategy", ""),
                        "Type":     t.get("trade_type", ""),
                        "Grade":    t.get("grade", ""),
                        "Entry":    t.get("entry"),
                        "Stop":     t.get("stop_loss"),
                        "T1":       t.get("target_1"),
                        "T2":       t.get("target_2"),
                        "Risk %":   t.get("risk_%"),
                    })

            if summary_rows:
                df_sum = pd.DataFrame(summary_rows)
                sum_fmt = {}
                for col in ("Entry", "Stop", "T1", "T2"):
                    if col in df_sum.columns and pd.api.types.is_numeric_dtype(df_sum[col]):
                        sum_fmt[col] = "${:.2f}"
                if "Risk %" in df_sum.columns and pd.api.types.is_numeric_dtype(df_sum["Risk %"]):
                    sum_fmt["Risk %"] = "{:.2f}%"

                def _dir_colour(v):
                    if "BULL" in str(v): return "color: #16a34a; font-weight:bold"
                    if "BEAR" in str(v): return "color: #ef4444; font-weight:bold"
                    return ""

                styler_sum = df_sum.style.map(_dir_colour, subset=["Dir"])
                if sum_fmt:
                    styler_sum = styler_sum.format(sum_fmt, na_rep="—")
                st.dataframe(styler_sum, use_container_width=True, height=380)

                csv2 = io.StringIO()
                df_sum.to_csv(csv2, index=False)
                st.download_button("⬇️ Download Setups CSV", csv2.getvalue(),
                                   file_name=f"setups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DAY TRADING SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

with tab_day:
    day_strat_names = list(DAY_STRATEGIES.keys())
    day_results = []
    for r in filtered:
        day_sigs = [s for s in r["Signals"] if any(k in s for k in day_strat_names)]
        if day_sigs:
            row = r.copy()
            row["Day Signals"] = ", ".join(day_sigs)
            row["# Day Sigs"]  = len(day_sigs)
            day_results.append(row)

    if not day_results:
        st.info("No day trading signals found. Run a scan with day trading strategies enabled.")
    else:
        st.markdown(f"### ⚡ {len(day_results)} Day Trading Setups")

        rows = []
        for r in day_results:
            rows.append({
                "Ticker":      r["Ticker"],
                "Price":       r["Price"],
                "Chg %":       r["Chg%"],
                "Rel Vol":     r["RelVol"],
                "RSI":         r["RSI"],
                "# DT Sigs":   r["# Day Sigs"],
                "DT Signals":  r["Day Signals"],
            })

        df_day = pd.DataFrame(rows).sort_values("# DT Sigs", ascending=False)

        def colour_chg(val):
            if isinstance(val, (int, float)):
                return "color: #16a34a" if val > 0 else "color: #dc2626" if val < 0 else ""
            return ""

        st.dataframe(
            df_day.style
                .map(colour_chg, subset=["Chg %"])
                .format({"Price": "${:.2f}", "Chg %": "{:+.2f}%",
                         "Rel Vol": "{:.1f}x", "RSI": "{:.0f}"}),
            use_container_width=True, height=420
        )

        # Tips
        st.info(
            "**Day Trading Tips:** VWAP and ORB are strongest near market open (9:30–11 AM ET). "
            "Gap & Go works best on stocks with float < 50M. "
            "High Rel Volume spikes > 5x are highest conviction. "
            "Always use a hard stop — day trades should be closed by EOD."
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — OPTIONS ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════

with tab_opts:
    st.markdown("### 🎯 Options Activity Scanner")

    opts_mode = st.radio("Mode", ["Single stock", "Batch scan signals"], horizontal=True)

    if opts_mode == "Single stock":
        all_tickers = [r["Ticker"] for r in filtered] if filtered else []
        manual_tick = st.text_input("Or type any ticker", placeholder="AAPL")
        choices     = ([manual_tick.upper()] if manual_tick.strip() else []) + all_tickers
        choices     = list(dict.fromkeys(choices))   # deduplicate, preserve order

        if choices:
            sel_opt = st.selectbox("Select ticker", choices)
            if st.button("Load Options Data", type="primary"):
                render_options_panel(sel_opt)
        else:
            st.info("Run a scan first or type a ticker above.")

    else:  # Batch mode
        batch_tickers = [r["Ticker"] for r in filtered][:30]  # max 30 to avoid rate limits
        if not batch_tickers:
            st.info("Run a scan first to populate the ticker list.")
        else:
            if st.button(f"Scan Options for Top {len(batch_tickers)} Stocks", type="primary"):
                with st.spinner("Scanning options chains… (this may take 30–60 seconds)"):
                    opt_df = scan_options_for_list(batch_tickers, max_workers=4)

                if opt_df.empty:
                    st.warning("No options data returned.")
                else:
                    st.dataframe(
                        opt_df.style.format({
                            "PCR":      "{:.2f}",
                            "ATM IV %": "{:.1f}%",
                            "Call Vol": "{:,.0f}",
                            "Put Vol":  "{:,.0f}",
                        }),
                        use_container_width=True, height=420
                    )

                    # Drill into a specific ticker
                    st.markdown("---")
                    sel2 = st.selectbox("Drill into ticker", opt_df["Ticker"].tolist())
                    render_options_panel(sel2)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — GAMMA WALL
# ══════════════════════════════════════════════════════════════════════════════

with tab_gamma:
    st.markdown("### 🧲 Gamma Wall & GEX Analysis")
    st.caption(
        "Gamma Exposure (GEX) shows where market-maker hedging pressure creates "
        "price walls, support zones, and volatility regime shifts. "
        "**Gamma Wall** = price magnet. **Gamma Flip** = low-vol / high-vol dividing line."
    )

    gex_col1, gex_col2 = st.columns([2, 1])
    with gex_col1:
        all_tickers_gex = [r["Ticker"] for r in filtered] if filtered else []
        manual_gex      = st.text_input("Or type any ticker", placeholder="AAPL", key="gex_manual")
        choices_gex     = ([manual_gex.upper()] if manual_gex.strip() else []) + all_tickers_gex
        choices_gex     = list(dict.fromkeys(choices_gex))
        sel_gex         = st.selectbox("Select ticker", choices_gex if choices_gex else ["—"], key="gex_sel")

    with gex_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_gex = st.button("📊 Load Gamma Wall", type="primary", key="gex_load")

    if load_gex and sel_gex and sel_gex != "—":
        render_gamma_wall(sel_gex)
    elif not load_gex:
        st.info("Select a ticker and click **Load Gamma Wall** to see GEX analysis.")

    # ── Quick Explainer ───────────────────────────────────────────────────────
    with st.expander("📚 What is Gamma Exposure? (click to expand)"):
        st.markdown("""
**Gamma** measures how much an option's delta changes as the stock price moves.
Market makers who sell options must delta-hedge continuously — and their hedging
*creates* price forces at key strikes.

| GEX Concept | What It Means for Price |
|---|---|
| **Positive Net GEX** | Market makers buy dips & sell rallies → price mean-reverts, low volatility |
| **Negative Net GEX** | Market makers chase price moves → volatility expands, trends accelerate |
| **Gamma Wall** | Highest positive GEX strike — strong resistance / price magnet |
| **Put Wall** | Highest negative GEX strike — support floor or, if broken, fast drop |
| **Gamma Flip** | Where cumulative GEX crosses zero — the vol regime dividing line |

**Trading with GEX:**
- *Above gamma flip + below gamma wall* → range trade, sell calls/puts
- *Below gamma flip* → expect big moves, trade breakouts with momentum
- *Near gamma wall* → fade the move (likely to reverse or pin)
- *Near put wall* → high conviction bounce level, or beware fast break

> GEX is most useful on large-cap, highly optioned stocks (AAPL, SPY, TSLA, NVDA, QQQ).
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — CHART
# ══════════════════════════════════════════════════════════════════════════════

with tab_chart:
    if not filtered:
        st.info("Run a scan first to see charts.")
    else:
        sel_tick = st.selectbox("Select stock", [r["Ticker"] for r in filtered], key="chart_sel")
        df_raw   = st.session_state.raw_data.get(sel_tick)

        if df_raw is not None and len(df_raw) > 30:
            df_c = compute_indicators(df_raw).tail(120)

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=[0.55,0.15,0.15,0.15],
                                subplot_titles=("Price + MAs + VWAP","Volume","RSI (14)","MACD"))

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_c.index, open=df_c["open"], high=df_c["high"],
                low=df_c["low"], close=df_c["close"], name="Price",
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
            ), row=1, col=1)

            # MAs
            for ma, col in [("ma20","#facc15"),("ma50","#60a5fa"),("ma200","#f87171")]:
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c[ma],
                    name=ma.upper(), line=dict(color=col, width=1.2), opacity=0.8), row=1, col=1)

            # VWAP
            if "vwap" not in df_c.columns:
                df_c["tp"]   = (df_c["high"] + df_c["low"] + df_c["close"]) / 3
                df_c["vwap"] = (df_c["tp"] * df_c["volume"]).rolling(20).sum() / df_c["volume"].rolling(20).sum()
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["vwap"],
                name="VWAP", line=dict(color="#a3e635", width=1.5, dash="dot")), row=1, col=1)

            # Volume
            vcol = ["#26a69a" if c >= o else "#ef5350"
                    for c, o in zip(df_c["close"], df_c["open"])]
            fig.add_trace(go.Bar(x=df_c.index, y=df_c["volume"],
                marker_color=vcol, name="Volume", opacity=0.7), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["vol_avg20"],
                name="Avg Vol", line=dict(color="#94a3b8", width=1, dash="dash")), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["rsi"],
                name="RSI", line=dict(color="#a78bfa", width=1.5)), row=3, col=1)
            for lv, lc in [(70,"#ef4444"),(50,"#94a3b8"),(30,"#22c55e")]:
                fig.add_hline(y=lv, line_dash="dot", line_color=lc, row=3, col=1, opacity=0.5)

            # MACD
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["macd"],
                name="MACD", line=dict(color="#38bdf8", width=1.2)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c["macd_sig"],
                name="Signal", line=dict(color="#fb923c", width=1.2)), row=4, col=1)
            hcol = ["#26a69a" if v >= 0 else "#ef5350" for v in df_c["macd_hist"]]
            fig.add_trace(go.Bar(x=df_c.index, y=df_c["macd_hist"],
                marker_color=hcol, name="Histogram", opacity=0.6), row=4, col=1)

            fig.update_layout(template="plotly_dark", height=750,
                              title=f"{sel_tick} — Last 120 Days",
                              xaxis_rangeslider_visible=False,
                              margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(fig, use_container_width=True)

            lat = compute_indicators(df_raw).iloc[-1]
            s1,s2,s3,s4,s5 = st.columns(5)
            s1.metric("Close",   f"${lat['close']:.2f}")
            s2.metric("RSI",     f"{lat['rsi']:.1f}")
            s3.metric("Rel Vol", f"{lat['rel_vol']:.1f}x")
            s4.metric("ATR",     f"${lat['atr']:.2f}")
            s5.metric("vs MA50", f"{((lat['close']/lat['ma50'])-1)*100:+.1f}%")
        else:
            st.warning(f"No chart data for {sel_tick}.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — STRATEGY DETAIL
# ══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    if not filtered:
        st.info("Run a scan first.")
    else:
        sel3 = st.selectbox("Select stock", [r["Ticker"] for r in filtered], key="det_sel")
        sel_row = next((r for r in filtered if r["Ticker"] == sel3), None)

        if sel_row:
            st.markdown(f"### {sel3} — ${sel_row['Price']:.2f}  |  {sel_row['Chg%']:+.2f}%  |  RSI {sel_row['RSI']}")
            st.markdown(f"**Signals:** {sel_row['Signal List']}")
            st.divider()

            for sname, res in sel_row.get("_detail", {}).items():
                if not res:
                    continue
                passed = res.get("passed", False)
                icon   = "✅" if passed else "❌"
                label  = f"{icon} **{sname}** — {res.get('score',0)}/{res.get('total',0)}  Grade: **{res.get('grade','—')}**"

                with st.expander(label, expanded=passed):
                    if res.get("error"):
                        st.error(res["error"]); continue
                    st.table(pd.DataFrame([
                        {"Check": k, "Result": "✅" if v else "❌"}
                        for k, v in res.get("detail", {}).items()
                    ]))
                    extras = {k: v for k, v in res.items()
                              if k not in ("passed","score","total","detail","grade","pattern","error")}
                    if extras:
                        ec = st.columns(min(len(extras), 4))
                        for i, (k, v) in enumerate(extras.items()):
                            ec[i % 4].metric(k, str(v))

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — STRATEGY GUIDE
# ══════════════════════════════════════════════════════════════════════════════

with tab_guide:
    st.markdown("""
## 📚 Strategy Reference Guide

---
### SWING STRATEGIES

**1. Minervini SEPA** — Trend Template + VCP. 9 conditions all passing. Grade A = full template + VCP tightening on low volume.

**2. Bonde Momentum** — Up ≥4% on volume > yesterday + 100K. Episodic Pivot (Grade A) = up ≥8% on 3x+ volume with catalyst.

**3. Island Reversal** — Gap down → tight island → gap up on 2x+ volume. Stop at island low. ~72% bullish historically.

**4. Gap Up/Down** — Open ≥3% from prior close with 1.5x+ relative volume. Day trade: ≥5% gap + 3x volume.

**5. Failed Breakdown** — Pierces support intraday, closes back above. RSI divergence confirms.

**6. Momentum Filter** — ≥7/9 bullish checks: MAs aligned, RSI 50–80, MACD bullish, near 52W high.

---
### DAY TRADING STRATEGIES

**7. VWAP Trend** — Price above/below rolling 20-day VWAP + RSI direction + rel vol ≥1.5x. Best used near market open.

**8. Opening Range Breakout (ORB)** — Gap + price closing in top/bottom 25% of day's range + 2x volume. Strongest 9:30–11 AM ET.

**9. High Rel Volume** — Volume ≥3x average with range expansion. Highest conviction at 5x+. Works best with float < 50M.

**10. Gap & Go** — Gap ≥3% that continues in gap direction (no fade). Grade A = ≥5% gap, stays above/below prior day's high/low.

**11. Intraday Momentum** — 7-factor: move >2%, rel vol ≥2x, strong candle body, MA trend, MACD, BB expansion, above prev close.

---
### OPTIONS SIGNALS

**Put/Call Ratio (PCR):** < 0.7 = bullish sentiment | > 1.3 = bearish | 0.7–1.3 = neutral.

**Unusual Activity (Vol/OI ≥ 3x):** Large money is betting on a move. Calls = bullish. Puts = bearish or hedge.

**IV Spike (>50%):** Big move expected. Options are expensive — buy spreads, not naked options.

**Top Strikes:** Highest volume strikes reveal where institutional money is positioned.

---
| Grade | Meaning |
|-------|---------|
| A | All key conditions met — highest quality |
| B | Strong setup — most conditions met |
| C | Developing — watch for confirmation |
| D | Weak — avoid |

*Not financial advice. Always use stop-losses and proper position sizing.*
""")
