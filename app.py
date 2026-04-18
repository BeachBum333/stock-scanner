"""
app.py — Stock Strategy Scanner Dashboard
Deploy on Streamlit Cloud: https://streamlit.io/cloud

Strategies:
  • Minervini SEPA / Trend Template + VCP
  • Pradeep Bonde Momentum Burst + Episodic Pivot
  • Bullish Island Reversal
  • Gap Up / Gap Down
  • Failed Breakdown (Bear Trap)
  • Bullish / Bearish Momentum Filter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import io

# ── Local modules ──────────────────────────────────────────────────────────
from strategies import run_strategies, STRATEGY_MAP, compute_indicators
from data_fetcher import fetch_stock_data, market_is_open, fetch_premarket_bars
from universes import get_universe

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Strategy Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p  { margin: 0.2rem 0 0; opacity: 0.8; font-size: 0.95rem; }

/* Signal badges */
.badge-a { background:#16a34a; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold; }
.badge-b { background:#2563eb; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold; }
.badge-c { background:#d97706; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold; }
.badge-d { background:#6b7280; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold; }

/* Metric cards */
.metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    color: white;
}
/* Make sidebar look clean */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "scan_meta" not in st.session_state:
    st.session_state.scan_meta = {}
if "raw_data" not in st.session_state:
    st.session_state.raw_data = {}


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Strategy Scanner")

    # ── API Keys ─────────────────────────────────────────────────────────────
    st.markdown("### 🔑 Alpaca API Keys")
    st.caption("Get free keys at [alpaca.markets](https://alpaca.markets)")

    # Try secrets first (for cloud deploy), then let user type
    default_key    = st.secrets.get("ALPACA_API_KEY", "")    if hasattr(st, "secrets") else ""
    default_secret = st.secrets.get("ALPACA_SECRET_KEY", "") if hasattr(st, "secrets") else ""

    api_key    = st.text_input("API Key",    value=default_key,    type="password", placeholder="PKxxx…")
    secret_key = st.text_input("Secret Key", value=default_secret, type="password", placeholder="xxx…")
    use_alpaca = bool(api_key and secret_key)

    if not use_alpaca:
        st.info("ℹ️ No API keys → using Yahoo Finance (EOD data, free)")

    st.divider()

    # ── Universe ─────────────────────────────────────────────────────────────
    st.markdown("### 🌐 Stock Universe")
    universe_choice = st.radio(
        "Select universe",
        ["S&P 500", "NASDAQ 100", "Custom"],
        horizontal=False,
    )

    custom_tickers = ""
    if universe_choice == "Custom":
        custom_tickers = st.text_area(
            "Enter tickers (comma or space separated)",
            placeholder="AAPL, NVDA, TSLA, META ...",
            height=100,
        )

    # Max stocks to scan (avoids hitting rate limits)
    max_stocks = st.slider("Max stocks to scan", 10, 500, 100, 10)

    st.divider()

    # ── Strategies ───────────────────────────────────────────────────────────
    st.markdown("### 🧠 Strategies")
    selected_strategies = []
    strategy_info = {
        "Minervini SEPA":   "Trend Template + VCP pattern",
        "Bonde Momentum":   "Momentum Burst + Episodic Pivot",
        "Island Reversal":  "Bullish Island Reversal pattern",
        "Gap Up/Down":      "Significant gap with high volume",
        "Failed Breakdown": "Bear trap / support reclaim",
        "Momentum Filter":  "Multi-factor bull/bear momentum",
    }
    for name, desc in strategy_info.items():
        if st.checkbox(name, value=True, help=desc):
            selected_strategies.append(name)

    st.divider()

    # ── Filters ──────────────────────────────────────────────────────────────
    st.markdown("### 🎚️ Filters")
    min_price  = st.number_input("Min Price ($)",  value=5.0,   step=1.0)
    min_vol    = st.number_input("Min Avg Volume", value=100000, step=50000, format="%d")
    min_rel_vol= st.slider("Min Rel. Volume", 0.5, 5.0, 1.0, 0.1)
    min_signals= st.slider("Min # Signals",   1, 6, 1)

    st.divider()

    # ── Scan Button ──────────────────────────────────────────────────────────
    scan_clicked = st.button("🚀 Run Scan", type="primary", use_container_width=True)

    st.caption("Data: Alpaca Markets / Yahoo Finance")
    st.caption("⚠️ Not financial advice")


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

is_open, market_status = market_is_open()
now_str = datetime.now().strftime("%A, %B %d %Y  %H:%M:%S")

st.markdown(f"""
<div class="main-header">
  <h1>📈 Stock Strategy Scanner</h1>
  <p>{market_status} &nbsp;|&nbsp; {now_str} &nbsp;|&nbsp;
     Data: {"Alpaca (real-time/delayed)" if use_alpaca else "Yahoo Finance (EOD)"}</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SCAN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(universe: list, strategies: list, alpaca_key: str, alpaca_secret: str,
             p_min_price: float, p_min_vol: int, p_min_rel_vol: float):
    """Fetch data and run strategies; returns list of result dicts."""

    prog_bar   = st.progress(0, text="Fetching market data…")
    status_txt = st.empty()

    # 1. Fetch data ─────────────────────────────────────────────────────────
    status_txt.info(f"⏳ Downloading data for {len(universe)} stocks…")
    raw = fetch_stock_data(
        universe,
        api_key    = alpaca_key,
        secret_key = alpaca_secret,
        period_days= 400,
        use_alpaca = bool(alpaca_key and alpaca_secret),
    )
    st.session_state.raw_data = raw
    prog_bar.progress(30, text=f"Data loaded for {len(raw)} stocks. Running strategies…")

    # 2. Apply pre-scan price/volume filters ────────────────────────────────
    filtered_symbols = []
    for sym, df in raw.items():
        try:
            latest  = df.iloc[-1]
            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            if latest["close"] >= p_min_price and vol_avg >= p_min_vol:
                filtered_symbols.append(sym)
        except Exception:
            pass

    status_txt.info(f"⚡ {len(filtered_symbols)} stocks passed filters. Scanning strategies…")

    # 3. Run strategies concurrently ────────────────────────────────────────
    results    = []
    done_count = 0
    total      = len(filtered_symbols)

    def _scan_one(sym):
        df = raw.get(sym)
        return run_strategies(sym, df, strategies)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_scan_one, sym): sym for sym in filtered_symbols}
        for fut in as_completed(futures):
            done_count += 1
            pct = int(30 + (done_count / max(total, 1)) * 65)
            prog_bar.progress(pct, text=f"Scanned {done_count}/{total}…")
            try:
                row = fut.result()
                if row and row["# Signals"] > 0:
                    results.append(row)
            except Exception:
                pass

    prog_bar.progress(100, text="Scan complete ✅")
    status_txt.empty()
    prog_bar.empty()
    return results


# ── Trigger scan ──────────────────────────────────────────────────────────

if scan_clicked:
    if not selected_strategies:
        st.error("Please select at least one strategy.")
    else:
        universe = get_universe(universe_choice, custom_tickers)
        if not universe:
            st.error("No tickers found. Please check your selection.")
        else:
            universe = universe[:max_stocks]
            st.session_state.scan_results = run_scan(
                universe, selected_strategies,
                api_key, secret_key,
                min_price, int(min_vol), min_rel_vol,
            )
            st.session_state.scan_meta = {
                "universe":   universe_choice,
                "total":      len(universe),
                "strategies": selected_strategies,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
            }


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

results = st.session_state.scan_results
meta    = st.session_state.scan_meta

if not results:
    # ── Welcome / empty state ───────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    **How to use:**
    1. Enter your Alpaca API keys (sidebar)
    2. Choose a stock universe
    3. Select strategies
    4. Click **Run Scan**
    """)
    c2.markdown("""
    **Strategies available:**
    - 📊 Minervini SEPA Trend Template
    - ⚡ Bonde Momentum Burst
    - 🏝️ Island Reversal
    - 🚀 Gap Up / Gap Down
    - 🪤 Failed Breakdown
    - 📈 Momentum Filter
    """)
    c3.markdown("""
    **Deploy to cloud:**
    - Push folder to GitHub
    - Go to [share.streamlit.io](https://share.streamlit.io)
    - Connect repo → select `app.py`
    - Add secrets in Settings
    """)
    st.stop()

# ── Apply signal count filter ───────────────────────────────────────────────
filtered = [r for r in results if r["# Signals"] >= min_signals]

# ── Summary metrics ─────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🔍 Stocks Scanned",  meta.get("total", 0))
m2.metric("✅ Signals Found",   len(results))
m3.metric("🎯 After Filter",    len(filtered))
m4.metric("⏰ Scan Time",       meta.get("timestamp", "—"))
m5.metric("📋 Strategies Run",  len(meta.get("strategies", [])))

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_results, tab_chart, tab_detail, tab_guide = st.tabs([
    "📋 Scan Results",
    "📈 Chart View",
    "🔍 Strategy Detail",
    "📚 Strategy Guide",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    if not filtered:
        st.warning("No stocks matched all filters. Try lowering Min # Signals.")
    else:
        # Build display DataFrame
        rows = []
        for r in filtered:
            rows.append({
                "Ticker":    r["Ticker"],
                "Price":     r["Price"],
                "Chg %":     r["Chg%"],
                "Volume":    f"{r['Vol']:,}",
                "Rel Vol":   r["RelVol"],
                "RSI":       r["RSI"],
                "# Signals": r["# Signals"],
                "Signals":   r["Signal List"],
            })

        df_display = pd.DataFrame(rows).sort_values("# Signals", ascending=False)

        # Colour function for change column
        def colour_chg(val):
            if isinstance(val, (int, float)):
                colour = "color: #16a34a" if val > 0 else "color: #dc2626" if val < 0 else ""
                return colour
            return ""

        styled = df_display.style\
            .applymap(colour_chg, subset=["Chg %"])\
            .format({"Price": "${:.2f}", "Chg %": "{:+.2f}%", "Rel Vol": "{:.1f}x", "RSI": "{:.0f}"})

        st.dataframe(styled, use_container_width=True, height=500)

        # Download button
        csv_buf = io.StringIO()
        df_display.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv_buf.getvalue(),
            file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        # Signal distribution bar chart
        st.markdown("#### Signal Distribution by Strategy")
        strat_counts = {}
        for r in filtered:
            for sig in r["Signals"]:
                name = sig.split("[")[0].strip()
                strat_counts[name] = strat_counts.get(name, 0) + 1

        if strat_counts:
            fig_bar = go.Figure(go.Bar(
                x=list(strat_counts.keys()),
                y=list(strat_counts.values()),
                marker_color=["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#8b5cf6"],
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Strategy",
                yaxis_title="# Stocks",
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: CHART VIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_chart:
    if not filtered:
        st.info("Run a scan first to see charts.")
    else:
        ticker_choices = [r["Ticker"] for r in filtered]
        sel_ticker = st.selectbox("Select stock to chart", ticker_choices)

        df_raw = st.session_state.raw_data.get(sel_ticker)

        if df_raw is not None and len(df_raw) > 30:
            df_chart = compute_indicators(df_raw).tail(120)  # last 120 days

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.55, 0.15, 0.15, 0.15],
                subplot_titles=("Price + MAs", "Volume", "RSI (14)", "MACD"),
            )

            # ── Candlestick ──────────────────────────────────────────────
            fig.add_trace(go.Candlestick(
                x=df_chart.index,
                open=df_chart["open"],
                high=df_chart["high"],
                low=df_chart["low"],
                close=df_chart["close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ), row=1, col=1)

            # ── Moving Averages ──────────────────────────────────────────
            ma_config = [("ma20", "#facc15"), ("ma50", "#60a5fa"), ("ma200", "#f87171")]
            for col, colour in ma_config:
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart[col],
                    name=col.upper(), line=dict(color=colour, width=1.2),
                    opacity=0.8,
                ), row=1, col=1)

            # ── Volume bars ──────────────────────────────────────────────
            vol_colours = [
                "#26a69a" if c >= o else "#ef5350"
                for c, o in zip(df_chart["close"], df_chart["open"])
            ]
            fig.add_trace(go.Bar(
                x=df_chart.index, y=df_chart["volume"],
                marker_color=vol_colours, name="Volume", opacity=0.7,
            ), row=2, col=1)

            # Avg volume line
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["vol_avg20"],
                name="Avg Vol", line=dict(color="#94a3b8", width=1, dash="dash"),
            ), row=2, col=1)

            # ── RSI ──────────────────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["rsi"],
                name="RSI", line=dict(color="#a78bfa", width=1.5),
            ), row=3, col=1)
            for level, colour in [(70, "#ef4444"), (50, "#94a3b8"), (30, "#22c55e")]:
                fig.add_hline(y=level, line_dash="dot", line_color=colour,
                              row=3, col=1, opacity=0.5)

            # ── MACD ─────────────────────────────────────────────────────
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["macd"],
                name="MACD", line=dict(color="#38bdf8", width=1.2),
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["macd_sig"],
                name="Signal", line=dict(color="#fb923c", width=1.2),
            ), row=4, col=1)
            hist_colours = ["#26a69a" if v >= 0 else "#ef5350" for v in df_chart["macd_hist"]]
            fig.add_trace(go.Bar(
                x=df_chart.index, y=df_chart["macd_hist"],
                name="Histogram", marker_color=hist_colours, opacity=0.6,
            ), row=4, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=750,
                title=f"{sel_ticker} — Last 120 Days",
                showlegend=True,
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Quick stats strip
            latest = compute_indicators(df_raw).iloc[-1]
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Close",       f"${latest['close']:.2f}")
            s2.metric("RSI (14)",    f"{latest['rsi']:.1f}")
            s3.metric("Rel Vol",     f"{latest['rel_vol']:.1f}x")
            s4.metric("ATR",         f"${latest['atr']:.2f}")
            s5.metric("vs MA50",     f"{((latest['close']/latest['ma50'])-1)*100:+.1f}%")
        else:
            st.warning(f"No chart data available for {sel_ticker}.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: STRATEGY DETAIL DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    if not filtered:
        st.info("Run a scan first to see details.")
    else:
        sel2 = st.selectbox("Select stock", [r["Ticker"] for r in filtered], key="detail_sel")
        sel_row = next((r for r in filtered if r["Ticker"] == sel2), None)

        if sel_row:
            st.markdown(f"### {sel2}  —  ${sel_row['Price']:.2f}  |  {sel_row['Chg%']:+.2f}%  |  RSI {sel_row['RSI']}")
            st.markdown(f"**Active Signals:** {sel_row['Signal List']}")
            st.divider()

            detail = sel_row.get("_detail", {})
            for strat_name, res in detail.items():
                if not res:
                    continue
                passed = res.get("passed", False)
                grade  = res.get("grade", "—")
                score  = res.get("score", 0)
                total  = res.get("total", 0)
                checks = res.get("detail", {})
                error  = res.get("error")

                icon  = "✅" if passed else "❌"
                label = f"{icon} **{strat_name}** — Score: {score}/{total}  Grade: **{grade}**"

                with st.expander(label, expanded=passed):
                    if error:
                        st.error(f"Error: {error}")
                        continue

                    # Checks table
                    check_rows = [
                        {"Check": k, "Result": "✅ Pass" if v else "❌ Fail"}
                        for k, v in checks.items()
                    ]
                    st.table(pd.DataFrame(check_rows))

                    # Extra info
                    extras = {k: v for k, v in res.items()
                              if k not in ("passed", "score", "total", "detail",
                                           "grade", "pattern", "error")}
                    if extras:
                        ec = st.columns(min(len(extras), 4))
                        for i, (k, v) in enumerate(extras.items()):
                            ec[i % 4].metric(k, str(v))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: STRATEGY GUIDE
# ══════════════════════════════════════════════════════════════════════════════

with tab_guide:
    st.markdown("""
## 📚 Strategy Reference Guide

---

### 1. 📊 Minervini SEPA — Trend Template + VCP

Mark Minervini's **SEPA** (Specific Entry Point Analysis) system identifies Stage 2 uptrending stocks
forming tight Volatility Contraction Patterns (VCP) before a breakout.

**9 Trend Template Conditions (all must pass):**
- Price above MA50, MA150, MA200
- MA150 > MA200 | MA50 > MA150 & MA200
- MA200 trending up for ≥ 1 month
- Price ≥ 25% above 52-week low
- Price within 25% of 52-week high
- VCP: range contracting + volume drying up

**Grade A setup:** All 9 conditions + VCP tightening on low volume.

---

### 2. ⚡ Pradeep Bonde — Momentum Burst + Episodic Pivot

Bonde's method catches stocks moving in **3–5 day explosive bursts of 8–40%**.

**Momentum Burst (Grade B):** Up ≥4% | Volume > yesterday | Rel Vol ≥1.5x | Low-vol base before burst

**Episodic Pivot (Grade A):** Up ≥8% | Rel Vol ≥3x | Major catalyst (earnings beat, FDA, contract)

**Key insight:** Entry on Day 1 of the burst; exit Days 3–5 or trail the position.

---

### 3. 🏝️ Bullish Island Reversal

Forms after a downtrend in three steps:
1. **Gap Down** — exhaustion gap creates the island
2. **Consolidation** — tight range forms the island body (≤8% range)
3. **Gap Up** — breakaway gap on high volume closes the island

Stop loss: island's lowest low. Historically bullish ~72% of the time.

---

### 4. 🚀 Gap Up / Gap Down Scanner

**Gap Up (Bullish):** Open ≥3% above prior close | Rel Vol ≥1.5x | Near 52-week high | MACD bullish

**Gap Down (Bearish):** Open ≥3% below prior close | Rel Vol ≥1.5x | Breaking key support

**Day trading filter:** Gap ≥5% + float < 50M + rel vol ≥3x → trade opening range breakout.

---

### 5. 🪤 Failed Breakdown (Bear Trap)

When shorts get trapped below a key support level:
1. Price breaks below support by 0.7%+ (intraday)
2. Reverses and **closes back above** support
3. Closes in upper 40% of day's range
4. Volume confirms the recovery

The trapped sellers become fuel for the rally. Best when RSI diverges (price new low, RSI higher).

---

### 6. 📈 Momentum Filter (Bullish / Bearish)

Multi-factor trend confirmation scoring 9 checks:

**Bullish (≥7/9):** Price > MA20/50/200 | MA50 > MA200 | RSI 50–80 | MACD above signal and 0 |
Near 52-week high | High relative volume

**Bearish (≥7/9):** Inverted — price below all MAs | RSI 20–50 | MACD below signal and 0 | Near 52-week low

---

### ⚙️ Scanner Scoring

| Grade | Meaning |
|-------|---------|
| **A** | Highest quality setup — all key conditions met |
| **B** | Strong setup — most conditions met |
| **C** | Developing setup — watch for confirmation |
| **D** | Weak — avoid |

---
*Not financial advice. Always use stop-losses and proper position sizing.*
""")
