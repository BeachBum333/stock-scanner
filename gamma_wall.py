"""
gamma_wall.py — Gamma Exposure (GEX) / Gamma Wall calculator
Data source: yfinance (free, no API key needed)

What it computes
────────────────
  GEX per strike  = OI × Gamma × 100 × Spot² × 0.01   (in $-millions)
  Calls give +GEX (market-maker is long gamma → buys dips, sells rips)
  Puts  give −GEX (market-maker is short gamma → amplifies moves)
  Net GEX = Σ call_GEX − Σ put_GEX

Key levels
──────────
  Gamma Wall     — strike with highest positive net GEX (price magnet)
  Put Wall       — strike with most negative net GEX (support / acceleration)
  Gamma Flip     — strike where cumulative GEX crosses zero
                   Above flip → low volatility, mean-reverting
                   Below flip → high volatility, trending / accelerating
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from typing import Optional, Dict


# ─── Black-Scholes Gamma ──────────────────────────────────────────────────────

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0.0


# ─── Main computation ─────────────────────────────────────────────────────────

@st.cache_data(ttl=600)   # cache 10 minutes
def compute_gex(ticker: str, risk_free_rate: float = 0.05) -> Optional[Dict]:
    """
    Fetch options chain for nearest 4 expiries and compute GEX by strike.
    Returns a dict with DataFrames and summary stats, or None on failure.
    """
    try:
        import yfinance as yf
        from datetime import datetime, date

        tk    = yf.Ticker(ticker)
        info  = tk.fast_info
        spot  = getattr(info, "last_price", None) or getattr(info, "previous_close", None)

        if not spot or spot <= 0:
            return {"error": f"Could not get spot price for {ticker}"}

        expiries = tk.options
        if not expiries:
            return {"error": f"No options found for {ticker}"}

        today   = date.today()
        records = []

        for exp in expiries[:4]:           # nearest 4 expiries
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                T = max((exp_date - today).days / 365.0, 1 / 365)

                chain = tk.option_chain(exp)

                for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                    df = df.copy()
                    df["volume"]            = pd.to_numeric(df.get("volume", 0),            errors="coerce").fillna(0)
                    df["openInterest"]      = pd.to_numeric(df.get("openInterest", 0),      errors="coerce").fillna(0)
                    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", 0), errors="coerce").fillna(0)

                    for _, row in df.iterrows():
                        K     = float(row["strike"])
                        iv    = float(row["impliedVolatility"])
                        oi    = float(row["openInterest"])
                        if oi == 0 or iv == 0:
                            continue

                        gamma = _bs_gamma(spot, K, T, risk_free_rate, iv)
                        # GEX in $-millions: OI × gamma × 100 shares × spot² × 0.01 / 1e6
                        gex_raw = oi * gamma * 100 * spot * spot * 0.01
                        gex_m   = gex_raw / 1_000_000          # convert to $M

                        records.append({
                            "expiry":    exp,
                            "type":      opt_type,
                            "strike":    K,
                            "iv":        iv,
                            "oi":        oi,
                            "gamma":     gamma,
                            "gex_$M":    gex_m if opt_type == "call" else -gex_m,
                        })
            except Exception:
                continue

        if not records:
            return {"error": "No valid options data to compute GEX"}

        df_all = pd.DataFrame(records)

        # ── Aggregate by strike ───────────────────────────────────────────────
        gex_by_strike = (
            df_all.groupby("strike")["gex_$M"]
            .sum()
            .reset_index()
            .rename(columns={"gex_$M": "net_gex"})
            .sort_values("strike")
        )

        # ── Key levels ───────────────────────────────────────────────────────
        # Focus on strikes within ±20% of spot (relevant range)
        mask = (
            (gex_by_strike["strike"] >= spot * 0.80) &
            (gex_by_strike["strike"] <= spot * 1.20)
        )
        relevant = gex_by_strike[mask].copy()

        if relevant.empty:
            relevant = gex_by_strike.copy()

        gamma_wall_strike = float(relevant.loc[relevant["net_gex"].idxmax(), "strike"])
        put_wall_strike   = float(relevant.loc[relevant["net_gex"].idxmin(), "strike"])

        # Gamma Flip: cumulative GEX sorted by strike — find where it crosses 0
        relevant_sorted = relevant.sort_values("strike")
        cum_gex         = relevant_sorted["net_gex"].cumsum()
        flip_idx        = (cum_gex >= 0).idxmax() if (cum_gex < 0).any() else None
        gamma_flip      = float(relevant_sorted.loc[flip_idx, "strike"]) if flip_idx is not None else None

        total_call_gex  = float(df_all[df_all["type"] == "call"]["gex_$M"].sum())
        total_put_gex   = float(df_all[df_all["type"] == "put"]["gex_$M"].sum())
        net_total_gex   = total_call_gex + total_put_gex  # puts are already negative

        regime = (
            "🟢 Positive GEX — Low Volatility / Mean-Reverting"
            if net_total_gex >= 0 else
            "🔴 Negative GEX — High Volatility / Trending"
        )

        # Call/Put GEX by strike (for stacked bar)
        call_by_strike = (
            df_all[df_all["type"] == "call"]
            .groupby("strike")["gex_$M"].sum()
            .reset_index().rename(columns={"gex_$M": "call_gex"})
        )
        put_by_strike = (
            df_all[df_all["type"] == "put"]
            .groupby("strike")["gex_$M"].sum()
            .reset_index().rename(columns={"gex_$M": "put_gex"})
        )
        stacked = pd.merge(call_by_strike, put_by_strike, on="strike", how="outer").fillna(0)
        stacked = stacked[
            (stacked["strike"] >= spot * 0.80) &
            (stacked["strike"] <= spot * 1.20)
        ].sort_values("strike")

        return {
            "ticker":           ticker,
            "spot":             round(spot, 2),
            "gex_by_strike":    gex_by_strike,
            "stacked":          stacked,
            "gamma_wall":       round(gamma_wall_strike, 2),
            "put_wall":         round(put_wall_strike, 2),
            "gamma_flip":       round(gamma_flip, 2) if gamma_flip else None,
            "total_call_gex":   round(total_call_gex, 2),
            "total_put_gex":    round(total_put_gex, 2),
            "net_total_gex":    round(net_total_gex, 2),
            "regime":           regime,
            "expiries_used":    list(expiries[:4]),
        }

    except Exception as e:
        return {"error": str(e)}


# ─── Streamlit renderer ───────────────────────────────────────────────────────

def render_gamma_wall(ticker: str) -> None:
    """Full Gamma Wall panel — call inside a Streamlit tab."""

    with st.spinner(f"Computing GEX for {ticker}…"):
        data = compute_gex(ticker)

    if data is None:
        st.warning(f"No options data for {ticker}.")
        return
    if "error" in data:
        st.error(f"GEX error: {data['error']}")
        return

    spot        = data["spot"]
    gamma_wall  = data["gamma_wall"]
    put_wall    = data["put_wall"]
    gamma_flip  = data["gamma_flip"]
    regime      = data["regime"]
    stacked     = data["stacked"]

    # ── Header metrics ────────────────────────────────────────────────────────
    st.markdown(f"### 🧲 Gamma Wall — **{ticker}**  (Spot: **${spot:,.2f}**)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Regime",       regime.split("—")[0].strip(),
              help="Positive GEX → price mean-reverts. Negative GEX → price trends/accelerates.")
    c2.metric("Gamma Wall",   f"${gamma_wall:,.2f}",
              help="Strike with highest positive GEX — acts as a price ceiling/magnet")
    c3.metric("Put Wall",     f"${put_wall:,.2f}",
              help="Strike with most negative GEX — strong support or acceleration zone")
    c4.metric("Gamma Flip",   f"${gamma_flip:,.2f}" if gamma_flip else "—",
              help="Price above flip = low vol regime. Below = high vol/trending regime.")
    c5.metric("Net GEX ($M)", f"${data['net_total_gex']:,.1f}M")

    # ── Regime explanation ────────────────────────────────────────────────────
    if data["net_total_gex"] >= 0:
        st.success(
            f"**{regime}**  — Market makers are net long gamma. "
            f"They sell rallies and buy dips, compressing volatility. "
            f"Price tends to pin near **${gamma_wall}** (Gamma Wall). "
            f"Expect choppy, range-bound trading unless a catalyst breaks the wall."
        )
    else:
        st.warning(
            f"**{regime}**  — Market makers are net short gamma. "
            f"They must chase price moves to hedge, *amplifying* volatility. "
            f"Price can accelerate quickly through strikes. "
            f"Watch the Gamma Flip at **${gamma_flip}** — a break below is high-risk."
        )

    st.divider()

    # ── GEX Bar Chart ─────────────────────────────────────────────────────────
    if stacked.empty:
        st.info("Not enough data to plot GEX chart.")
        return

    fig = go.Figure()

    # Call GEX (positive, green)
    fig.add_trace(go.Bar(
        x=stacked["strike"],
        y=stacked["call_gex"],
        name="Call GEX",
        marker_color="#16a34a",
        opacity=0.85,
    ))

    # Put GEX (negative, red — already stored as negative values)
    fig.add_trace(go.Bar(
        x=stacked["strike"],
        y=stacked["put_gex"],
        name="Put GEX",
        marker_color="#ef4444",
        opacity=0.85,
    ))

    # Net GEX line
    net_by_strike = stacked["call_gex"] + stacked["put_gex"]
    fig.add_trace(go.Scatter(
        x=stacked["strike"],
        y=net_by_strike,
        name="Net GEX",
        line=dict(color="#facc15", width=2),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    # Spot price vertical line
    fig.add_vline(
        x=spot, line_dash="dash", line_color="#60a5fa", line_width=2,
        annotation_text=f"Spot ${spot:.2f}",
        annotation_position="top right",
        annotation_font_color="#60a5fa",
    )

    # Gamma Wall line
    fig.add_vline(
        x=gamma_wall, line_dash="dot", line_color="#4ade80", line_width=1.5,
        annotation_text=f"γ Wall ${gamma_wall:.2f}",
        annotation_position="top left",
        annotation_font_color="#4ade80",
    )

    # Put Wall line
    fig.add_vline(
        x=put_wall, line_dash="dot", line_color="#f87171", line_width=1.5,
        annotation_text=f"Put Wall ${put_wall:.2f}",
        annotation_position="bottom right",
        annotation_font_color="#f87171",
    )

    # Gamma Flip line
    if gamma_flip:
        fig.add_vline(
            x=gamma_flip, line_dash="longdash", line_color="#a78bfa", line_width=2,
            annotation_text=f"γ Flip ${gamma_flip:.2f}",
            annotation_position="top",
            annotation_font_color="#a78bfa",
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",
        height=480,
        title=f"{ticker} — Gamma Exposure by Strike (±20% of Spot, Nearest 4 Expiries)",
        xaxis_title="Strike Price ($)",
        yaxis_title="GEX ($M)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=60, b=40),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Interpretation table ──────────────────────────────────────────────────
    st.markdown("#### 📖 How to Read the Levels")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""
| Level | Price | Meaning |
|-------|-------|---------|
| **Gamma Wall** | ${gamma_wall:,.2f} | Strong resistance / price magnet |
| **Put Wall** | ${put_wall:,.2f} | Key support or breakdown accelerator |
| **Gamma Flip** | {"$"+f"{gamma_flip:,.2f}" if gamma_flip else "—"} | Vol regime dividing line |
| **Spot** | ${spot:,.2f} | Current price |
""")

    with col_r:
        above_flip = (gamma_flip is None) or (spot >= gamma_flip)
        st.markdown(f"""
**Current Regime:** {"🟢 Above Gamma Flip" if above_flip else "🔴 Below Gamma Flip"}

{"Price is in a **low-vol pinning zone**. Market makers suppress moves. Trade ranges and fades." if above_flip else "Price is in a **high-vol trending zone**. Market makers amplify moves. Trade breakouts and momentum."}

**Bias:**
- Price **above Gamma Wall** → likely to pull back
- Price **between Flip & Wall** → constructive, grind higher
- Price **below Gamma Flip** → caution, vol expansion likely
- Price **at Put Wall** → watch for either strong bounce or fast breakdown
""")

    # ── Raw GEX table (top 15 strikes) ───────────────────────────────────────
    with st.expander("📋 Raw GEX by Strike (top 15 nearest spot)"):
        df_show = (
            data["gex_by_strike"]
            .assign(dist=lambda d: abs(d["strike"] - spot))
            .sort_values("dist")
            .head(15)
            .drop(columns="dist")
            .sort_values("strike")
            .reset_index(drop=True)
        )
        st.dataframe(
            df_show.style.format({"strike": "${:.2f}", "net_gex": "${:.3f}M"})
                   .bar(subset=["net_gex"], color=["#ef4444", "#16a34a"]),
            use_container_width=True,
            height=320,
        )

    st.caption(f"Expiries used: {', '.join(data['expiries_used'])}  |  Data: yfinance  |  GEX cached 10 min")
