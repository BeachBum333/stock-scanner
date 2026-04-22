"""
strategies.py — All stock scanning strategy implementations.
Each strategy returns direction (BULL/BEAR) + entry, stop, targets, R:R.

Strategies:
  SWING  — Minervini SEPA, Bonde Momentum, Island Reversal,
            Gap Up/Down, Failed Breakdown, Momentum Filter
  DAY    — VWAP Trend, Opening Range Breakout, High Rel Volume,
            Gap & Go, Intraday Momentum
"""

import pandas as pd
import numpy as np
from typing import Dict

# ═══════════════════════════════════════════════════════════
#  INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    df["ma10"]  = close.rolling(10).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["ma50"]  = close.rolling(50).mean()
    df["ma150"] = close.rolling(150).mean()
    df["ma200"] = close.rolling(200).mean()

    df["vol_avg20"] = vol.rolling(20).mean()
    df["rel_vol"]   = vol / df["vol_avg20"].replace(0, np.nan)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12           = close.ewm(span=12, adjust=False).mean()
    ema26           = close.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    bb_mid          = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    df["high_52w"] = high.rolling(252).max()
    df["low_52w"]  = low.rolling(252).min()

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low  - close.shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["pct_chg"] = close.pct_change()
    df["gap_pct"] = (df["open"] - close.shift()) / close.shift()

    # VWAP (20-day rolling)
    df["tp"]   = (high + low + close) / 3
    df["vwap"] = (df["tp"] * vol).rolling(20).sum() / vol.rolling(20).sum()

    return df


# ═══════════════════════════════════════════════════════════
#  ENTRY / EXIT CALCULATOR
# ═══════════════════════════════════════════════════════════

def _calc_trade(direction: str, entry: float, stop: float,
                atr: float, trade_type: str = "Swing") -> dict:
    """
    Given entry and stop, compute targets at 1.5R, 2.5R, 4R.
    Returns a dict ready to merge into a strategy result.
    """
    risk = abs(entry - stop)
    if risk < 0.01:
        risk = max(atr * 0.5, 0.01)

    mult = 1 if direction == "BULL" else -1

    t1 = round(entry + mult * risk * 1.5, 2)
    t2 = round(entry + mult * risk * 2.5, 2)
    t3 = round(entry + mult * risk * 4.0, 2)

    pct_risk   = round(risk / entry * 100, 2)
    pct_t1     = round(abs(t1 - entry) / entry * 100, 2)

    return {
        "direction":  direction,
        "entry":      round(entry, 2),
        "stop_loss":  round(stop, 2),
        "target_1":   t1,
        "target_2":   t2,
        "target_3":   t3,
        "risk_$":     round(risk, 2),
        "risk_%":     pct_risk,
        "reward_%_T1":pct_t1,
        "R:R":        "1:2.5",
        "trade_type": trade_type,
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 1 — Minervini SEPA (BULL only)
# ═══════════════════════════════════════════════════════════

def minervini_sepa(df: pd.DataFrame, min_rs: float = 70.0) -> Dict:
    if len(df) < 252:
        return {"passed": False, "score": 0, "detail": {}, "grade": "N/A",
                "pattern": "Minervini SEPA", "direction": "BULL"}

    latest   = df.iloc[-1]
    price    = latest["close"]
    ma50, ma150, ma200 = latest["ma50"], latest["ma150"], latest["ma200"]
    high_52w, low_52w  = latest["high_52w"], latest["low_52w"]
    atr      = latest["atr"]

    ma200_1mo   = df["ma200"].iloc[-22] if len(df) >= 22 else np.nan
    ma200_up    = bool(ma200 > ma200_1mo) if not np.isnan(ma200_1mo) else False

    last10_range = (df["high"].iloc[-10:] - df["low"].iloc[-10:]).mean()
    prior_range  = (df["high"].iloc[-30:-10] - df["low"].iloc[-30:-10]).mean()
    vcp_tighten  = bool(last10_range < prior_range * 0.7) if prior_range > 0 else False

    # Pivot = highest high of last 10 bars (buy on breakout)
    pivot   = round(df["high"].iloc[-10:].max(), 2)
    vcp_low = round(df["low"].iloc[-10:].min(), 2)

    checks = {
        "Price > MA50":           bool(price > ma50),
        "Price > MA150":          bool(price > ma150),
        "Price > MA200":          bool(price > ma200),
        "MA150 > MA200":          bool(ma150 > ma200),
        "MA200 Trending Up":      ma200_up,
        "MA50 > MA150 & MA200":   bool(ma50 > ma150 and ma50 > ma200),
        ">=25% Above 52W Low":    bool(price >= low_52w  * 1.25),
        "Within 25% of 52W High": bool(price >= high_52w * 0.75),
        "VCP Tightening":         vcp_tighten,
    }

    score  = sum(checks.values())
    passed = all(list(checks.values())[:8])
    grade  = "A" if score >= 9 else "B" if score >= 7 else "C" if score >= 5 else "D"

    trade = _calc_trade("BULL", pivot, vcp_low, atr, "Swing")

    return {
        "passed":  passed, "score": score, "total": 9,
        "detail":  checks, "grade": grade,
        "pattern": "Minervini SEPA",
        "rsi":     round(latest["rsi"], 1),
        "rel_vol": round(latest["rel_vol"], 1),
        "pct_from_high": round((price / high_52w - 1) * 100, 1),
        "entry_note": f"BUY breakout above pivot ${pivot} on 2x+ volume",
        "exit_note":  f"Stop below VCP base ${vcp_low} | Trail with MA50 | Exit on 3 down days",
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 2 — Bonde Momentum Burst
# ═══════════════════════════════════════════════════════════

def bonde_momentum(df: pd.DataFrame) -> Dict:
    if len(df) < 30:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Bonde Momentum",
                "direction": "BULL"}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    pct_chg = latest["pct_chg"]
    rel_vol = latest["rel_vol"]
    vol     = latest["volume"]
    close   = latest["close"]
    atr     = latest["atr"]

    bb_squeeze   = bool(latest["bb_width"] < df["bb_width"].iloc[-20:].mean() * 0.75)
    low_vol_base = bool(df["rel_vol"].iloc[-6:-1].mean() < 0.8)
    episodic     = bool(pct_chg >= 0.08 and rel_vol >= 3.0)
    burst_base   = bool(pct_chg >= 0.04 and vol > prev["volume"] and vol > 100_000)

    checks = {
        "Up >=4% Today":    bool(pct_chg >= 0.04),
        "Vol > Yesterday":  bool(vol > prev["volume"]),
        "Vol > 100K":       bool(vol > 100_000),
        "Above MA50":       bool(close > latest["ma50"]),
        "Above MA20":       bool(close > latest["ma20"]),
        "Low-Vol Base":     low_vol_base or bb_squeeze,
        "Rel Vol >=1.5x":   bool(rel_vol >= 1.5),
        "Episodic Pivot":   episodic,
    }

    score  = sum(checks.values())
    passed = burst_base
    setup  = "Episodic Pivot" if episodic else "Momentum Burst" if burst_base else "Not Triggered"
    grade  = "A" if episodic else "B" if score >= 6 else "C" if score >= 4 else "D"

    # Entry: current close; Stop: prior day low
    entry = round(close, 2)
    stop  = round(prev["low"], 2)
    trade = _calc_trade("BULL", entry, stop, atr, "Swing")

    return {
        "passed": passed, "score": score, "total": 8,
        "detail": checks, "grade": grade,
        "pattern": "Bonde Momentum", "setup": setup,
        "pct_chg": round(pct_chg * 100, 2),
        "rel_vol": round(rel_vol, 1),
        "rsi":     round(latest["rsi"], 1),
        "entry_note": f"BUY on Day 1 close ${entry} — do NOT chase Day 2+",
        "exit_note":  f"Exit Days 3–5, or RSI > 80, or -5% stop | Hard stop: ${stop}",
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 3 — Bullish Island Reversal
# ═══════════════════════════════════════════════════════════

def bullish_island_reversal(df: pd.DataFrame, lookback: int = 15) -> Dict:
    empty = {"passed": False, "score": 0, "detail": {}, "pattern": "Island Reversal",
             "grade": "N/A", "direction": "BULL", "gap_down_date": None, "island_days": 0}

    if len(df) < lookback + 5:
        return empty

    latest       = df.iloc[-1]
    gap_up_today = latest["gap_pct"] >= 0.01
    if not gap_up_today:
        return {**empty, "detail": {"Gap Up Today": False}}

    for j in range(2, lookback):
        idx_gap  = -j
        if abs(idx_gap) >= len(df):
            break
        gap_bar  = df.iloc[idx_gap]
        prev_bar = df.iloc[idx_gap - 1]

        if gap_bar["open"] >= prev_bar["close"] * 0.99:
            continue

        island       = df.iloc[idx_gap:-1]
        island_high  = island["high"].max()
        island_low   = island["low"].min()
        island_range = (island_high - island_low) / island_low if island_low > 0 else 1

        if island_range > 0.10:
            continue

        gap_fills    = latest["open"] >= prev_bar["close"] * 0.97
        high_vol     = latest["rel_vol"] >= 1.5
        closes_above = latest["close"] > island_high

        checks = {
            "Gap Down Found":      True,
            "Tight Island":        bool(island_range <= 0.08),
            "Gap Up Today":        True,
            "Gap Fills Prior":     bool(gap_fills),
            "High Volume":         bool(high_vol),
            "Closes Above Island": bool(closes_above),
        }

        score  = sum(checks.values())
        passed = score >= 4
        grade  = "A" if score >= 6 else "B" if score >= 4 else "C"

        entry = round(latest["close"], 2)
        stop  = round(island_low, 2)
        trade = _calc_trade("BULL", entry, stop, latest["atr"], "Swing")

        return {
            "passed":        passed,  "score": score, "total": 6,
            "detail":        checks,  "grade": grade,
            "pattern":       "Island Reversal",
            "gap_down_date": str(df.index[idx_gap].date()) if hasattr(df.index[idx_gap], "date") else str(df.index[idx_gap]),
            "island_days":   j - 1,
            "rel_vol":       round(latest["rel_vol"], 1),
            "gap_up_pct":    round(latest["gap_pct"] * 100, 2),
            "entry_note":    f"BUY on gap-up close ${entry} — island confirmed",
            "exit_note":     f"Stop below island low ${stop} | Target = island height × 2.5",
            **trade,
        }

    return empty


# ═══════════════════════════════════════════════════════════
#  STRATEGY 4 — Gap Up / Gap Down
# ═══════════════════════════════════════════════════════════

def gap_scanner(df: pd.DataFrame, min_gap_pct: float = 0.03,
                min_rel_vol: float = 1.5) -> Dict:
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Gap Scanner",
                "direction": "BULL"}

    latest   = df.iloc[-1]
    prev     = df.iloc[-2]
    gap_pct  = latest["gap_pct"]
    rel_vol  = latest["rel_vol"]
    close    = latest["close"]
    high_52w = latest["high_52w"]
    low_52w  = latest["low_52w"]
    atr      = latest["atr"]

    direction = "BULL" if gap_pct > 0 else "BEAR"
    abs_gap   = abs(gap_pct)

    if direction == "BULL":
        checks = {
            f"Gap >={int(min_gap_pct*100)}%": bool(abs_gap >= min_gap_pct),
            "Rel Vol >=1.5x":                 bool(rel_vol >= min_rel_vol),
            "Above MA50":                     bool(close > latest["ma50"]),
            "Near 52W High (<=15%)":          bool(close >= high_52w * 0.85),
            "RSI 40-80":                      bool(40 <= latest["rsi"] <= 80),
            "MACD Bullish":                   bool(latest["macd"] > latest["macd_sig"]),
        }
        entry      = round(close, 2)
        stop       = round(prev["close"], 2)          # stop = fill of gap
        entry_note = f"BUY above gap open ${entry} | Stop if gap fills ${stop}"
        exit_note  = "Exit T1 at +1.5R | Hold T2 if volume stays elevated | Trail MA20"
    else:
        checks = {
            f"Gap >={int(min_gap_pct*100)}%": bool(abs_gap >= min_gap_pct),
            "Rel Vol >=1.5x":                 bool(rel_vol >= min_rel_vol),
            "Below MA50":                     bool(close < latest["ma50"]),
            "Near 52W Low (<=15%)":           bool(close <= low_52w * 1.15),
            "RSI 20-60":                      bool(20 <= latest["rsi"] <= 60),
            "MACD Bearish":                   bool(latest["macd"] < latest["macd_sig"]),
        }
        entry      = round(close, 2)
        stop       = round(prev["close"], 2)          # stop = gap fills (bounce)
        entry_note = f"SHORT below gap open ${entry} | Cover if gap fills ${stop}"
        exit_note  = "Cover T1 at -1.5R | Trail with MA20 | Cover all on RSI < 30"

    score  = sum(checks.values())
    passed = bool(abs_gap >= min_gap_pct and rel_vol >= min_rel_vol)
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C" if score >= 3 else "D"
    trade  = _calc_trade(direction, entry, stop, atr, "Day/Swing")

    return {
        "passed":     passed,  "score": score, "total": 6,
        "detail":     checks,  "grade": grade,
        "pattern":    "Gap Scanner",
        "gap_pct":    round(gap_pct * 100, 2),
        "rel_vol":    round(rel_vol, 1),
        "rsi":        round(latest["rsi"], 1),
        "entry_note": entry_note,
        "exit_note":  exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 5 — Failed Breakdown (Bear Trap)
# ═══════════════════════════════════════════════════════════

def failed_breakdown(df: pd.DataFrame, support_lookback: int = 20) -> Dict:
    if len(df) < support_lookback + 2:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Failed Breakdown",
                "direction": "BULL"}

    latest    = df.iloc[-1]
    window    = df.iloc[-(support_lookback + 1):-1]
    support   = window["low"].min()
    atr       = latest["atr"]

    today_low   = latest["low"]
    today_close = latest["close"]
    today_high  = latest["high"]
    day_range   = today_high - today_low

    broke_below  = bool(today_low   < support * 0.993)
    recovered    = bool(today_close > support)
    upper_close  = bool(day_range > 0 and (today_close - today_low) / day_range > 0.60)
    high_vol     = bool(latest["rel_vol"] >= 1.2)
    rsi_now      = latest["rsi"]
    rsi_div      = bool(rsi_now > df["rsi"].iloc[-6])
    ma50_held    = bool(today_close > latest["ma50"])

    checks = {
        "Broke Below Support":  broke_below,
        "Closed Back Above":    recovered,
        "Bullish Candle Close": upper_close,
        "High Volume":          high_vol,
        "RSI Divergence":       rsi_div,
        "MA50 Held":            ma50_held,
    }

    score  = sum(checks.values())
    passed = bool(broke_below and recovered and upper_close)
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C" if score >= 3 else "D"

    # Next resistance = prior swing high (last 20 bars excluding today)
    next_res  = round(window["high"].max(), 2)
    entry     = round(today_close, 2)
    stop      = round(today_low * 0.998, 2)      # just below today's low
    trade     = _calc_trade("BULL", entry, stop, atr, "Swing")

    return {
        "passed":   passed,  "score": score, "total": 6,
        "detail":   checks,  "grade": grade,
        "pattern":  "Failed Breakdown",
        "support":  round(support, 2),
        "low":      round(today_low, 2),
        "close":    round(today_close, 2),
        "rel_vol":  round(latest["rel_vol"], 1),
        "rsi":      round(rsi_now, 1),
        "entry_note": f"BUY above support reclaim ${entry} | Next resistance ${next_res}",
        "exit_note":  f"Hard stop: ${stop} (below today's wick) | T1: ${next_res} | Trail MA20",
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 6 — Momentum Filter (Bull / Bear)
# ═══════════════════════════════════════════════════════════

def momentum_filter(df: pd.DataFrame) -> Dict:
    if len(df) < 200:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Momentum",
                "direction": "BULL"}

    latest   = df.iloc[-1]
    close    = latest["close"]
    rsi      = latest["rsi"]
    macd     = latest["macd"]
    macd_sig = latest["macd_sig"]
    rel_vol  = latest["rel_vol"]
    high_52w = latest["high_52w"]
    low_52w  = latest["low_52w"]
    atr      = latest["atr"]

    bull_checks = {
        "Price > MA20":          bool(close > latest["ma20"]),
        "Price > MA50":          bool(close > latest["ma50"]),
        "Price > MA200":         bool(close > latest["ma200"]),
        "MA50 > MA200":          bool(latest["ma50"] > latest["ma200"]),
        "RSI 50-80":             bool(50 <= rsi <= 80),
        "MACD > Signal":         bool(macd > macd_sig),
        "MACD > 0":              bool(macd > 0),
        "Near 52W High (<=10%)": bool(close >= high_52w * 0.90),
        "Rel Vol >=1.2x":        bool(rel_vol >= 1.2),
    }
    bull_score = sum(bull_checks.values())
    is_bullish = bull_score >= 7

    bear_checks = {
        "Price < MA20":          bool(close < latest["ma20"]),
        "Price < MA50":          bool(close < latest["ma50"]),
        "Price < MA200":         bool(close < latest["ma200"]),
        "MA50 < MA200":          bool(latest["ma50"] < latest["ma200"]),
        "RSI 20-50":             bool(20 <= rsi <= 50),
        "MACD < Signal":         bool(macd < macd_sig),
        "MACD < 0":              bool(macd < 0),
        "Near 52W Low (<=10%)":  bool(close <= low_52w * 1.10),
        "Rel Vol >=1.2x":        bool(rel_vol >= 1.2),
    }
    bear_score = sum(bear_checks.values())
    is_bearish = bear_score >= 7

    is_bullish_final = is_bullish
    direction = "BULL" if is_bullish_final else "BEAR" if is_bearish else "NEUTRAL"
    passed    = is_bullish or is_bearish
    score     = bull_score if is_bullish_final else bear_score
    grade     = "A" if score >= 9 else "B" if score >= 7 else "C" if score >= 5 else "D"

    if direction == "BULL":
        entry      = round(latest["ma20"], 2)         # enter on pullback to MA20
        stop       = round(latest["ma50"] * 0.99, 2)  # stop below MA50
        entry_note = f"BUY pullback to MA20 ${entry} | Stop below MA50 ${stop}"
        exit_note  = "Exit if RSI > 80 or price closes below MA50 | Trail MA20 for runners"
    else:
        entry      = round(latest["ma20"], 2)
        stop       = round(latest["ma50"] * 1.01, 2)
        entry_note = f"SHORT bounce to MA20 ${entry} | Cover above MA50 ${stop}"
        exit_note  = "Cover if RSI < 25 or price closes above MA50 | Trail MA20"

    trade = _calc_trade(direction, entry, stop, atr, "Swing")

    return {
        "passed":            passed,  "score": score, "total": 9,
        "detail":            bull_checks if is_bullish_final else bear_checks,
        "grade":             grade,
        "pattern":           "Momentum",
        "direction_label":   direction,
        "rsi":               round(rsi, 1),
        "rel_vol":           round(rel_vol, 1),
        "pct_from_52w_high": round((close / high_52w - 1) * 100, 1),
        "entry_note":        entry_note,
        "exit_note":         exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADE 1 — VWAP Trend
# ═══════════════════════════════════════════════════════════

def vwap_strategy(df: pd.DataFrame) -> Dict:
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "VWAP",
                "direction": "BULL"}

    latest    = df.iloc[-1]
    close     = latest["close"]
    vwap      = latest["vwap"]
    rel_vol   = latest["rel_vol"]
    rsi       = latest["rsi"]
    rsi_prev  = df["rsi"].iloc[-4]
    atr       = latest["atr"]

    above_vwap    = bool(close > vwap)
    pct_from_vwap = (close - vwap) / vwap
    direction     = "BULL" if above_vwap else "BEAR"

    if above_vwap:
        checks = {
            "Price Above VWAP":   True,
            "VWAP Gap > 0.5%":    bool(pct_from_vwap > 0.005),
            "Rel Vol >= 1.5x":    bool(rel_vol >= 1.5),
            "RSI Rising":         bool(rsi > rsi_prev),
            "RSI 45-80":          bool(45 <= rsi <= 80),
            "MA20 Aligned":       bool(close > latest["ma20"]),
        }
        entry      = round(vwap * 1.002, 2)    # entry just above VWAP
        stop       = round(vwap * 0.997, 2)    # stop just below VWAP
        entry_note = f"BUY on pullback to VWAP ${vwap:.2f} and bounce | Entry: ${entry}"
        exit_note  = f"Stop if close below VWAP ${stop} | Exit at T1 or EOD"
    else:
        checks = {
            "Price Below VWAP":   True,
            "VWAP Gap > 0.5%":    bool(abs(pct_from_vwap) > 0.005),
            "Rel Vol >= 1.5x":    bool(rel_vol >= 1.5),
            "RSI Falling":        bool(rsi < rsi_prev),
            "RSI 20-55":          bool(20 <= rsi <= 55),
            "MA20 Aligned":       bool(close < latest["ma20"]),
        }
        entry      = round(vwap * 0.998, 2)
        stop       = round(vwap * 1.003, 2)
        entry_note = f"SHORT on bounce to VWAP ${vwap:.2f} and rejection | Entry: ${entry}"
        exit_note  = f"Cover if close above VWAP ${stop} | Exit at T1 or EOD"

    score  = sum(checks.values())
    passed = score >= 4
    grade  = "A" if score >= 6 else "B" if score >= 4 else "C"
    trade  = _calc_trade(direction, entry, stop, atr, "Day Trade")

    return {
        "passed":        passed,  "score": score, "total": 6,
        "detail":        checks,  "grade": grade,
        "pattern":       "VWAP",
        "vwap":          round(vwap, 2),
        "pct_from_vwap": round(pct_from_vwap * 100, 2),
        "rsi":           round(rsi, 1),
        "rel_vol":       round(rel_vol, 1),
        "entry_note":    entry_note,
        "exit_note":     exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADE 2 — Opening Range Breakout (ORB)
# ═══════════════════════════════════════════════════════════

def opening_range_breakout(df: pd.DataFrame) -> Dict:
    empty = {"passed": False, "score": 0, "detail": {"No significant gap": False},
             "pattern": "ORB", "grade": "D", "direction": "BULL"}
    if len(df) < 5:
        return empty

    latest    = df.iloc[-1]
    prev      = df.iloc[-2]
    day_range = latest["high"] - latest["low"]
    close_pos = (latest["close"] - latest["low"]) / day_range if day_range > 0 else 0.5
    gap_pct   = latest["gap_pct"]
    rel_vol   = latest["rel_vol"]
    atr       = latest["atr"]

    bullish_orb = gap_pct > 0.005
    bearish_orb = gap_pct < -0.005

    if bullish_orb:
        direction = "BULL"
        checks = {
            "Gap Up Open":           bool(gap_pct > 0.005),
            "Closed Top 25% Range":  bool(close_pos >= 0.75),
            "Rel Vol >= 2x":         bool(rel_vol >= 2.0),
            "Above Yesterday High":  bool(latest["close"] > prev["high"]),
            "RSI > 50":              bool(latest["rsi"] > 50),
            "MACD Bullish":          bool(latest["macd"] > latest["macd_sig"]),
        }
        orb_high   = round(latest["high"], 2)
        orb_low    = round(latest["low"],  2)
        entry      = round(orb_high + 0.01, 2)
        stop       = round(orb_low  - 0.01, 2)
        entry_note = f"BUY break above ORB high ${orb_high} | Intraday trigger"
        exit_note  = f"Stop below ORB low ${orb_low} | Target = ORB height × 2 above entry | Exit EOD"

    elif bearish_orb:
        direction = "BEAR"
        checks = {
            "Gap Down Open":         bool(gap_pct < -0.005),
            "Closed Bot 25% Range":  bool(close_pos <= 0.25),
            "Rel Vol >= 2x":         bool(rel_vol >= 2.0),
            "Below Yesterday Low":   bool(latest["close"] < prev["low"]),
            "RSI < 50":              bool(latest["rsi"] < 50),
            "MACD Bearish":          bool(latest["macd"] < latest["macd_sig"]),
        }
        orb_high   = round(latest["high"], 2)
        orb_low    = round(latest["low"],  2)
        entry      = round(orb_low  - 0.01, 2)
        stop       = round(orb_high + 0.01, 2)
        entry_note = f"SHORT break below ORB low ${orb_low}"
        exit_note  = f"Cover above ORB high ${orb_high} | Target = ORB height × 2 below entry | Cover EOD"
    else:
        return empty

    score  = sum(checks.values())
    passed = score >= 4
    grade  = "A" if score >= 6 else "B" if score >= 4 else "C"
    trade  = _calc_trade(direction, entry, stop, atr, "Day Trade")

    return {
        "passed":     passed,  "score": score, "total": 6,
        "detail":     checks,  "grade": grade,
        "pattern":    "ORB",
        "gap_pct":    round(gap_pct * 100, 2),
        "close_pos":  round(close_pos * 100, 1),
        "rel_vol":    round(rel_vol, 1),
        "rsi":        round(latest["rsi"], 1),
        "entry_note": entry_note,
        "exit_note":  exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADE 3 — High Relative Volume
# ═══════════════════════════════════════════════════════════

def high_relative_volume(df: pd.DataFrame, threshold: float = 3.0) -> Dict:
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "High Rel Vol",
                "direction": "BULL"}

    latest     = df.iloc[-1]
    rel_vol    = latest["rel_vol"]
    pct_chg    = latest["pct_chg"]
    close      = latest["close"]
    atr        = latest["atr"]
    today_range = latest["high"] - latest["low"]
    avg_range   = (df["high"] - df["low"]).rolling(20).mean().iloc[-1]
    range_exp   = today_range / avg_range if avg_range > 0 else 1
    direction   = "BULL" if pct_chg >= 0 else "BEAR"

    checks = {
        f"Rel Vol >={threshold}x":   bool(rel_vol >= threshold),
        "Range Expansion >= 1.5x":   bool(range_exp >= 1.5),
        "Price Moving (>1%)":        bool(abs(pct_chg) >= 0.01),
        "MA Aligned":                bool(close > latest["ma20"]) if direction == "BULL"
                                     else bool(close < latest["ma20"]),
        "RSI Active (40-85)":        bool(40 <= latest["rsi"] <= 85),
        "MACD Confirming":           bool(
            (pct_chg > 0 and latest["macd"] > latest["macd_sig"]) or
            (pct_chg < 0 and latest["macd"] < latest["macd_sig"])
        ),
    }

    score  = sum(checks.values())
    passed = bool(rel_vol >= threshold and abs(pct_chg) >= 0.01)
    grade  = "A" if score >= 6 else "B" if score >= 4 else "C"

    if direction == "BULL":
        entry      = round(latest["vwap"], 2)
        stop       = round(latest["low"] * 0.997, 2)
        entry_note = f"BUY first pullback to VWAP ${entry} | Avoid chasing"
        exit_note  = f"Stop: ${stop} | Exit if vol drops to <1.5x avg | T1 at +1.5R"
    else:
        entry      = round(latest["vwap"], 2)
        stop       = round(latest["high"] * 1.003, 2)
        entry_note = f"SHORT first bounce to VWAP ${entry}"
        exit_note  = f"Cover: ${stop} | Cover if vol drops | T1 at -1.5R"

    trade = _calc_trade(direction, entry, stop, atr, "Day Trade")

    return {
        "passed":    passed,  "score": score, "total": 6,
        "detail":    checks,  "grade": grade,
        "pattern":   "High Rel Vol",
        "rel_vol":   round(rel_vol, 1),
        "range_exp": round(range_exp, 1),
        "pct_chg":   round(pct_chg * 100, 2),
        "rsi":       round(latest["rsi"], 1),
        "entry_note":entry_note,
        "exit_note": exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADE 4 — Pre-Market Gap & Go
# ═══════════════════════════════════════════════════════════

def premarket_gap_go(df: pd.DataFrame) -> Dict:
    empty = {"passed": False, "score": 0,
             "detail": {"Gap >= 3% Required": False},
             "pattern": "Gap & Go", "grade": "D", "direction": "BULL"}

    if len(df) < 10:
        return empty

    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    gap_pct = latest["gap_pct"]
    abs_gap = abs(gap_pct)
    atr     = latest["atr"]

    if abs_gap < 0.03:
        return empty

    gap_up    = gap_pct > 0
    direction = "BULL" if gap_up else "BEAR"
    close_pos = (latest["close"] - latest["low"]) / max(latest["high"] - latest["low"], 0.01)

    if gap_up:
        checks = {
            "Gap Up >= 3%":           bool(gap_pct >= 0.03),
            "Gap Up >= 5% (Strong)":  bool(gap_pct >= 0.05),
            "Continues (no fade)":    bool(close_pos >= 0.5),
            "Rel Vol >= 2x":          bool(latest["rel_vol"] >= 2.0),
            "Above Prev Day High":    bool(latest["close"] > prev["high"]),
            "RSI 50-85":              bool(50 <= latest["rsi"] <= 85),
        }
        entry      = round(latest["open"], 2)
        stop       = round(prev["close"] * 0.995, 2)   # stop = gap fills
        entry_note = f"BUY at open ${entry} or first 5-min breakout | Gap fills = stop"
        exit_note  = f"Hard stop: ${stop} (gap fill) | Trail HOD | Exit full by 3:30 PM"
    else:
        checks = {
            "Gap Down >= 3%":          bool(gap_pct <= -0.03),
            "Gap Down >= 5% (Strong)": bool(gap_pct <= -0.05),
            "Continues Down":          bool(close_pos <= 0.5),
            "Rel Vol >= 2x":           bool(latest["rel_vol"] >= 2.0),
            "Below Prev Day Low":      bool(latest["close"] < prev["low"]),
            "RSI 15-50":               bool(15 <= latest["rsi"] <= 50),
        }
        entry      = round(latest["open"], 2)
        stop       = round(prev["close"] * 1.005, 2)
        entry_note = f"SHORT at open ${entry} or 5-min breakdown | Gap fills = cover"
        exit_note  = f"Cover: ${stop} | Trail LOD | Cover all by 3:30 PM"

    score  = sum(checks.values())
    passed = score >= 4
    grade  = "A" if score >= 6 else "B" if score >= 5 else "C" if score >= 4 else "D"
    trade  = _calc_trade(direction, entry, stop, atr, "Day Trade")

    return {
        "passed":     passed,  "score": score, "total": 6,
        "detail":     checks,  "grade": grade,
        "pattern":    "Gap & Go",
        "gap_pct":    round(gap_pct * 100, 2),
        "rel_vol":    round(latest["rel_vol"], 1),
        "rsi":        round(latest["rsi"], 1),
        "entry_note": entry_note,
        "exit_note":  exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADE 5 — Intraday Momentum
# ═══════════════════════════════════════════════════════════

def intraday_momentum(df: pd.DataFrame) -> Dict:
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Intraday Momentum",
                "direction": "BULL"}

    latest     = df.iloc[-1]
    prev       = df.iloc[-2]
    close      = latest["close"]
    pct_chg    = latest["pct_chg"]
    atr        = latest["atr"]
    direction  = "BULL" if pct_chg >= 0 else "BEAR"

    candle_range = latest["high"] - latest["low"]
    candle_body  = abs(latest["close"] - latest["open"])
    body_ratio   = candle_body / candle_range if candle_range > 0 else 0

    checks = {
        "Move > 2% Today":            bool(abs(pct_chg) >= 0.02),
        "Rel Vol >= 2x":              bool(latest["rel_vol"] >= 2.0),
        "Strong Candle (>50% body)":  bool(body_ratio > 0.5),
        "Trending (MA20 aligned)":    bool(close > latest["ma20"]) if pct_chg > 0
                                      else bool(close < latest["ma20"]),
        "MACD Aligned":               bool(latest["macd"] > latest["macd_sig"]) if pct_chg > 0
                                      else bool(latest["macd"] < latest["macd_sig"]),
        "BB Expansion":               bool(latest["bb_width"] > df["bb_width"].iloc[-10:].mean() * 1.2),
        "Above/Below Prev Close":     bool(close > prev["close"]) if pct_chg > 0
                                      else bool(close < prev["close"]),
    }

    score  = sum(checks.values())
    passed = score >= 5
    grade  = "A" if score >= 7 else "B" if score >= 5 else "C"

    if direction == "BULL":
        entry      = round(close, 2)
        stop       = round(close - atr * 1.5, 2)
        entry_note = f"BUY close ${entry} | Stop = 1.5× ATR below: ${stop}"
        exit_note  = "Exit T1 at +1.5R | Hold T2 if momentum continues | Hard stop = 1.5×ATR"
    else:
        entry      = round(close, 2)
        stop       = round(close + atr * 1.5, 2)
        entry_note = f"SHORT close ${entry} | Stop = 1.5× ATR above: ${stop}"
        exit_note  = "Cover T1 at -1.5R | Trail if momentum continues | Cover EOD"

    trade = _calc_trade(direction, entry, stop, atr, "Day Trade")

    return {
        "passed":     passed,  "score": score, "total": 7,
        "detail":     checks,  "grade": grade,
        "pattern":    "Intraday Momentum",
        "pct_chg":    round(pct_chg * 100, 2),
        "body_ratio": round(body_ratio * 100, 1),
        "rel_vol":    round(latest["rel_vol"], 1),
        "rsi":        round(latest["rsi"], 1),
        "entry_note": entry_note,
        "exit_note":  exit_note,
        **trade,
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY MAPS
# ═══════════════════════════════════════════════════════════

SWING_STRATEGIES = {
    "Minervini SEPA":   minervini_sepa,
    "Bonde Momentum":   bonde_momentum,
    "Island Reversal":  bullish_island_reversal,
    "Gap Up/Down":      gap_scanner,
    "Failed Breakdown": failed_breakdown,
    "Momentum Filter":  momentum_filter,
}

DAY_STRATEGIES = {
    "VWAP Trend":          vwap_strategy,
    "Opening Range (ORB)": opening_range_breakout,
    "High Rel Volume":     high_relative_volume,
    "Gap & Go":            premarket_gap_go,
    "Intraday Momentum":   intraday_momentum,
}

STRATEGY_MAP = {**SWING_STRATEGIES, **DAY_STRATEGIES}


# ═══════════════════════════════════════════════════════════
#  MASTER RUNNER
# ═══════════════════════════════════════════════════════════

def run_strategies(ticker: str, df_raw, selected: list) -> dict:
    if df_raw is None or len(df_raw) < 30:
        return None

    df  = compute_indicators(df_raw)
    lat = df.iloc[-1]

    row = {
        "Ticker":  ticker,
        "Price":   round(lat["close"], 2),
        "Chg%":    round(lat["pct_chg"] * 100, 2),
        "Vol":     int(lat["volume"]),
        "RelVol":  round(lat["rel_vol"], 1),
        "RSI":     round(lat["rsi"], 1),
        "Signals": [],
        "_detail": {},
        "_trades": [],     # list of trade setups for Entry/Exit tab
    }

    for name in selected:
        fn = STRATEGY_MAP.get(name)
        if fn is None:
            continue
        try:
            res = fn(df)
            row["_detail"][name] = res
            if res.get("passed"):
                grade = res.get("grade", "")
                row["Signals"].append(f"{name} [{grade}]")
                row["_trades"].append({
                    "strategy":   name,
                    "direction":  res.get("direction", "BULL"),
                    "grade":      grade,
                    "entry":      res.get("entry"),
                    "stop_loss":  res.get("stop_loss"),
                    "target_1":   res.get("target_1"),
                    "target_2":   res.get("target_2"),
                    "target_3":   res.get("target_3"),
                    "risk_%":     res.get("risk_%"),
                    "R:R":        res.get("R:R", "1:2.5"),
                    "trade_type": res.get("trade_type", "Swing"),
                    "entry_note": res.get("entry_note", ""),
                    "exit_note":  res.get("exit_note", ""),
                })
        except Exception as e:
            row["_detail"][name] = {"passed": False, "error": str(e)}

    row["# Signals"]   = len(row["Signals"])
    row["Signal List"] = ", ".join(row["Signals"]) if row["Signals"] else "—"

    # Dominant direction from first passing strategy
    if row["_trades"]:
        bulls = sum(1 for t in row["_trades"] if t["direction"] == "BULL")
        bears = sum(1 for t in row["_trades"] if t["direction"] == "BEAR")
        row["Direction"] = "BULL" if bulls >= bears else "BEAR"
    else:
        row["Direction"] = "—"

    return row
