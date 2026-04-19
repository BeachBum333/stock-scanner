"""
universes.py — Stock universe definitions
Fetches S&P 500 and NASDAQ 100 tickers from Wikipedia (always up to date).
Uses requests with a browser User-Agent to avoid 403 blocks.
"""

import pandas as pd
import streamlit as st
from typing import List
import requests
import io

# Mimic a real browser so Wikipedia doesn't block us
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _fetch_html(url: str) -> str:
    """Fetch page HTML with browser-like headers."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sp500() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        html    = _fetch_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tables  = pd.read_html(io.StringIO(html))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except Exception as e:
        st.warning(f"Could not fetch S&P 500 list: {e}. Using fallback.")
        return SP500_FALLBACK


@st.cache_data(ttl=86400)
def get_nasdaq100() -> List[str]:
    """Fetch NASDAQ 100 tickers from Wikipedia."""
    try:
        html   = _fetch_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        tables = pd.read_html(io.StringIO(html))
        for tbl in tables:
            if "Ticker" in tbl.columns or "Symbol" in tbl.columns:
                col     = "Ticker" if "Ticker" in tbl.columns else "Symbol"
                tickers = tbl[col].dropna().tolist()
                if len(tickers) > 50:
                    return sorted([str(t).replace(".", "-") for t in tickers])
        raise ValueError("Ticker table not found on page")
    except Exception as e:
        st.warning(f"Could not fetch NASDAQ 100 list: {e}. Using fallback.")
        return NASDAQ100_FALLBACK


def get_universe(selection: str, custom_tickers: str = "") -> List[str]:
    """
    Returns the appropriate list of tickers based on user selection.
    selection: 'S&P 500' | 'NASDAQ 100' | 'Custom'
    """
    if selection == "S&P 500":
        return get_sp500()
    elif selection == "NASDAQ 100":
        return get_nasdaq100()
    elif selection == "Custom":
        if not custom_tickers.strip():
            return []
        tickers = [t.strip().upper() for t in custom_tickers.replace(",", " ").split()]
        return sorted(list(set(tickers)))
    return []


# ── Fallback lists (used if Wikipedia is unreachable) ──────────────────────

SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "XOM", "V", "AVGO", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "COST", "PEP", "KO", "ADBE", "WMT", "MCD", "CRM", "BAC", "ACN",
    "NFLX", "AMD", "LIN", "TMO", "CSCO", "ORCL", "ABT", "TXN", "NEE", "DHR",
    "NKE", "PM", "QCOM", "RTX", "HON", "AMGN", "INTU", "SPGI", "IBM", "GE",
    "AMAT", "CAT", "INTC", "BA", "GS", "MS", "BLK", "SYK", "ISRG", "T",
    "GILD", "C", "PLD", "CI", "AXP", "MDLZ", "ADI", "REGN", "VRTX", "SO",
    "DUK", "MO", "TGT", "ZTS", "BKNG", "ELV", "MMC", "CB", "NOW", "LRCX",
    "SCHW", "MU", "APD", "DE", "TJX", "ETN", "KLAC", "BSX", "HCA", "PGR",
    "CME", "AON", "SHW", "SNPS", "CDNS", "FI", "GD", "ECL", "ADP", "MCO",
    "CRWD", "PLTR", "MELI", "SNOW", "DDOG", "PANW", "ZS", "FTNT", "NET", "OKTA"
]

NASDAQ100_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTU", "CSCO", "AMGN", "AMAT",
    "HON", "TXN", "INTC", "BKNG", "LRCX", "PANW", "SNPS", "CDNS", "SBUX",
    "ADI", "GILD", "MDLZ", "KLAC", "REGN", "MU", "MRVL", "CRWD", "PYPL",
    "ORLY", "ABNB", "CEG", "CTAS", "MCHP", "MAR", "ADSK", "FTNT", "KDP",
    "MNST", "PAYX", "ROST", "AEP", "IDXX", "BIIB", "ODFL", "PCAR", "EXC",
    "DXCM", "XEL", "FAST", "FANG", "WBD", "ON", "VRSK", "KHC", "DDOG",
    "ZS", "TEAM", "ANSS", "GEHC", "CTSH", "CPRT", "EA", "DLTR", "GFS",
    "ILMN", "WBA", "SIRI", "LCID", "RIVN", "ZM", "DOCU", "OKTA", "SNOW",
    "PLTR", "NET", "COIN", "HOOD", "SOFI", "AFRM", "UPST", "RBLX", "DKNG"
]
