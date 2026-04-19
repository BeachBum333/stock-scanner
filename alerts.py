"""
alerts.py — Alert System for Stock Scanner

Supports:
  1. In-app toast notifications (Streamlit)
  2. Email alerts via Gmail SMTP
  3. Browser sound alert (JavaScript)
  4. Alert log (session state history)
"""

import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional


# ── Alert Log (stored in session state) ──────────────────────────────────────

def _init_log():
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []


def log_alert(ticker: str, signal: str, details: str = ""):
    _init_log()
    st.session_state.alert_log.append({
        "time":    datetime.now().strftime("%H:%M:%S"),
        "ticker":  ticker,
        "signal":  signal,
        "details": details,
    })


def get_alert_log() -> List[Dict]:
    _init_log()
    return st.session_state.alert_log


def clear_alert_log():
    st.session_state.alert_log = []


# ── In-app Toast ─────────────────────────────────────────────────────────────

def toast_alert(ticker: str, signal: str):
    """Show a Streamlit toast notification."""
    st.toast(f"🚨 **{ticker}** — {signal}", icon="📈")


# ── Browser Sound Alert ───────────────────────────────────────────────────────

ALERT_SOUND_JS = """
<script>
(function() {
  try {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    function beep(freq, duration, vol) {
      var osc  = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = freq;
      osc.type = 'sine';
      gain.gain.setValueAtTime(vol, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + duration);
    }
    beep(880, 0.15, 0.4);
    setTimeout(function(){ beep(1100, 0.15, 0.4); }, 150);
    setTimeout(function(){ beep(1320, 0.25, 0.4); }, 300);
  } catch(e) {}
})();
</script>
"""

def play_sound():
    """Inject JavaScript to play a triple-beep alert in the browser."""
    st.components.v1.html(ALERT_SOUND_JS, height=0)


# ── Email Alert ───────────────────────────────────────────────────────────────

def send_email_alert(
    ticker: str,
    signals: List[str],
    details: str,
    smtp_user: str,
    smtp_pass: str,
    to_email: str,
) -> bool:
    """
    Send an email alert via Gmail SMTP.

    smtp_user: your Gmail address (e.g. you@gmail.com)
    smtp_pass: Gmail App Password (NOT your regular password)
               Generate at: myaccount.google.com → Security → App Passwords
    to_email:  recipient (can be same as smtp_user)

    Returns True if sent successfully.
    """
    if not (smtp_user and smtp_pass and to_email):
        return False

    subject = f"📈 Scanner Alert: {ticker} — {', '.join(signals)}"

    body_html = f"""
    <html><body style="font-family:Arial,sans-serif; background:#0f172a; color:#e2e8f0; padding:24px;">
      <h2 style="color:#6366f1;">📈 Stock Scanner Alert</h2>
      <table style="border-collapse:collapse; width:100%; max-width:500px;">
        <tr><td style="padding:8px; color:#94a3b8;">Ticker</td>
            <td style="padding:8px; font-weight:bold; font-size:1.3rem;">{ticker}</td></tr>
        <tr><td style="padding:8px; color:#94a3b8;">Signals</td>
            <td style="padding:8px; color:#10b981;">{'<br>'.join(signals)}</td></tr>
        <tr><td style="padding:8px; color:#94a3b8;">Details</td>
            <td style="padding:8px;">{details}</td></tr>
        <tr><td style="padding:8px; color:#94a3b8;">Time</td>
            <td style="padding:8px;">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
      </table>
      <p style="color:#475569; font-size:0.8rem; margin-top:24px;">
        Not financial advice. Always use stop-losses.
      </p>
    </body></html>
    """

    try:
        msg                    = MIMEMultipart("alternative")
        msg["Subject"]         = subject
        msg["From"]            = smtp_user
        msg["To"]              = to_email
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        st.warning(f"Email alert failed: {e}")
        return False


# ── Batch Alert Dispatcher ────────────────────────────────────────────────────

def dispatch_alerts(
    results: List[Dict],
    enable_sound:  bool = True,
    enable_toast:  bool = True,
    enable_email:  bool = False,
    smtp_user:     str  = "",
    smtp_pass:     str  = "",
    to_email:      str  = "",
    min_grade:     str  = "B",
):
    """
    Process scan results and fire alerts for qualifying signals.

    min_grade: only alert on signals graded A or B (set "C" to alert on all).
    """
    grade_rank = {"A": 3, "B": 2, "C": 1, "D": 0}
    threshold  = grade_rank.get(min_grade, 2)

    alerted = []

    for row in results:
        ticker  = row.get("Ticker", "")
        signals = row.get("Signals", [])
        if not signals:
            continue

        # Filter by grade
        qualifying = [
            s for s in signals
            if grade_rank.get(s.split("[")[-1].replace("]", ""), 0) >= threshold
        ]
        if not qualifying:
            continue

        detail_str = f"Price: ${row.get('Price','?')} | RSI: {row.get('RSI','?')} | RelVol: {row.get('RelVol','?')}x"

        # Log it
        log_alert(ticker, ", ".join(qualifying), detail_str)
        alerted.append(ticker)

        # Toast
        if enable_toast:
            toast_alert(ticker, ", ".join(qualifying))

        # Email
        if enable_email:
            send_email_alert(ticker, qualifying, detail_str, smtp_user, smtp_pass, to_email)

    # Sound (once if any alerts fired)
    if enable_sound and alerted:
        play_sound()

    return alerted
