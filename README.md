# Stock Strategy Scanner

## Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Create free GitHub account at github.com
2. Upload all these files to a new repo
3. Go to share.streamlit.io → New app → select repo → app.py
4. Add Alpaca API keys under Settings → Secrets:
   ALPACA_API_KEY = "your-key"
   ALPACA_SECRET_KEY = "your-secret"

## No API keys?
The app works without Alpaca keys — it falls back to Yahoo Finance (free, end-of-day data).
