
# Trade Analyzer (IBKR + Fidelity) — Options FIFO P&L → Excel

This build handles Fidelity "Accounts History" exports that include disclaimer footer text.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```


## Insights
This version includes out-of-the-box analytics: expectancy, profit factor, payoff ratio, drawdown, holding-time buckets, and leaderboards.


## Actionable recommendations
This version adds file-based diagnostics (intraday fee drag, segment leaks, oversizing) and prints concrete do-more/do-less recommendations.
