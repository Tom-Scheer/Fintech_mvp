import yfinance as yf
import pandas as pd

# Define the list of tickers
tickers = [
    'BTC-USD', 'ETH-USD', 'GC=F', 'SI=F', 'SPY', 'QQQ', 'SOXX', 'VT', 'VO', 'VWO',
    'NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN', 'META', 'TSM', 'TSLA', 'V', 'WMT',
    'XOM', 'ASML', 'PG', 'BAC', 'NFLX', 'KO', 'QCOM', 'AMD', 'PEP', 'SHEL',
    'DIS', 'BABA', 'MCD', 'VZ', 'NKE', 'JPM', 'GE', 'BA', 'SBUX', 'UL', 'PM', 'MO',
    'SHOP', 'UBER', 'SPOT'
]

# Fetch historical data
data = yf.download(tickers, start="2010-01-01", end="2024-06-19")['Adj Close']

# Calculate daily returns
returns = data.pct_change()

# Calculate the volatility (standard deviation of returns)
volatility = returns.std()

# Sort assets by volatility in descending order
sorted_volatility = volatility.sort_values(ascending=False)

# Define volatility thresholds for bucketing into risk levels
low_vol_threshold = sorted_volatility.quantile(0.33)
high_vol_threshold = sorted_volatility.quantile(0.66)

# Categorize assets into risk levels
low_risk_assets = {ticker: vol for ticker, vol in sorted_volatility.items() if vol <= low_vol_threshold}
medium_risk_assets = {ticker: vol for ticker, vol in sorted_volatility.items() if low_vol_threshold < vol <= high_vol_threshold}
high_risk_assets = {ticker: vol for ticker, vol in sorted_volatility.items() if vol > high_vol_threshold}

# Print assets and their volatility sorted by risk level and volatility
print("High Risk Assets:")
for ticker, vol in high_risk_assets.items():
    print(f"{ticker}: {vol:.4f}")

print("\nMedium Risk Assets:")
for ticker, vol in medium_risk_assets.items():
    print(f"{ticker}: {vol:.4f}")

print("\nLow Risk Assets:")
for ticker, vol in low_risk_assets.items():
    print(f"{ticker}: {vol:.4f}")
