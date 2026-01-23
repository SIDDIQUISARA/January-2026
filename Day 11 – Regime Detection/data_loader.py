import yfinance as yf
import numpy as np

def load_returns(ticker="SPY", start="2018-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end)
    data["Returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
    data.dropna(inplace=True)
    return data
