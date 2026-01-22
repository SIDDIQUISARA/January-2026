# Day 10 – Cointegration & Pairs Trading Strategy

This project implements a market-neutral pairs trading strategy using cointegration.

## Strategy Overview
- Identify cointegrated asset pairs
- Estimate hedge ratio via OLS
- Trade spread mean reversion using Z-score
- Compute PnL and cumulative returns

## Trading Rules
- Long spread when Z-score < -1
- Short spread when Z-score > 1
- Exit when |Z-score| < 0.5

## Run
```bash
pip install -r requirements.txt
python backtest.py
