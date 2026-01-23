# Day 11 – Regime Detection using Hidden Markov Models

This project uses Hidden Markov Models (HMM) to identify hidden market regimes
such as bull and bear markets.

## Methodology
- Compute log returns
- Train Gaussian HMM
- Infer hidden market states
- Analyze regime statistics

## Applications
- Volatility regime detection
- Strategy switching
- Risk management

## Run
```bash
pip install -r requirements.txt
python backtest.py
