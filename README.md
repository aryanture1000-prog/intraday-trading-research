# Intraday Trading Research System

A Python framework for researching intraday strategies on NSE equities using live Angel One market data. **Orders are simulated. No real orders are placed.**

## What it does
- **Forecasting ensemble (`MLEnsemble`)**
  - An LSTM network for sequence patterns
  - An XGBoost classifier, with a random forest fallback
  - A per-symbol ARIMA(0,d,1) model, where d is chosen with an Augmented Dickey-Fuller stationarity test and the model is refit every 24 hours
- **Live data**: connects to Angel One SmartAPI over WebSocket, with session and TOTP refresh and reconnect handling.
- **Risk controls**: position-size limits, daily-loss and drawdown caps, stop-loss and take-profit, and a limit on capital at risk.
- **Order simulation**: market orders fill at the current price. Positions are tracked and trade history is saved to CSV.
- **Performance tracking** of the paper-trading results.

## Structure
| Component | Role |
|---|---|
| `Config` | trading universe, risk and timing parameters |
| `MLEnsemble` | features, training, prediction |
| `AngelOneConnector` | login, WebSocket feed, session refresh |
| `MarketDataManager` | tick handling and historical data |
| `OrderManager` / `PositionManager` | simulated execution and position state |
| `RiskManager` | limits and checks |
| `PerformanceTracker` | metrics and reporting |
| `TradingSystem` | orchestrates everything |

## Run
```
pip install -r requirements.txt
cp .env.example .env     # add your own Angel One credentials
python trading_system.py
```
Credentials are read from environment variables and never stored in code.

## Disclaimer
Research project, not financial advice. Not validated for live trading.
