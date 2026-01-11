# Tool Scripts

This directory contains utility and tool scripts for data processing, testing, and analysis.

## Data Processing Tools

- **create_period_datasets.py** - Split stock data into time period datasets (1963-1983, 1983-2003, 1990-2024)
- **filter_top500_per_period.py** - Filter and select top 500 companies per period using S&P 500-like criteria
- **split_stock_data.py** - Split stock data by time periods
- **create_sp500_index.py** - Create S&P 500 index data
- **create_spy_metadata.py** - Create SPY metadata
- **create_vix_proxy.py** - Create VIX proxy from stock data
- **generate_sp500_data.py** - Generate S&P 500 stock data

## Optimization & Testing Tools

- **optimize_strategy.py** - Optimize trading strategy parameters
- **optimize_v20_leverage.py** - Optimize V20 strategy leverage settings
- **run_best_bear_protection.py** - Run backtest with best bear market protection settings

## Production Run Scripts

- **run_v22_production.py** - Execute V22 production strategy
- **run_v13_production_5stocks.py** - Execute V13 5-stock production strategy

## Usage

All scripts in this directory are standalone tools. Run them from the project root:

```bash
python3 src/tool/<script_name>.py
```

Example:
```bash
python3 src/tool/filter_top500_per_period.py
```
