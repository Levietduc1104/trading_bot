# Trading Bot for Multi-Timeframe Strategy

This repository contains a backtesting system that implements a multi-timeframe trading strategy using MetaTrader 5 (MT5) data. The strategy is based on multiple indicators such as EMAs, ATR, and ADX, and provides functionality to plot entry/exit points and export trading statistics and results to Excel.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Backtest Results](#backtest-results)
- [Contributing](#contributing)

## Overview

The system is designed to:
- Fetch historical data from MT5
- Execute a custom multi-timeframe trading strategy
- Calculate various statistics (e.g., profit factor, win rate, drawdown)
- Export results and statistics to Excel
- Plot entry/exit points on the price chart

## Installation

### Requirements

- Python 3.x
- MetaTrader 5 (MT5) installed and running
- Virtual environment set up (optional but recommended)

### Dependencies

Install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt


Running the Trading Bot
python backtest.py --run_cross_sma



### Instructions:
- This file provides an overview, installation, usage instructions, and an outline of the system's functionality.
- You can modify the `LOGIN`, `PASSWORD`, `SERVER`, `SYMBOL`, `START_DATE`, `END_DATE`, and other parameters as needed.
- The `usage` section explains how to run the bot and the files it will generate.