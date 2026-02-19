# FMP Data Fetcher

Download historical fundamental data from Financial Modeling Prep (FMP) API.

## 📁 Folder Contents

```
fmp_data_fetcher/
├── README.md                      # This file
├── FMP_DOWNLOAD_GUIDE.md         # Detailed usage guide
├── test_fmp_api.py               # Test API and check data availability
├── fetch_fmp_market_cap.py       # Download historical market cap (2000-2024)
├── fetch_fmp_fundamentals.py     # Download fundamental ratios (P/E, EPS, ROE, etc.)
└── test_symbols.txt              # 10 test symbols for testing
```

## 🔑 API Key (Already Configured)

Your FMP API key is already set in all scripts: `yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w`

## 🚀 Quick Start

⚠️ **IMPORTANT**: Run from HOME network (current network has proxy blocking API calls)

### Test Single Stock

```bash
cd /Users/levietduc/Documents/Documents\ -\ Le\'s\ MacBook\ Pro/Learning/ml_tranning/trading_bot/src/tool/fmp_data_fetcher

# Test API
python3 test_fmp_api.py

# Download AAPL market cap
python3 fetch_fmp_market_cap.py --symbol AAPL

# Download AAPL fundamentals
python3 fetch_fmp_fundamentals.py --symbol AAPL --period quarter
```

### Test Multiple Stocks

```bash
# Download market cap for 10 test stocks
python3 fetch_fmp_market_cap.py --file test_symbols.txt --output test_market_cap.csv

# Download fundamentals for 10 test stocks
python3 fetch_fmp_fundamentals.py --file test_symbols.txt --type comprehensive --output test_fundamentals.csv
```

### Download Full S&P 500

```bash
# First, create sp500_symbols.txt with all S&P 500 symbols (one per line)
# Then:

# Market cap (takes ~2 days with free tier, 250 calls/day limit)
python3 fetch_fmp_market_cap.py --file sp500_symbols.txt --output sp500_market_cap_2000_2024.csv

# Fundamentals (takes ~2 days with free tier)
python3 fetch_fmp_fundamentals.py --file sp500_symbols.txt --type ratios --period quarter --output sp500_ratios_quarterly.csv
```

## 📊 What Data You Get

### Historical Market Cap
- **Coverage**: 2000-2024 (daily)
- **Output**: `date, symbol, marketCap`
- **Use**: Identify megacap stocks at each point in time for V30 strategy

### Fundamental Ratios
- **Coverage**: 2000-2024 (quarterly or annual)
- **Metrics**: P/E ratio, EPS, ROE, ROA, debt ratios, profit margins, etc.
- **Output**: `date, symbol, peRatio, eps, roeTTM, roaTTM, debtToEquity, currentRatio, ...`
- **Use**: ML features for V30 stock ranking

## ⚠️ Limitations

1. **Coverage starts ~2000** (not 1990)
   - For 1990-1999 data, need IEX Cloud ($9/month)
   - Or start V30 backtest from 2000 (24 years still good!)

2. **Rate Limits** (Free/Starter tier)
   - 250 API calls per day
   - Downloading 500 stocks = 2 days
   - Scripts handle rate limiting automatically

3. **Network Requirements**
   - Must run from network without proxy blocking
   - Home network or mobile hotspot recommended

## 💰 Pricing

**Current**: FREE tier (250 calls/day)
**Upgrade**: $15/month Starter plan (same 250 calls/day, faster support)
**Upgrade**: $30/month Professional (750 calls/day, 3x faster downloads)

## 📚 Documentation

See `FMP_DOWNLOAD_GUIDE.md` for:
- Detailed usage examples
- All command-line options
- Troubleshooting guide
- Batch download strategies
- Expected output formats

## 🎯 For V30 Strategy

After downloading data:

1. **Market Cap** → Use for megacap filtering (top 10-20 stocks at each date)
2. **Fundamentals** → Use as ML features (P/E, EPS, ROE for stock ranking)
3. **Integration** → Load CSV files in V30 backtest code

## ✅ Next Steps

1. Run from HOME network
2. Test with single stock: `python3 fetch_fmp_market_cap.py --symbol AAPL`
3. Download test dataset (10 stocks): `python3 fetch_fmp_market_cap.py --file test_symbols.txt`
4. Verify data quality
5. Download full S&P 500 (2 days)
6. Integrate into V30 backtest

## 🔗 Links

- **FMP Website**: https://financialmodelingprep.com
- **API Docs**: https://site.financialmodelingprep.com/developer/docs
- **Pricing**: https://financialmodelingprep.com/developer/docs/pricing

---

**Need help?** Check `FMP_DOWNLOAD_GUIDE.md` or test the API with `python3 test_fmp_api.py`
