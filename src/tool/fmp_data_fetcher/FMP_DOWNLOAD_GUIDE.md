# FMP Data Download Guide

## 🚀 Quick Start

Your FMP API key is already configured in the scripts: `yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w`

⚠️ **IMPORTANT**: Run these scripts from **HOME network** (not work/school network - proxy blocks API calls)

---

## 📁 Scripts Available

### 1. **`test_fmp_api.py`** - Test FMP API
Check what data is available and date ranges

```bash
cd src/tool
python3 test_fmp_api.py
```

**What it does**: Tests all FMP endpoints and shows date coverage

---

### 2. **`fetch_fmp_market_cap.py`** - Download Historical Market Cap
Get daily market cap data (2000-2024)

```bash
# Test with single stock
python3 fetch_fmp_market_cap.py --symbol AAPL

# Test with multiple stocks
python3 fetch_fmp_market_cap.py --symbols AAPL,MSFT,GOOGL

# Download for 10 test stocks
python3 fetch_fmp_market_cap.py --file test_symbols.txt --output test_market_cap.csv

# Download for all S&P 500 (create sp500_symbols.txt first)
python3 fetch_fmp_market_cap.py --file sp500_symbols.txt --output sp500_market_cap.csv
```

**Output**: CSV with columns: `date`, `symbol`, `marketCap`

---

### 3. **`fetch_fmp_fundamentals.py`** - Download Fundamental Ratios
Get P/E, EPS, ROE, ROA, and more (quarterly or annual)

```bash
# Test with single stock (comprehensive data)
python3 fetch_fmp_fundamentals.py --symbol AAPL --period quarter

# Test with multiple stocks
python3 fetch_fmp_fundamentals.py --symbols AAPL,MSFT,GOOGL --period quarter

# Download for 10 test stocks
python3 fetch_fmp_fundamentals.py --file test_symbols.txt --type comprehensive --output test_fundamentals.csv

# Download for all S&P 500 (quarterly ratios only - faster)
python3 fetch_fmp_fundamentals.py --file sp500_symbols.txt --type ratios --period quarter --output sp500_ratios_quarterly.csv
```

**Output**: CSV with columns: `date`, `symbol`, `peRatio`, `eps`, `roeTTM`, `roaTTM`, `debtToEquity`, `currentRatio`, etc.

---

## 📊 Data Types Available

### **Market Cap** (`fetch_fmp_market_cap.py`)
- Daily market capitalization
- Coverage: 2000-2024
- Use for: Identifying megacap stocks at each point in time

### **Financial Ratios** (`--type ratios`)
- P/E ratio, P/B ratio, P/S ratio
- ROE, ROA, profit margins
- Debt ratios, liquidity ratios
- Coverage: ~2000-2024 (quarterly and annual)

### **Key Metrics** (`--type metrics`)
- Market cap, enterprise value
- Revenue per share, book value per share
- Coverage: ~2000-2024

### **Income Statement** (`--type income`)
- Revenue, net income, EPS
- Operating income, gross profit
- Coverage: ~2000-2024

### **Comprehensive** (`--type comprehensive`)
- All of the above combined
- Most complete dataset
- Slower to download (3 API calls per stock)

---

## ⏱️ Download Time Estimates

### **FMP Free Tier** (250 calls/day)
- Single stock: ~5 seconds
- 10 stocks: ~1 minute
- 100 stocks: ~10 minutes
- 500 stocks: **2 days** (250/day limit)

### **FMP Starter ($15/month)** (250 calls/day)
- Same as free tier

### **FMP Professional ($30/month)** (750 calls/day)
- 500 stocks: **1 day**

### **Speed Up Tips**:
- Use `--type ratios` instead of `comprehensive` (1 API call vs 3)
- Download in batches (250 stocks per day)
- Use `--delay 0.5` to speed up (but stay within rate limits)

---

## 📝 Create S&P 500 Symbols File

You need a file with all S&P 500 symbols (one per line):

```bash
# If you have sp500_official_list.json
python3 << 'EOF'
import json

with open('../sp500_data/sp500_official_list.json', 'r') as f:
    data = json.load(f)

# Extract symbols
if isinstance(data, list):
    symbols = data
elif isinstance(data, dict) and 'symbols' in data:
    symbols = data['symbols']
else:
    symbols = []

# Save to txt
with open('sp500_symbols.txt', 'w') as f:
    for sym in symbols:
        f.write(f"{sym}\n")

print(f"Created sp500_symbols.txt with {len(symbols)} symbols")
EOF
```

---

## 🎯 Recommended Workflow

### **Step 1: Test (5 minutes)**

```bash
cd src/tool

# Test API works
python3 test_fmp_api.py

# Test market cap download (single stock)
python3 fetch_fmp_market_cap.py --symbol AAPL

# Test fundamentals download (single stock)
python3 fetch_fmp_fundamentals.py --symbol AAPL --period quarter
```

### **Step 2: Download Test Dataset (10 minutes)**

```bash
# Download market cap for 10 test stocks
python3 fetch_fmp_market_cap.py --file test_symbols.txt --output test_market_cap.csv

# Download fundamentals for 10 test stocks
python3 fetch_fmp_fundamentals.py --file test_symbols.txt --type comprehensive --output test_fundamentals.csv
```

### **Step 3: Verify Data Quality**

```bash
# Check the CSV files
head test_market_cap.csv
head test_fundamentals.csv

# Check date ranges
python3 << 'EOF'
import pandas as pd

# Market cap
df_mc = pd.read_csv('test_market_cap.csv')
print("Market Cap:")
print(f"  Date range: {df_mc['date'].min()} to {df_mc['date'].max()}")
print(f"  Records: {len(df_mc)}")
print(f"  Symbols: {df_mc['symbol'].nunique()}")

# Fundamentals
df_fa = pd.read_csv('test_fundamentals.csv')
print("\nFundamentals:")
print(f"  Date range: {df_fa['date'].min()} to {df_fa['date'].max()}")
print(f"  Records: {len(df_fa)}")
print(f"  Symbols: {df_fa['symbol'].nunique()}")
print(f"  Columns: {list(df_fa.columns[:10])}")
EOF
```

### **Step 4: Download Full S&P 500 (2 days with free tier)**

```bash
# Create S&P 500 symbols file first (see above)

# Download market cap (500 stocks)
python3 fetch_fmp_market_cap.py --file sp500_symbols.txt --output sp500_market_cap_2000_2024.csv

# Download fundamentals (500 stocks) - TAKES 2 DAYS WITH FREE TIER
# Option A: Comprehensive (3 calls/stock = 1500 total calls = 6 days)
python3 fetch_fmp_fundamentals.py --file sp500_symbols.txt --type comprehensive --output sp500_fundamentals_comprehensive.csv

# Option B: Ratios only (1 call/stock = 500 total calls = 2 days) ✅ RECOMMENDED
python3 fetch_fmp_fundamentals.py --file sp500_symbols.txt --type ratios --period quarter --output sp500_ratios_quarterly.csv
```

---

## 💡 Tips & Tricks

### **Speed Up Downloads**
```bash
# Reduce delay between requests (default 1.0s)
python3 fetch_fmp_market_cap.py --file test_symbols.txt --delay 0.5

# Warning: Too fast might hit rate limits
```

### **Download in Batches**
```bash
# Split symbols file into batches
split -l 250 sp500_symbols.txt sp500_batch_

# Download batch 1 (day 1)
python3 fetch_fmp_market_cap.py --file sp500_batch_aa --output batch1.csv

# Download batch 2 (day 2)
python3 fetch_fmp_market_cap.py --file sp500_batch_ab --output batch2.csv

# Combine batches
cat batch1.csv batch2.csv > sp500_market_cap_full.csv
```

### **Resume Failed Downloads**
If download fails midway, create a new file with remaining symbols and continue

---

## ❌ Troubleshooting

### **Error: "Proxy connection failed"**
**Solution**: Run from HOME network (not work/school network)

### **Error: "API rate limit exceeded"**
**Solution**:
- Wait 24 hours for rate limit reset
- Or upgrade to paid tier ($15/month = 250 calls/day)
- Or reduce --delay to stay within limits

### **Error: "No data returned for symbol"**
**Possible reasons**:
- Stock delisted or not in FMP database
- Symbol incorrect (check spelling)
- Stock too new (no historical data)

### **Data looks incomplete**
- Some stocks have limited history before 2010
- Newer companies (TSLA, NVDA IPO dates) have less data
- Check date range in output

---

## 📊 Expected Output

### **Market Cap CSV**:
```csv
date,symbol,marketCap
2000-01-03,AAPL,15234000000
2000-01-04,AAPL,14987000000
...
2024-12-31,AAPL,3759400000000
```

### **Fundamentals CSV (Comprehensive)**:
```csv
date,symbol,peRatio,pbRatio,roeTTM,roaTTM,debtToEquity,currentRatio,revenue,netIncome,eps
2000-03-31,AAPL,28.5,3.2,0.15,0.08,0.25,2.1,5678000000,234000000,0.15
2000-06-30,AAPL,30.2,3.5,0.16,0.09,0.23,2.3,6123000000,267000000,0.17
...
```

---

## ✅ Summary

**Created scripts:**
- ✅ `test_fmp_api.py` - Test API and check data availability
- ✅ `fetch_fmp_market_cap.py` - Download historical market cap
- ✅ `fetch_fmp_fundamentals.py` - Download fundamental ratios
- ✅ `test_symbols.txt` - 10 test symbols to start with

**Next steps:**
1. Run from HOME network
2. Test with single stock
3. Download test dataset (10 stocks)
4. Download full S&P 500 (2 days)

**Need help?** Check the examples in this README!
