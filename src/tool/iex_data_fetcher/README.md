# IEX Cloud Data Fetcher

Download historical market capitalization data from 1990-2024 using IEX Cloud API.

## 📁 Folder Contents

```
iex_data_fetcher/
├── README.md                      # This file
├── SETUP_GUIDE.md                # Step-by-step setup guide
├── test_iex_api.py               # Test API connection
├── fetch_iex_market_cap.py       # Download historical market cap
└── test_symbols.txt              # Test symbols
```

## ⭐ Why IEX Cloud?

✅ **Historical market cap: 1990-2024** (exactly what V30 needs!)
✅ **Daily market cap values**
✅ **100K API calls/month** (Starter plan)
✅ **Stable, reliable API** (used by professionals)
✅ **Only $9/month** (cheaper than alternatives)

## 🚀 Quick Start

### **Step 1: Sign Up for IEX Cloud**

1. Go to: https://iexcloud.io/
2. Click **"Sign Up"**
3. Choose **"Starter"** plan ($9/month)
4. Complete registration

### **Step 2: Get API Token**

1. Log into IEX Cloud console
2. Go to **"API Tokens"** section
3. Copy your **Publishable Token** (starts with `pk_`)
4. Save it (you'll need it in next step)

### **Step 3: Configure Scripts**

```bash
cd /Users/levietduc/Documents/Documents\ -\ Le\'s\ MacBook\ Pro/Learning/ml_tranning/trading_bot/src/tool/iex_data_fetcher

# Edit test_iex_api.py
nano test_iex_api.py
# Replace: IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'
# With: IEX_API_KEY = 'pk_your_actual_key_here'

# Edit fetch_iex_market_cap.py
nano fetch_iex_market_cap.py
# Replace: IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'
# With: IEX_API_KEY = 'pk_your_actual_key_here'
```

### **Step 4: Test API**

```bash
python3 test_iex_api.py
```

**Expected output:**
```
✅ Working endpoints: 4
   - Quote (current price + market cap)
   - Stats (company statistics)
   - Historical Prices (5 years)
   - Time Series - Historical Market Cap

✅ API KEY WORKS! Ready to download data.
```

### **Step 5: Download Data**

```bash
# Test with single stock
python3 fetch_iex_market_cap.py --symbol AAPL --years 10

# Test with multiple stocks
python3 fetch_iex_market_cap.py --symbols AAPL,MSFT,GOOGL --years 20

# Download for test symbols
python3 fetch_iex_market_cap.py --file test_symbols.txt --years 34 --output test_market_cap.csv
```

---

## 📊 What Data You Get

### **CSV Output:**
```csv
date,symbol,market_cap,market_cap_billions
1990-01-02,AAPL,4500000000,4.5
1990-01-03,AAPL,4520000000,4.52
...
2024-12-31,AAPL,3759400000000,3759.4
```

### **Coverage:**
- **Period**: 1990-2024 (34 years) ✅
- **Frequency**: Daily values
- **Data**: Market capitalization in dollars

### **Use for V30:**
- Identify megacap stocks (top 10-20) at each date
- Avoid survivorship bias
- Avoid lookahead bias
- Accurate historical backtesting

---

## 💰 Pricing

### **Starter Plan ($9/month):**
- 100,000 API calls/month
- Historical data back to 1990
- Core data feeds
- Perfect for V30 strategy

### **Usage Estimate:**
- 500 stocks × 1 API call each = 500 calls
- Well under 100K limit! ✅
- Can download full S&P 500 easily

---

## 📝 Download Full S&P 500

### **Create symbols file:**

```bash
# If you have sp500_official_list.json
python3 << 'EOF'
import json

with open('../../sp500_data/sp500_official_list.json', 'r') as f:
    data = json.load(f)

# Extract symbols (adjust based on your JSON structure)
if isinstance(data, list):
    symbols = data
else:
    symbols = list(data.keys())[:500]

# Save to txt
with open('sp500_symbols.txt', 'w') as f:
    for sym in symbols:
        f.write(f"{sym}\n")

print(f"✅ Created sp500_symbols.txt with {len(symbols)} symbols")
EOF
```

### **Download market cap:**

```bash
# Download 1990-2024 for all S&P 500
python3 fetch_iex_market_cap.py \
    --file sp500_symbols.txt \
    --years 34 \
    --output sp500_market_cap_1990_2024.csv
```

**Time**: ~5-10 minutes for 500 stocks

---

## 🎯 Integration with V30

After downloading, use the market cap data in your V30 strategy:

```python
import pandas as pd

# Load historical market cap
market_caps = pd.read_csv('sp500_market_cap_1990_2024.csv')
market_caps['date'] = pd.to_datetime(market_caps['date'])

def get_megacap_stocks(date, top_n=20):
    """Get top N stocks by market cap on specific date"""
    day_data = market_caps[market_caps['date'] == date]
    top_stocks = day_data.nlargest(top_n, 'market_cap_billions')
    return top_stocks['symbol'].tolist()

# Example: Get top 20 stocks on 2000-01-01
top_20_2000 = get_megacap_stocks('2000-01-01', top_n=20)
print(f"Top 20 in 2000: {top_20_2000}")

# Example: Get top 20 stocks on 2024-01-01
top_20_2024 = get_megacap_stocks('2024-01-01', top_n=20)
print(f"Top 20 in 2024: {top_20_2024}")
```

---

## ⚠️ Important Notes

### **API Rate Limits:**
- Starter plan: 100K calls/month
- Scripts use 0.5s delay (safe, not too slow)
- Can download 500 stocks in ~5 minutes

### **Data Coverage:**
- IEX Cloud has data back to 1990
- Some stocks may have gaps before IPO date
- Daily market cap values (trading days only)

### **Network:**
- Must run from network without proxy blocking
- Home network or mobile hotspot

---

## 🔧 Troubleshooting

### **Error: "API key not set"**
**Solution**: Edit scripts and replace `YOUR_IEX_API_KEY_HERE` with your actual token

### **Error: "401 Unauthorized"**
**Solution**:
- Check API key is correct
- Verify IEX Cloud subscription is active
- Try copying key again from console

### **Error: "No data"**
**Solution**:
- Check symbol is correct (e.g., AAPL not Apple)
- Check IEX Cloud has data for that symbol
- Try a different well-known symbol (MSFT, GOOGL)

### **Error: "Proxy blocking"**
**Solution**: Run from home network, not work/school network

---

## 📚 Additional Resources

- **IEX Cloud Console**: https://iexcloud.io/console/
- **IEX Documentation**: https://iexcloud.io/docs/
- **Pricing**: https://iexcloud.io/pricing
- **Support**: support@iexcloud.io

---

## ✅ Summary

**What you need:**
1. IEX Cloud Starter subscription ($9/month)
2. API token from console
3. Add token to scripts

**What you get:**
- Historical market cap 1990-2024
- Daily values for all S&P 500 stocks
- Perfect for V30 megacap filtering

**Setup time:** 15 minutes
**Download time:** 5-10 minutes for 500 stocks
**Monthly cost:** $9

---

## 🎯 Next Steps

1. ✅ Sign up for IEX Cloud: https://iexcloud.io/
2. ✅ Get API token from console
3. ✅ Add token to scripts (replace `YOUR_IEX_API_KEY_HERE`)
4. ✅ Run `python3 test_iex_api.py` to verify
5. ✅ Download data: `python3 fetch_iex_market_cap.py --symbol AAPL`
6. ✅ Download S&P 500: `python3 fetch_iex_market_cap.py --file sp500_symbols.txt --years 34`

**Ready to start!** 🚀
