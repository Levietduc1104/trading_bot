# IEX Cloud Setup Guide - Step by Step

Complete guide to set up IEX Cloud and download historical market cap data for V30 strategy.

---

## 📋 What You'll Get

- **Historical market cap**: 1990-2024 (34 years)
- **Daily values**: Market capitalization for each trading day
- **Coverage**: All S&P 500 stocks
- **Use case**: Identify megacap stocks for V30 strategy at each point in time

---

## 💰 Cost

**IEX Cloud Starter Plan**: $9/month
- 100,000 API calls/month
- Historical data back to 1990
- Core data feeds
- Cancel anytime

---

## 🚀 Step-by-Step Setup

### **Step 1: Sign Up for IEX Cloud (5 minutes)**

1. **Go to IEX Cloud website:**
   - URL: https://iexcloud.io/

2. **Click "Sign Up" or "Get Started"**

3. **Create account:**
   - Enter email address
   - Create password
   - Verify email

4. **Choose plan:**
   - Select **"Starter"** plan ($9/month)
   - Click **"Subscribe"**

5. **Enter payment info:**
   - Credit card details
   - Billing address
   - Complete purchase

6. **Confirmation:**
   - You'll receive confirmation email
   - Account is now active

---

### **Step 2: Get Your API Token (2 minutes)**

1. **Log into IEX Cloud Console:**
   - URL: https://iexcloud.io/console/

2. **Navigate to API Tokens:**
   - Look for "API Tokens" in sidebar
   - Or go to: https://iexcloud.io/console/tokens

3. **Copy your token:**
   - You'll see **"Publishable Token"**
   - Starts with `pk_`
   - Example: `pk_abc123def456ghi789...`
   - Click **"Copy"** button

4. **Save it somewhere safe:**
   - Paste into a text file temporarily
   - You'll add it to scripts in next step

---

### **Step 3: Configure Scripts (3 minutes)**

1. **Navigate to iex_data_fetcher folder:**

```bash
cd /Users/levietduc/Documents/Documents\ -\ Le\'s\ MacBook\ Pro/Learning/ml_tranning/trading_bot/src/tool/iex_data_fetcher
```

2. **Edit test script:**

```bash
# Open test_iex_api.py in your editor
nano test_iex_api.py

# Find this line (around line 8):
IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'

# Replace with your actual token:
IEX_API_KEY = 'pk_abc123def456ghi789...'  # Your actual token from Step 2

# Save and exit: Ctrl+O, Enter, Ctrl+X
```

3. **Edit fetch script:**

```bash
# Open fetch_iex_market_cap.py in your editor
nano fetch_iex_market_cap.py

# Find this line (around line 14):
IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'

# Replace with your actual token:
IEX_API_KEY = 'pk_abc123def456ghi789...'  # Same token as above

# Save and exit: Ctrl+O, Enter, Ctrl+X
```

---

### **Step 4: Test API Connection (2 minutes)**

```bash
# Run test script
python3 test_iex_api.py
```

**Expected output:**

```
================================================================================
IEX CLOUD API TEST
================================================================================
API Key: pk_abc123def456ghi789...
Test Symbol: AAPL
================================================================================

================================================================================
Testing: Quote (current price + market cap)
================================================================================
✅ SUCCESS
{
  "symbol": "AAPL",
  "companyName": "Apple Inc",
  "marketCap": 3759400000000,
  ...
}

================================================================================
Testing: Stats (company statistics)
================================================================================
✅ SUCCESS
...

================================================================================
SUMMARY
================================================================================

✅ Working endpoints: 5
   - Quote (current price + market cap)
   - Stats (company statistics)
   - Historical Prices (5 years)
   - Company Information
   - Time Series - Historical Market Cap

================================================================================
✅ API KEY WORKS! Ready to download data.

Next steps:
1. Edit fetch_iex_market_cap.py and add your API key
2. Run: python3 fetch_iex_market_cap.py --symbol AAPL --years 10
================================================================================
```

**If you see errors:**
- ❌ "API key not set" → Go back to Step 3
- ❌ "401 Unauthorized" → Check API key is correct, try copying again
- ❌ "Connection error" → Check network (proxy blocking?)

---

### **Step 5: Download Test Data (5 minutes)**

**Test with single stock:**

```bash
python3 fetch_iex_market_cap.py --symbol AAPL --years 10
```

**Expected output:**

```
🚀 IEX Cloud Market Cap Downloader
API Key: pk_abc123def456ghi789...
Symbols: 1
Years: 10 (back to 2014)

================================================================================
Downloading market cap for 1 stocks (10 years)
IEX Cloud - Rate limit delay: 0.5 seconds
================================================================================

[1/1] Fetching AAPL... ✅ 2517 records (2014-02-14 to 2024-02-14)

================================================================================
✅ Saved 2517 records to: iex_market_cap_20240214_231545.csv
================================================================================

Data summary:
  Symbols: 1
  Date range: 2014-02-14 to 2024-02-14
  Total records: 2517

📄 Sample data:
       date symbol      market_cap  market_cap_billions
2014-02-14   AAPL  450000000000           450.00
2014-02-15   AAPL  452000000000           452.00
...
```

**Test with multiple stocks:**

```bash
python3 fetch_iex_market_cap.py --file test_symbols.txt --years 5 --output test_market_cap.csv
```

**Expected output:**

```
[1/10] Fetching AAPL... ✅ 1258 records (2019-02-14 to 2024-02-14)
[2/10] Fetching MSFT... ✅ 1258 records (2019-02-14 to 2024-02-14)
[3/10] Fetching GOOGL... ✅ 1258 records (2019-02-14 to 2024-02-14)
...
[10/10] Fetching JNJ... ✅ 1258 records (2019-02-14 to 2024-02-14)

✅ SUCCESS! Data saved to: test_market_cap.csv
```

---

### **Step 6: Download Full S&P 500 (1 hour total)**

**Create S&P 500 symbols file:**

```bash
# Extract symbols from your existing S&P 500 list
python3 << 'EOF'
import json

# Load your S&P 500 list
# Adjust path based on where your sp500_official_list.json is
with open('../../sp500_data/sp500_official_list.json', 'r') as f:
    data = json.load(f)

# Extract symbols (adjust based on your JSON structure)
if isinstance(data, list):
    symbols = data
elif isinstance(data, dict):
    # Try common keys
    if 'symbols' in data:
        symbols = data['symbols']
    elif 'tickers' in data:
        symbols = data['tickers']
    else:
        symbols = list(data.keys())
else:
    symbols = []

# Take first 500 (S&P 500)
symbols = symbols[:500]

# Save to txt file
with open('sp500_symbols.txt', 'w') as f:
    for sym in symbols:
        if isinstance(sym, str):
            f.write(f"{sym}\n")
        elif isinstance(sym, dict) and 'symbol' in sym:
            f.write(f"{sym['symbol']}\n")

print(f"✅ Created sp500_symbols.txt with {len(symbols)} symbols")
EOF
```

**Download historical market cap (1990-2024):**

```bash
# This will take ~10 minutes for 500 stocks
python3 fetch_iex_market_cap.py \
    --file sp500_symbols.txt \
    --years 34 \
    --output sp500_market_cap_1990_2024.csv \
    --delay 0.5
```

**Expected output:**

```
🚀 IEX Cloud Market Cap Downloader
API Key: pk_abc123def456ghi789...
Symbols: 500
Years: 34 (back to 1990)

================================================================================
Downloading market cap for 500 stocks (34 years)
IEX Cloud - Rate limit delay: 0.5 seconds
================================================================================

[1/500] Fetching A... ✅ 8570 records (1990-01-02 to 2024-02-14)
[2/500] Fetching AAL... ✅ 6234 records (1996-05-15 to 2024-02-14)
[3/500] Fetching AAPL... ✅ 8570 records (1990-01-02 to 2024-02-14)
...
[500/500] Fetching ZTS... ✅ 3024 records (2013-01-02 to 2024-02-14)

================================================================================
✅ Saved 3,458,234 records to: sp500_market_cap_1990_2024.csv
================================================================================

Data summary:
  Symbols: 500
  Date range: 1990-01-02 to 2024-02-14
  Total records: 3,458,234

✅ SUCCESS! Data saved to: sp500_market_cap_1990_2024.csv
```

---

## 📊 Verify Downloaded Data

```bash
# Check the CSV file
head -20 sp500_market_cap_1990_2024.csv

# Count records
wc -l sp500_market_cap_1990_2024.csv

# Check date range and symbols with Python
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('sp500_market_cap_1990_2024.csv')

print(f"Total records: {len(df)}")
print(f"Symbols: {df['symbol'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSample data:")
print(df.head(10))
print(f"\nMarket cap range:")
print(df['market_cap_billions'].describe())
EOF
```

---

## ✅ Success Checklist

- [x] Signed up for IEX Cloud Starter ($9/month)
- [x] Got API token from console
- [x] Added token to both scripts
- [x] Tested API with `test_iex_api.py` (✅ works)
- [x] Downloaded single stock (AAPL 10 years)
- [x] Downloaded test dataset (10 stocks, 5 years)
- [x] Created sp500_symbols.txt (500 symbols)
- [x] Downloaded full S&P 500 (1990-2024)
- [x] Verified CSV data looks correct

---

## 🎯 Next Steps: Integration with V30

Now that you have historical market cap data, integrate into V30 strategy:

### **1. Load market cap data in V30:**

```python
import pandas as pd

# Load historical market cap
market_caps = pd.read_csv('sp500_market_cap_1990_2024.csv')
market_caps['date'] = pd.to_datetime(market_caps['date'])
```

### **2. Get megacap stocks at each date:**

```python
def get_megacap_stocks(date, top_n=20):
    """Get top N stocks by market cap on specific date"""
    day_data = market_caps[market_caps['date'] == date]
    top_stocks = day_data.nlargest(top_n, 'market_cap_billions')
    return top_stocks['symbol'].tolist()
```

### **3. Use in V30 strategy:**

```python
# At each quarterly rebalance
rebalance_date = '2000-01-01'
megacap_universe = get_megacap_stocks(rebalance_date, top_n=20)

# Apply ML ranking to megacap universe only
# Then select top 10 from ML rankings
```

---

## 💡 Tips

### **API Call Usage:**
- Each stock = 1 API call
- 500 stocks = 500 calls
- 100K calls/month = plenty for experimentation

### **Speed Up Downloads:**
- Use `--delay 0.3` for faster (but stay safe)
- Default `--delay 0.5` is safe and not too slow

### **Batch Downloads:**
- Can split symbols into batches
- Download different time periods separately
- Combine CSV files later

---

## 🆘 Troubleshooting

### **Problem: "No data returned"**
**Solution:**
- Check symbol is correct (uppercase, no spaces)
- Some stocks don't exist back to 1990 (IPO date later)
- Try well-known stock like AAPL first

### **Problem: "Rate limit exceeded"**
**Solution:**
- Wait a few minutes
- Increase `--delay` value
- Starter plan has 100K calls/month (plenty for 500 stocks)

### **Problem: "Connection refused / Proxy error"**
**Solution:**
- Run from home network
- Not work/school network (proxy blocks)
- Try mobile hotspot

### **Problem: "API key unauthorized"**
**Solution:**
- Copy API key again from IEX console
- Make sure it's the Publishable token (starts with `pk_`)
- Check subscription is active

---

## 📞 Support

**IEX Cloud Support:**
- Email: support@iexcloud.io
- Documentation: https://iexcloud.io/docs/
- Console: https://iexcloud.io/console/

**Check subscription status:**
- Login to: https://iexcloud.io/console/
- View: Account → Subscription → Usage

---

## ✅ Done!

You now have:
- ✅ IEX Cloud subscription active
- ✅ Historical market cap data 1990-2024
- ✅ Ready to integrate into V30 backtest

**Total setup time:** ~1 hour
**Monthly cost:** $9
**Data quality:** Professional-grade

**Ready to backtest V30 from 1990-2024 with accurate megacap filtering!** 🎉
