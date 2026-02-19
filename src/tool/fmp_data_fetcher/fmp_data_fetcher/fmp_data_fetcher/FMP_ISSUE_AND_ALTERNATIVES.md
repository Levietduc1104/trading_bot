# FMP API Issue - Legacy Endpoints Deprecated

## ❌ Problem

Your FMP API test showed:
```
"Legacy Endpoint : Due to Legacy endpoints being no longer supported -
This endpoint is only available for legacy users who have valid subscriptions
prior August 31, 2025."
```

**What this means:**
- The `/api/v3/` endpoints I used are DEPRECATED
- FMP moved to `/stable/` endpoints
- Your free API key may not have access to historical data anymore
- FMP may have restricted free tier access

---

## 🔍 What Happened

FMP recently restructured their API:
- **Old**: `https://financialmodelingprep.com/api/v3/...`
- **New**: `https://financialmodelingprep.com/stable/...`

The free tier may no longer include:
- Historical market capitalization
- Historical financial ratios
- Quarterly financial statements (historical)

---

## ✅ RECOMMENDED SOLUTIONS

### **Option 1: IEX Cloud ($9/month)** ⭐ **BEST FOR V30**

**What you get:**
- ✅ Historical market cap: 1990-2024 (exactly what you need!)
- ✅ Daily market cap values
- ✅ Clean, reliable API
- ✅ 100K API calls/month
- ✅ No "legacy endpoint" issues

**Coverage:**
- Market cap: 1990-2024 (34 years) ✅
- For fundamentals: You already have Alpha Vantage (current data)

**Cost:** $9/month

**Sign up:** https://iexcloud.io/pricing

**Why IEX instead of FMP:**
1. Longer history (1990 vs 2000)
2. More reliable (no sudden API changes)
3. Better documentation
4. Used by institutional traders

---

### **Option 2: Alpha Vantage (FREE)** ✅ **YOU ALREADY HAVE THIS**

**What you get:**
- ✅ Current fundamentals: P/E, EPS, ROE, ROA, etc.
- ✅ 500 calls/day (enough for V30)
- ✅ Your API key: `PEI9KPIV5GAG81KZ`

**Limitation:**
- ❌ Only CURRENT fundamentals (no historical)
- ❌ No historical market cap

**Use for:**
- V30 quarterly rebalancing (get current fundamentals)
- Quality filtering (current P/E, ROE, etc.)

**Scripts already working:**
- `src/live_trading/fundamental_data_fetchers/get_alpha_vantage_data.py`
- `src/live_trading/fundamental_data_fetchers/screen_quality_stocks.py`

---

### **Option 3: Upgrade FMP ($15-30/month)**

**Contact FMP support:**
- Email: support@financialmodelingprep.com
- Ask if paid plans have access to `/stable/` historical endpoints
- Check if they still offer historical market cap

**Before paying:**
- ❌ Unclear if they still offer 1990-2024 data
- ❌ Recent API changes suggest free tier very limited
- ⚠️ IEX Cloud is cheaper ($9 vs $15) with better history

---

### **Option 4: Hybrid Approach** ⭐ **PRACTICAL SOLUTION**

**For V30 backtesting (2000-2024):**

1. **Market Cap**: Use IEX Cloud ($9/month)
   - Download historical market cap 1990-2024
   - Use for megacap filtering

2. **Current Fundamentals**: Use Alpha Vantage (FREE)
   - Get current P/E, EPS, ROE at each rebalance
   - Sufficient for quarterly strategy

3. **Historical Fundamentals**: Optional
   - V30 can work without historical fundamentals
   - ML model uses mostly technical features
   - Test if historical FA actually improves Sharpe ratio

**Total cost:** $9/month
**Coverage:** 1990-2024 market cap + current fundamentals

---

## 🎯 MY RECOMMENDATION

### **Best Path Forward:**

**Step 1: Test V30 without historical fundamentals**
```bash
# Your ML model can work with technical features only:
- EMA ratios, momentum, RSI, ATR
- Relative strength vs SPY
- Volume patterns
```

**If Sharpe > 1.5 without fundamentals:**
- Just get IEX Cloud for market cap ($9/month)
- Use Alpha Vantage for current screening
- Done! ✅

**If fundamentals significantly improve Sharpe:**
- Add IEX Cloud for market cap ($9/month)
- For 2000+ you can try other sources
- Or use technical features for 1990-1999

---

## 📊 What Data You Actually Need for V30

Looking at your ML strategy (`ml_stock_ranker_lgbm.py`):

### **Critical (Must Have):**
- ✅ Historical price data → You have this (1963-2024)
- ✅ Market cap (for megacap filter) → Need IEX Cloud

### **Optional (Might Improve):**
- ⚠️ P/E ratio → Alpha Vantage (current only)
- ⚠️ EPS → Alpha Vantage (current only)
- ⚠️ ROE, ROA → Alpha Vantage (current only)

### **The Key Insight:**
V30 rebalances **quarterly**. You only need fundamentals at each rebalance date (4 times/year), not daily historical values. Alpha Vantage current data might be sufficient!

---

## 💡 PRACTICAL TEST

**Before spending money, test this:**

1. **Run V30 backtest 2000-2024 with technical features only**
   - No P/E, EPS, ROE in ML model
   - Just price-based features
   - Check Sharpe ratio

2. **Compare with Alpha Vantage current fundamentals**
   - At each quarterly rebalance, get current fundamentals
   - Add to ML features
   - Check if Sharpe improves

3. **Decide based on results:**
   - If fundamentals don't help much → Just get IEX for market cap ($9)
   - If fundamentals help a lot → Get IEX + consider historical FA source

---

## 🚀 IMMEDIATE ACTION PLAN

### **Today:**

1. ✅ Accept that FMP free tier doesn't work for historical data
2. ✅ Stop trying to fix FMP scripts (waste of time)
3. ✅ Focus on proven alternatives

### **This Week:**

**Option A: Minimal Setup ($9/month)**
```bash
# Subscribe to IEX Cloud
# Download market cap 1990-2024
# Use Alpha Vantage for current fundamentals
# Run V30 backtest
```

**Option B: Test First (FREE)**
```bash
# Run V30 with technical features only (1990-2024)
# Check if results are acceptable
# Add fundamentals only if needed
```

---

## 📝 Updated Scripts Needed

Since FMP doesn't work, I'll create:

1. **IEX Cloud fetcher** (for historical market cap)
2. **Alpha Vantage integration guide** (for current fundamentals)
3. **V30 backtest without historical fundamentals** (fallback)

Want me to create these scripts for IEX Cloud instead?

---

## ✅ Summary

**FMP Issue:**
- Legacy endpoints deprecated
- Free tier very limited
- Not worth the effort to fix

**Better Solution:**
- IEX Cloud ($9/month) for historical market cap 1990-2024
- Alpha Vantage (FREE) for current fundamentals
- Test if historical fundamentals even needed

**Total Cost:** $9/month (vs $15+ for FMP with unclear coverage)

**Next Step:** Should I create IEX Cloud fetcher scripts for you?
