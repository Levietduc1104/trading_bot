# Paper Trading Quick Start - Tier 2 Strategy

## Why Paper Trade?

Before risking real capital, **paper trade for 3 months minimum** to:
- ✅ Verify execution works correctly
- ✅ Test broker integration
- ✅ Build confidence with the strategy
- ✅ Catch any technical issues
- ✅ Understand quarterly rebalancing behavior

---

## Setup (30 minutes)

### 1. Choose Paper Trading Platform

**Recommended options:**

| Platform | Pros | Cons | Setup Time |
|----------|------|------|------------|
| Interactive Brokers Paper | Same as production, realistic | Requires IB account | 30 min |
| Alpaca Paper Trading | Free, easy API, good docs | Limited to US markets | 15 min |
| TD Ameritrade PaperMoney | User-friendly, no coding needed | Manual trading only | 20 min |

**Best choice:** Interactive Brokers Paper (what you'll use in production)

### 2. Interactive Brokers Paper Setup

**Step 1: Create paper trading account**
1. Go to: https://www.interactivebrokers.com/
2. Sign up for paper trading account (free)
3. Wait for approval (24-48 hours)

**Step 2: Install TWS or IB Gateway**
```bash
# Download from: https://www.interactivebrokers.com/en/trading/tws.php
# Choose: "Download TWS" or "Download IB Gateway"
# Install and log in with paper trading credentials
```

**Step 3: Enable API**
1. Open TWS/IB Gateway
2. Go to: File → Global Configuration → API → Settings
3. Enable: "Enable ActiveX and Socket Clients"
4. Set port: 7497 (paper trading) or 4001 (IB Gateway paper)
5. Add trusted IP: 127.0.0.1
6. Click OK and restart

**Step 4: Configure strategy**
```python
# In your config or execution script
BROKER = 'interactive_brokers'
PAPER_MODE = True  # Important!
API_HOST = '127.0.0.1'
API_PORT = 7497  # TWS paper, or 4001 for IB Gateway paper
CLIENT_ID = 1
```

### 3. Alpaca Paper Setup (Alternative - Easier)

**Step 1: Create free Alpaca account**
1. Go to: https://alpaca.markets/
2. Sign up (free, instant approval)
3. Get API keys from dashboard

**Step 2: Configure strategy**
```python
# In your config
BROKER = 'alpaca'
API_KEY = 'your_paper_key'
API_SECRET = 'your_paper_secret'
BASE_URL = 'https://paper-api.alpaca.markets'
```

---

## Running Paper Trading

### Start of Quarter (Jan 1, Apr 1, Jul 1, Oct 1)

**Recommended:** Start paper trading at the beginning of a quarter for best experience.

**Initial setup:**
```bash
# 1. Ensure broker API is running (TWS/IB Gateway or Alpaca)

# 2. Set initial capital
INITIAL_CAPITAL = 100000  # $100k for testing

# 3. Run initial backtest to get current positions
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024

# 4. Execute recommended positions in paper account
# (Top 3 mega-caps + top 7 momentum stocks from backtest output)
```

### Daily Checks (5 minutes)

**Every morning before market open:**

1. Check TWS/IB Gateway or Alpaca is running
2. Verify API connection working
3. Review any overnight position changes
4. Check for any error messages

**Quick check script:**
```python
# Create: check_paper_account.py
import ibapi  # or alpaca_trade_api
# ... check connection, positions, cash ...
print("Status: Connected ✅" if connected else "Status: Disconnected ❌")
print(f"Positions: {len(positions)}")
print(f"Cash: ${cash:,.0f}")
print(f"Portfolio Value: ${total_value:,.0f}")
```

Run daily:
```bash
python3 check_paper_account.py
```

### Weekly Checks (15 minutes)

**Every Monday:**

1. **Performance review:**
   ```bash
   # Check paper account returns
   Weekly return: Calculate from last Monday
   Compare to SPY: How did we do vs market?
   ```

2. **Position health:**
   - Any positions stopped out? (Should be rare)
   - Any positions up/down >10%?
   - Do holdings still match expected Tier 2 picks?

3. **Log observations:**
   ```
   Week of [DATE]:
   - Return: +X.X%
   - SPY return: +X.X%
   - Positions stopped: X
   - Notes: [any observations]
   ```

### End of Quarter (Mar 31, Jun 30, Sep 30, Dec 31)

**This is the critical test - verify rebalancing works correctly.**

**Rebalancing checklist:**

**Day before quarter end:**
- [ ] Run fresh backtest to get new positions
- [ ] Compare new vs current holdings
- [ ] Identify sells (exiting positions)
- [ ] Identify buys (new positions)
- [ ] Calculate target allocation (70% mega-cap, 30% momentum)

**First day of new quarter:**
- [ ] **9:30 AM:** Market opens, execute all sells first
- [ ] **9:35 AM:** Verify all sells executed, check cash available
- [ ] **9:40 AM:** Execute all buys at target allocation
- [ ] **9:45 AM:** Verify all buys executed correctly
- [ ] **10:00 AM:** Final check - compare actual vs target allocation

**Rebalancing command:**
```bash
# Get new quarter positions
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024

# Output shows:
# - SELL: [list of current positions to exit]
# - BUY: [list of new positions to enter]
# - Allocation: 70% to top 3, 30% to top 7
```

**After rebalancing:**
- [ ] Document transaction costs (should be <2% of portfolio)
- [ ] Verify covered calls sold (should collect ~1% premium)
- [ ] Check final positions match Tier 2 recommendations
- [ ] Update tracking spreadsheet

### Month-End Reviews (30 minutes)

**Last day of each month:**

1. **Calculate monthly return:**
   ```
   Monthly return = (End value - Start value) / Start value
   Compare to SPY monthly return
   ```

2. **Review performance:**
   - On track with 15-20% annual target?
   - Any concerning trends?
   - Transaction costs reasonable (<1% monthly)?

3. **Update tracking:**
   ```
   Month | Portfolio | Return | SPY | vs SPY | Drawdown | Notes
   -------|-----------|--------|-----|--------|----------|-------
   Jan    | $102,500  | +2.5%  | +2.0% | +0.5% | -3.2%   | Good start
   Feb    | $101,800  | -0.7%  | -1.2% | +0.5% | -3.5%   | Normal pullback
   Mar    | $105,200  | +3.3%  | +2.8% | +0.5% | -1.0%   | Strong close
   ```

---

## What to Watch For

### ✅ Good Signs (Everything Working)

- Paper account executing trades correctly
- Positions matching backtest recommendations
- Returns roughly tracking expectations (12-20% annual)
- Rebalancing executing smoothly every quarter
- Transaction costs <2% per quarter
- Covered calls collecting ~1% premium per quarter
- API connection stable, no errors

### ⚠️ Warning Signs (Pay Attention)

- Trades not executing as expected
- Positions differing from backtest output
- Returns significantly off expectations (>5% difference)
- Transaction costs >3% per quarter
- Frequent API disconnections
- Error messages during rebalancing
- Covered calls not executing

### 🚨 Red Flags (Stop and Fix)

- Rebalancing completely failed
- Losing more than -25% from peak
- Consistent API errors preventing trading
- Transaction costs >5% per quarter
- Positions completely wrong vs recommendations
- Cannot execute covered calls at all
- Fundamental technical issues

---

## 3-Month Graduation Checklist

**After 3 months of paper trading, review:**

### Technical Requirements ✅

- [ ] All 3 quarterly rebalances executed successfully
- [ ] No major technical errors encountered
- [ ] API connection stable throughout period
- [ ] Transaction costs reasonable (avg <2% per quarter)
- [ ] Covered calls working (collecting ~1% per quarter)
- [ ] Positions consistently matching recommendations
- [ ] Monitoring routine established (daily/weekly/monthly)

### Performance Requirements ✅

- [ ] Returns in expected range (3-6% quarterly, 12-24% annualized)
- [ ] Drawdowns reasonable (<-15% from peak)
- [ ] Tracking SPY performance reasonably close
- [ ] No catastrophic losing months (>-15%)
- [ ] Consistent with backtest expectations

### Psychological Requirements ✅

- [ ] Comfortable with quarterly rebalancing
- [ ] Accepting of losing weeks/months (they happen)
- [ ] Not tempted to override strategy picks
- [ ] Confident in systematic approach
- [ ] Ready to execute without emotion
- [ ] Understand this takes 3+ years to prove out

### If All Checked ✅ → Graduate to Live Trading

**Start with 25% of target capital:**
- If targeting $100k → start with $25k
- Run for 3 months, then increase to 50%
- After 6 months, increase to 75%
- After 9-12 months, deploy full 100%

**Why scale slowly?**
- Builds confidence gradually
- Limits risk if something goes wrong
- Allows adjustments before full capital deployed
- Psychologically easier than going "all in"

---

## Quick Reference Commands

```bash
# Check backtest results
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024

# Get current quarter positions
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024 | grep "Top 10 stocks"

# Run visualization
cd src/visualize && python3 visualize_trades.py

# Open dashboard
open src/visualize/trading_analysis.html

# Check paper account (create this script)
python3 check_paper_account.py

# Run multi-period test (verify strategy still robust)
python3 test_tier2_multi_period.py
```

---

## Troubleshooting

### "Trades not executing"
1. Check TWS/IB Gateway is running
2. Verify API enabled (File → Global Config → API)
3. Check port settings (7497 for TWS paper)
4. Try restarting TWS/IB Gateway

### "Positions don't match backtest"
1. Verify using same data source
2. Check market data is up to date
3. Ensure using latest code version
4. Run backtest again to get fresh recommendations

### "Returns way off expectations"
1. Check transaction costs are enabled in backtest
2. Verify covered calls are enabled
3. Ensure using quarterly rebalancing
4. Compare to SPY - is market performing unusually?

### "API keeps disconnecting"
1. Check firewall settings
2. Verify port 7497 not blocked
3. Add 127.0.0.1 to TWS trusted IPs
4. Try IB Gateway instead of TWS (more stable)

### "Can't collect covered call premiums"
1. Verify options permissions enabled in IB
2. Check you're selling calls on positions you own
3. Ensure using realistic premiums (~1% quarterly)
4. Try different strike prices (5-10% OTM)

---

## Paper Trading Log Template

**Track your 3-month paper trading journey:**

```
TIER 2 PAPER TRADING LOG
========================

Start Date: ___________
End Date (target): ___________ (3 months)
Initial Capital: $100,000

WEEK 1 (DATE - DATE)
-------------------
Status: □ Connection ✅  □ Positions ✅  □ No errors ✅
Return: ____%
SPY return: ____%
Notes: _________________________________

WEEK 2 (DATE - DATE)
-------------------
Status: □ Connection ✅  □ Positions ✅  □ No errors ✅
Return: ____%
SPY return: ____%
Notes: _________________________________

[... continue for 12 weeks ...]

REBALANCE #1 (END OF QUARTER 1)
--------------------------------
Date: ___________
Sells executed: ___________
Buys executed: ___________
Transaction costs: ____% ($______)
Covered call premium: ____% ($______)
Issues: _________________________________

[... track all 3 rebalances ...]

FINAL REVIEW (AFTER 3 MONTHS)
------------------------------
Total return: ____%
Expected: 3-6% quarterly
Status: □ PASS  □ NEEDS MORE TIME

SPY return: ____%
Outperformance: ____%

Max drawdown: ____%
Expected: <-15%
Status: □ PASS  □ NEEDS ATTENTION

Technical issues: ___________
Status: □ NONE  □ MINOR  □ MAJOR (needs fixing)

Ready for live trading? □ YES - GRADUATE  □ NO - EXTEND PAPER TRADING
```

---

## Summary

**Paper trading is NOT optional.**

3 months minimum, but extend if:
- You encountered technical issues
- You're not comfortable with the strategy
- You want more data/confidence

**Paper trading gives you:**
- ✅ Technical validation (does it actually work?)
- ✅ Execution practice (can I do this correctly?)
- ✅ Psychological preparation (am I ready for losses?)
- ✅ Confidence building (do I trust the system?)

**After successful paper trading → Start live with 25% capital → Scale slowly over 9-12 months**

**Remember:** Discipline beats emotion. The system works if you follow it.

---

**Questions?** Review full deployment guide: `TIER2_PRODUCTION_DEPLOYMENT_GUIDE.md`
