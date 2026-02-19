# Live Trading Monitoring Checklist - Tier 2 Strategy

## Quick Reference Card

**Strategy:** V31 Tier 2 Growth Scoring
**Expected Annual Return:** 15-20% (bull markets), 7-10% (crisis), ~16% (long-term avg)
**Expected Drawdown:** -15% to -20%
**Rebalancing:** Quarterly (Jan/Apr/Jul/Oct)
**Risk Management:** 15% trailing stops, VIX-based cash (5-70%)

---

## DAILY CHECKS (5 minutes) ✅

**Run every morning before 9:30 AM ET**

### Connection Status
- [ ] Broker API connected
- [ ] No error messages
- [ ] Account accessible

### Portfolio Health
- [ ] Current value: $________
- [ ] Change from yesterday: _____%
- [ ] Any positions stopped out: □ No  □ Yes (list: ________)
- [ ] Any unusual alerts: □ No  □ Yes (describe: ________)

### Market Conditions
- [ ] VIX level: _____ (Low <15, Normal 15-25, High 25-35, Extreme >35)
- [ ] SPY change: _____%
- [ ] Current cash reserve: _____% (should adjust with VIX)

### Quick Status
```
✅ All systems normal → Done for today
⚠️ Warning signs → Review weekly checklist early
🚨 Critical issues → Stop and investigate immediately
```

---

## WEEKLY CHECKS (15 minutes) ✅

**Run every Monday morning**

### Performance Review

**Week of: __________**

| Metric | Value | Status |
|--------|-------|--------|
| Weekly return | _____% | □ Positive □ Negative |
| SPY return | _____% | □ Positive □ Negative |
| Relative performance | _____% | □ Outperform □ Underperform |
| Running drawdown | _____% | □ <-10% □ -10 to -15% □ >-15% ⚠️ |

### Position Review
- [ ] Top 3 mega-cap holdings: _____________, _____________, _____________
- [ ] Any underperforming >10%: □ No  □ Yes (list: ________)
- [ ] Any stopped out this week: □ No  □ Yes (count: ___, tickers: ________)
- [ ] Positions still match Tier 2 logic: □ Yes  □ No (investigate why)

### Technical Health
- [ ] API disconnections this week: _____ (0 = good, 1-2 = monitor, 3+ = fix)
- [ ] Error messages: □ None  □ Minor  □ Major (describe: ________)
- [ ] Data feeds working: □ Yes  □ No

### Red Flags Check
- [ ] Drawdown >-20%: □ No ✅  □ Yes 🚨 (CRITICAL)
- [ ] 3 consecutive down weeks: □ No  □ Yes (normal, but watch)
- [ ] Technical failures: □ No  □ Yes (needs attention)
- [ ] Unusual trading activity: □ No  □ Yes (investigate)

**Weekly Summary:**
```
Week __ of 20__:
Portfolio: $________ (___% from start)
Best performer: ________ (+___%)
Worst performer: ________ (___%)
Notes: _________________________________
```

---

## MONTHLY CHECKS (30 minutes) ✅

**Run last Friday of each month**

### Month: __________ 20__

### Performance Analysis

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Monthly return | _____% | +1-2% | □ On track □ Off track |
| YTD return | _____% | +12-20% annual | □ On track □ Off track |
| SPY monthly return | _____% | Benchmark | - |
| Outperformance | _____% | Positive | □ Yes □ No |
| Max drawdown (month) | _____% | <-10% | □ Good □ Watch |
| Max drawdown (YTD) | _____% | <-15% | □ Good □ Watch |

### Cost Analysis
- [ ] Transaction costs (month): $_______ (_____% of portfolio)
- [ ] Expected: <1% monthly | Status: □ Good □ High ⚠️
- [ ] Covered call premiums: $_______ (_____% of portfolio)
- [ ] Expected: ~0.33% monthly (~1% quarterly) | Status: □ Good □ Low ⚠️
- [ ] Total fees/expenses: $_______
- [ ] Unexpected costs: □ None □ Yes (describe: ________)

### Position Analysis
- [ ] Average position size: _____% (expect: 7-10% for top holdings)
- [ ] Number of positions: _____ (expect: 8-10)
- [ ] Mega-cap allocation: _____% (target: 70%)
- [ ] Momentum allocation: _____% (target: 30%)
- [ ] Cash allocation: _____% (varies with VIX: 5-70%)

### Holdings Review
```
Top Holdings (should be ~70% of portfolio):
1. __________ (___%)
2. __________ (___%)
3. __________ (___%)

Momentum Holdings (should be ~30% of portfolio):
4. __________ (___%)
5. __________ (___%)
6. __________ (___%)
7. __________ (___%)
8. __________ (___%)
9. __________ (___%)
10. __________ (___%)

Cash: _____% (VIX-based reserve)
```

### Risk Assessment
- [ ] Trailing stops working: □ Yes  □ No (how many triggered: ___)
- [ ] VIX-based cash adjusting: □ Yes  □ No (current VIX: ___)
- [ ] Diversification maintained: □ Yes  □ No
- [ ] No single position >15%: □ Yes  □ No ⚠️

### Strategy Health
- [ ] Growth scoring still valid: □ Yes  □ No (investigate fundamentals)
- [ ] Momentum signals working: □ Yes  □ No (check market regime)
- [ ] Quarterly rebalancing on schedule: □ Yes  □ No
- [ ] Following system rules: □ Yes  □ No (no emotional overrides!)

**Monthly Summary:**
```
Month __ Summary:
Return: ____%
YTD: ____%
Drawdown: ____%
Key wins: _________________________________
Key lessons: _________________________________
Next month focus: _________________________________
```

---

## QUARTERLY ACTIONS (2-3 hours) ✅

**Run on first trading day of each quarter: Jan 1, Apr 1, Jul 1, Oct 1**

### Pre-Rebalance (Day Before)

**Quarter: Q__ 20__**

#### 1. Run Fresh Backtest
```bash
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024
```

#### 2. Review Recommended Changes
- [ ] New positions to buy: _________________________________
- [ ] Current positions to sell: _________________________________
- [ ] Positions to hold: _________________________________

#### 3. Calculate Target Allocations
```
Target allocation:
- 70% to top 3 mega-caps: $_______ each
- 30% to top 7 momentum: $_______ each
- Cash reserve (VIX-based): $_______
```

#### 4. Pre-Rebalance Checklist
- [ ] Sufficient cash for purchases: $_______ available
- [ ] All sells will settle T+2: □ Confirmed
- [ ] Broker API tested and working: □ Yes
- [ ] Market conditions normal: □ Yes  □ Volatile (adjust if needed)
- [ ] No earnings announcements today: □ Confirmed

### Rebalance Day Execution

**Date: __________**

#### Phase 1: Sell Orders (9:30 AM)
- [ ] Execute all sell orders
- [ ] Verify fills: _________ (list tickers and prices)
- [ ] Cash available: $_______ (after sells)

#### Phase 2: Buy Orders (9:40 AM)
- [ ] Execute all buy orders at calculated allocations
- [ ] Verify fills: _________ (list tickers and prices)
- [ ] Final cash balance: $_______

#### Phase 3: Covered Calls (9:50 AM)
- [ ] Sell covered calls on all positions
- [ ] Target: 1% premium (90-day, 5% OTM)
- [ ] Total premium collected: $_______ (_____% of portfolio)

#### Phase 4: Final Verification (10:00 AM)
- [ ] All positions match target: □ Yes  □ Minor differences (ok) □ Major differences (fix)
- [ ] Allocations correct: □ 70% mega-cap □ 30% momentum □ Cash
- [ ] No pending orders: □ Confirmed
- [ ] All trades settled: □ Yes

### Post-Rebalance Analysis

#### Transaction Cost Review
- [ ] Total transaction costs: $_______ (_____% of portfolio)
- [ ] Target: <2% of portfolio | Status: □ Good □ High ⚠️
- [ ] Breakdown:
  - Spreads: $_______
  - Commissions: $_______
  - Slippage: $_______

#### Covered Calls Review
- [ ] Premiums collected: $_______ (_____% of portfolio)
- [ ] Target: ~1% quarterly | Status: □ Good □ Low ⚠️
- [ ] Annualized: _____% (target: 4% annual)

#### Position Review
```
New positions:
1. __________ @ $_____ (___% allocation)
2. __________ @ $_____ (___% allocation)
[... list all 10 positions ...]

Exited positions:
1. __________ @ $_____ (___% gain/loss)
2. __________ @ $_____ (___% gain/loss)
[... list all exits ...]
```

### Comprehensive Quarterly Review

#### Performance Scorecard

**Quarter: Q__ 20__**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Quarterly return | _____% | 3-6% | □ On track □ Off track |
| YTD return | _____% | 9-18% (annualized) | □ On track □ Off track |
| SPY quarterly | _____% | Benchmark | - |
| Outperformance | _____% | Positive | □ Yes □ No |
| Max quarterly DD | _____% | <-10% | □ Good □ High ⚠️ |
| Sharpe ratio (YTD) | _____ | >0.9 | □ Good □ Low ⚠️ |

#### Trade Analysis (Last Quarter)

- [ ] Total trades: _____ (sells + buys)
- [ ] Winning trades: _____ (_____%)
- [ ] Losing trades: _____ (_____%)
- [ ] Average gain: _____%
- [ ] Average loss: _____%
- [ ] Largest gain: __________ (+____%)
- [ ] Largest loss: __________ (-____%)

#### Best/Worst Performers

**Winners (Top 3):**
1. __________ (+____%)
2. __________ (+____%)
3. __________ (+____%)

**Losers (Bottom 3):**
1. __________ (-____%)
2. __________ (-____%)
3. __________ (-____%)

**Stopped out:**
- Count: _____
- Tickers: _________________________________
- Why: _________________________________

#### Strategy Health Check

**Momentum component (70%):**
- [ ] Working as expected: □ Yes  □ No
- [ ] Capturing trends: □ Yes  □ No
- [ ] Issues: _________________________________

**Growth component (30%):**
- [ ] Adding value: □ Yes  □ No
- [ ] Fundamentals still valid: □ Yes  □ No
- [ ] Issues: _________________________________

**Risk management:**
- [ ] Trailing stops effective: □ Yes  □ No
- [ ] VIX-based cash working: □ Yes  □ No
- [ ] Drawdown control good: □ Yes  □ No
- [ ] Issues: _________________________________

**Covered calls:**
- [ ] Collecting 1% quarterly: □ Yes  □ No
- [ ] Annualizing to 4%: □ Yes  □ No
- [ ] Any called away: □ No  □ Yes (count: ___)
- [ ] Issues: _________________________________

#### Market Regime Analysis

**Current market regime:**
- [ ] Bull (VIX <20, SPY trending up): □ Yes
- [ ] Normal (VIX 20-25, SPY mixed): □ Yes
- [ ] Volatile (VIX 25-35, SPY choppy): □ Yes
- [ ] Crisis (VIX >35, SPY declining): □ Yes

**Strategy performing as expected for regime:**
- Bull: Expect 4-6% quarterly (18-23% annual) → Actual: _____%
- Normal: Expect 3-5% quarterly (15-18% annual) → Actual: _____%
- Volatile: Expect 2-4% quarterly (10-15% annual) → Actual: _____%
- Crisis: Expect 1-3% quarterly (7-10% annual) → Actual: _____%

**Status:** □ Performing as expected □ Underperforming (investigate)

**Quarterly Summary:**
```
Q__ 20__ Summary:
====================
Return: ____%
Trades: _____ (___% win rate)
Costs: ____%
Covered call income: ____%
Best stock: __________ (+___%)
Worst stock: __________ (-___%)

Key learnings:
_________________________________
_________________________________

Next quarter focus:
_________________________________
_________________________________
```

---

## CRITICAL ALERTS 🚨

**Stop and address immediately if any of these occur:**

### Portfolio Alerts
- [ ] Drawdown exceeds -25% from peak 🚨
  - **Action:** Pause trading, review what went wrong, consider reducing position sizes
- [ ] 3 consecutive losing quarters 🚨
  - **Action:** Review strategy validity, check if market regime changed fundamentally
- [ ] Account value drops below $75,000 (25% loss) 🚨
  - **Action:** Stop new trades, evaluate if continuing is appropriate

### Technical Alerts
- [ ] Broker API repeatedly failing 🚨
  - **Action:** Contact broker support, switch to manual execution if needed
- [ ] Cannot execute rebalancing 🚨
  - **Action:** Manual execution required, document issues for fixing
- [ ] Data feeds incorrect/stale 🚨
  - **Action:** Stop trading, verify data source, do not trade on bad data

### Strategy Alerts
- [ ] Positions wildly different from backtest 🚨
  - **Action:** Stop, investigate why (data issue? code bug? market structure change?)
- [ ] Transaction costs >5% quarterly 🚨
  - **Action:** Review execution, consider less frequent rebalancing
- [ ] Covered calls stopped working 🚨
  - **Action:** Review options strategy, check market conditions for options

### Behavioral Alerts ⚠️
- [ ] Tempted to override strategy picks
  - **Warning:** Emotional decision-making destroys systematic edge. Trust the process.
- [ ] Second-guessing quarterly rebalancing
  - **Warning:** System is designed to rebalance quarterly. Don't deviate.
- [ ] Checking portfolio multiple times per day
  - **Warning:** Overmonitoring leads to emotional decisions. Stick to schedule.

---

## ANNUAL REVIEW (4-5 hours) ✅

**Run in January for previous full year**

### Year: 20__

### Full Year Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Annual return | _____% | 15-20% | □ Met □ Missed |
| SPY annual return | _____% | Benchmark | - |
| Outperformance | _____% | +2-5% | □ Met □ Missed |
| Max annual DD | _____% | <-20% | □ Good □ High |
| Sharpe ratio | _____ | >1.0 | □ Good □ Low |
| Win quarters | __/4 | 3/4 | □ Good □ Low |

### Full Year Statistics

- [ ] Total trades: _____
- [ ] Win rate: _____%
- [ ] Average quarterly return: _____%
- [ ] Best quarter: Q__ (____%)
- [ ] Worst quarter: Q__ (____%)
- [ ] Total costs: $_______ (____% of avg portfolio)
- [ ] Covered call income: $_______ (____% of avg portfolio)

### Strategy Validation

**Compare to backtest expectations:**
- Backtest annual (2015-2024 avg): 19-20%
- Actual annual: _____%
- Difference: _____%
- Status: □ Within range □ Needs investigation

**Multi-period comparison:**
- Bull market expected: 18-23%
- Normal expected: 15-18%
- Crisis expected: 7-10%
- Actual: _____%
- Market regime was: _____________
- Status: □ Performed as expected □ Underperformed □ Outperformed

### Lessons Learned

**What worked well:**
1. _________________________________
2. _________________________________
3. _________________________________

**What didn't work:**
1. _________________________________
2. _________________________________
3. _________________________________

**Adjustments for next year:**
1. _________________________________
2. _________________________________
3. _________________________________

**Continue/Stop/Start:**
- Continue: _________________________________
- Stop: _________________________________
- Start: _________________________________

### Multi-Year Tracking

**Track cumulative performance over years:**

| Year | Return | Cumulative | SPY | vs SPY | DD | Notes |
|------|--------|------------|-----|--------|----|----|
| 2024 | ___% | ___% | ___% | ___% | ___% | _________ |
| 2025 | ___% | ___% | ___% | ___% | ___% | _________ |
| 2026 | ___% | ___% | ___% | ___% | ___% | _________ |
| 2027 | ___% | ___% | ___% | ___% | ___% | _________ |

**3-Year Target:** CAGR of 15-17%
**5-Year Target:** CAGR of 15-17%, Sharpe >1.0

---

## DECISION FRAMEWORKS

### When to Keep Going ✅
- Returns within expected range for market regime
- Technical issues minor and resolved
- Drawdowns <-20%
- Following strategy rules consistently
- Comfortable with process and results

### When to Pause ⚠️
- Drawdowns approaching -25%
- Significant technical issues preventing proper execution
- 3+ consecutive losing quarters
- Behavioral issues (overriding strategy, emotional decisions)
- Need to reassess risk tolerance

### When to Stop 🚨
- Drawdowns exceeding -30%
- Fundamental market structure change invalidating strategy assumptions
- Cannot execute strategy due to technical limitations
- Lost confidence in systematic approach
- Personal circumstances changed (need liquidity, reduced risk tolerance)

---

## QUICK TROUBLESHOOTING

### "Returns not matching expectations"
1. Check market regime (bull/normal/crisis) - adjust expectations
2. Verify transaction costs enabled in backtest
3. Confirm covered calls executing
4. Compare to SPY - is market underperforming too?

### "Too many positions stopped out"
1. Check VIX level - high volatility = more stops
2. Verify stops are 15% (not tighter)
3. Review if market is in correction (normal to have stops)
4. Ensure not overriding entries (buying weak stocks)

### "Outperformance turned negative"
1. Check how long (1 quarter = normal, 3+ quarters = investigate)
2. Verify positions matching Tier 2 recommendations
3. Review if following rebalancing schedule
4. Compare growth scoring - are fundamentals still strong?

### "Costs higher than expected"
1. Review transaction frequency (should be quarterly)
2. Check execution quality (using limit orders?)
3. Verify broker fees reasonable
4. Consider if monthly rebalancing accidentally enabled

---

## PRINTABLE WEEKLY CHECKLIST

**Copy this section to track weekly:**

```
WEEK OF: __________, 20__

DAILY CHECKS (Mon-Fri):
□ Mon: Connected ✅ / Portfolio: $_______ / Change: ____%
□ Tue: Connected ✅ / Portfolio: $_______ / Change: ____%
□ Wed: Connected ✅ / Portfolio: $_______ / Change: ____%
□ Thu: Connected ✅ / Portfolio: $_______ / Change: ____%
□ Fri: Connected ✅ / Portfolio: $_______ / Change: ____%

WEEKLY SUMMARY:
Return: ____%
SPY: ____%
Relative: ____%
Drawdown: ____%

NOTES: _________________________________
_______________________________________
```

---

## SUMMARY

**Monitoring frequency:**
- ✅ Daily (5 min): Check connection, portfolio value, any stops
- ✅ Weekly (15 min): Review performance, check positions, scan for issues
- ✅ Monthly (30 min): Analyze costs, review holdings, assess health
- ✅ Quarterly (2-3 hrs): Execute rebalancing, comprehensive review
- ✅ Annual (4-5 hrs): Full year analysis, strategy validation, lessons learned

**Most important rule:** FOLLOW THE SYSTEM. Don't override picks. Don't skip rebalancing. Trust the 34-year validation.

**Remember:**
- Losing quarters happen (30-40% of time) - NORMAL
- Drawdowns happen (expect -15 to -20%) - NORMAL
- Underperforming SPY happens (30-40% of quarters) - NORMAL
- 3+ year horizon required for strategy to prove out - BE PATIENT

**For detailed guidance, see:**
- `TIER2_PRODUCTION_DEPLOYMENT_GUIDE.md` - Full deployment guide
- `PAPER_TRADING_QUICK_START.md` - Paper trading setup
- `TIER2_MULTI_PERIOD_RESULTS.md` - Strategy validation

---

**Last Updated:** 2026-02-18
**Strategy:** V31 Tier 2 Growth Scoring
**Status:** PRODUCTION READY ✅
