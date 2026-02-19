# Data Requirements to Increase Profits - Tier 2 Strategy

## Current Performance Baseline
- **2023-2024**: 35.7% annual return
- **2015-2024**: 20.02% annual return
- **1990-2024**: 16% long-term average
- **Current approach**: 70% momentum + 30% fundamental growth scoring

## Current Data Limitations

### What We Have:
1. **Price data**: Daily OHLCV
2. **Basic fundamentals**: ROE, margins, FCF yield, ROIC, current ratio, debt-to-equity
3. **Market indicators**: VIX for volatility

### What We're Missing:
- Earnings quality signals
- Market sentiment indicators
- Forward-looking growth signals
- Sector/macro regime data
- Risk-adjusted momentum metrics

---

## High-Impact Data Additions (Prioritized)

### 🎯 **TIER 1: Highest ROI - Implement First**

#### 1. **Earnings Quality Metrics**
**Impact**: ⭐⭐⭐⭐⭐ (Very High)
**Difficulty**: ⚡⚡ (Medium)
**Cost**: $ (Free via FMP or AlphaVantage)

**What to add**:
- **Accruals ratio**: (Net Income - Operating Cash Flow) / Total Assets
  - High accruals = earnings manipulation risk
  - Low/negative accruals = high-quality earnings
- **Earnings persistence**: Correlation of quarterly EPS over 3 years
  - Consistent earnings = sustainable growth
- **Cash conversion rate**: Operating Cash Flow / Net Income
  - >1.0 = converting earnings to cash (good)
  - <0.8 = earnings not backed by cash (red flag)

**Why it helps**:
- Current strategy can pick momentum stocks with poor earnings quality
- These stocks crash when fundamentals catch up to price
- **Expected improvement**: +2-3% annual return, -2-3% drawdown

**Implementation**:
```python
def calculate_earnings_quality_score(fa_data):
    score = 0

    # Accruals (20 points) - lower is better
    accruals = (fa_data['netIncome'] - fa_data['operatingCashFlow']) / fa_data['totalAssets']
    if accruals < -0.05: score += 20
    elif accruals < 0: score += 15
    elif accruals < 0.05: score += 10

    # Cash conversion (30 points)
    cash_conversion = fa_data['operatingCashFlow'] / fa_data['netIncome']
    if cash_conversion > 1.2: score += 30
    elif cash_conversion > 1.0: score += 20
    elif cash_conversion > 0.8: score += 10

    return score
```

---

#### 2. **Forward Guidance & Analyst Revisions**
**Impact**: ⭐⭐⭐⭐⭐ (Very High)
**Difficulty**: ⚡⚡⚡ (Medium-High)
**Cost**: $$ (FMP Premium or AlphaVantage paid)

**What to add**:
- **Earnings estimate revisions** (last 30/60/90 days)
  - Number of upward vs downward revisions
  - Magnitude of changes
- **Earnings surprise history**
  - % beat/miss vs estimates (last 4 quarters)
  - Average surprise magnitude
- **Forward P/E vs historical P/E**
  - Forward P/E < Historical P/E = growth acceleration

**Why it helps**:
- Catches growth inflection points early (before price fully reflects)
- Analyst upgrades predict next 3-6 months returns
- **Expected improvement**: +3-4% annual return

**Example usage**:
```python
def get_analyst_momentum_score(ticker, date):
    revisions = get_analyst_revisions(ticker, date, lookback=60)

    score = 0
    # Positive revisions
    if revisions['num_upgrades'] > revisions['num_downgrades']:
        score += 30

    # Earnings surprises
    surprises = get_earnings_surprises(ticker, last_n=4)
    if all(s > 0 for s in surprises):  # All beats
        score += 40

    # Forward P/E valuation
    if forward_pe < historical_pe * 0.9:  # 10% cheaper on forward basis
        score += 30

    return score
```

---

#### 3. **Sector Rotation & Macro Regime Signals**
**Impact**: ⭐⭐⭐⭐⭐ (Very High)
**Difficulty**: ⚡⚡⚡ (Medium-High)
**Cost**: $ (Free via FRED, Yahoo Finance)

**What to add**:
- **Economic regime indicators**:
  - Yield curve slope (10Y - 2Y Treasury)
  - ISM Manufacturing PMI
  - Leading Economic Indicators (LEI)
  - Unemployment rate trend
- **Sector performance cycles**:
  - Relative strength of 11 GICS sectors vs SPY
  - Identify which sectors are in uptrends
- **Fed policy stance**:
  - Fed Funds rate trend (raising/cutting/neutral)
  - Real rates (nominal - inflation)

**Why it helps**:
- Different sectors outperform in different macro regimes
- Growth stocks (tech/consumer discretionary) thrive when rates falling
- Defensive sectors (utilities/staples) outperform in recessions
- **Expected improvement**: +2-4% annual return, better risk-adjusted returns

**Example logic**:
```python
def adjust_for_macro_regime(megacap_scores, momentum_scores, date):
    regime = detect_regime(date)

    if regime == 'expansion':
        # Favor cyclicals: tech, consumer discretionary, industrials
        boost_sectors = ['Technology', 'Consumer Discretionary']
        penalize_sectors = ['Utilities', 'Consumer Staples']

    elif regime == 'recession_risk':
        # Favor defensives
        boost_sectors = ['Healthcare', 'Utilities', 'Consumer Staples']
        penalize_sectors = ['Energy', 'Materials']

    elif regime == 'rising_rates':
        # Favor financials, penalize growth
        boost_sectors = ['Financials']
        penalize_sectors = ['Technology', 'Real Estate']

    # Apply sector adjustments (±10-20%)
    adjusted_scores = apply_sector_boost(megacap_scores, boost_sectors, multiplier=1.15)
    adjusted_scores = apply_sector_penalty(adjusted_scores, penalize_sectors, multiplier=0.85)

    return adjusted_scores
```

---

#### 4. **Short Interest & Insider Trading**
**Impact**: ⭐⭐⭐⭐ (High)
**Difficulty**: ⚡⚡ (Medium)
**Cost**: $$ (FMP Premium or other paid sources)

**What to add**:
- **Short interest metrics**:
  - Short % of float
  - Days to cover (short interest / avg daily volume)
  - Change in short interest (last 30 days)
- **Insider trading activity**:
  - Net insider buying/selling (last 90 days)
  - Number of insiders buying vs selling
  - Dollar value of insider transactions

**Why it helps**:
- **High short interest + positive momentum** = potential short squeeze catalyst
- **Insider buying** = management confidence in future growth
- **Insider selling** = warning sign (unless options exercise)
- **Expected improvement**: +1-2% annual return, avoid blow-ups

**Usage**:
```python
def check_short_squeeze_potential(ticker, date):
    short_data = get_short_interest(ticker, date)

    # High short interest + momentum = squeeze setup
    if short_data['short_pct_float'] > 15 and short_data['days_to_cover'] > 5:
        return 20  # Boost score (potential squeeze)

    return 0

def check_insider_confidence(ticker, date):
    insider_data = get_insider_trades(ticker, date, lookback=90)

    net_buying = insider_data['buy_value'] - insider_data['sell_value']

    if net_buying > 1_000_000 and insider_data['num_buyers'] > 3:
        return 30  # Strong insider buying
    elif net_buying > 0:
        return 15  # Positive insider sentiment

    return 0
```

---

### 🎯 **TIER 2: Medium-High Impact**

#### 5. **Options Market Signals**
**Impact**: ⭐⭐⭐⭐ (High)
**Difficulty**: ⚡⚡⚡⚡ (High)
**Cost**: $$$ (Premium data feeds)

**What to add**:
- **Implied volatility percentile**: Where is current IV vs 1-year range?
- **Put/call ratio**: Sentiment indicator (high P/C = bearish)
- **Options flow**: Unusual call buying (smart money positioning)
- **Volatility skew**: Put skew = downside protection demand

**Why it helps**:
- IV percentile identifies cheap vs expensive options (for covered calls)
- Unusual options activity predicts price moves
- **Expected improvement**: +1-3% from better covered call timing

---

#### 6. **Volume Profile & Liquidity Metrics**
**Impact**: ⭐⭐⭐ (Medium-High)
**Difficulty**: ⚡⚡⚡ (Medium-High)
**Cost**: $$ (CBOE data, Yahoo Finance Plus)

**What to add**:
- **Volume surge detection**: Volume > 2x average = institutional accumulation
- **Accumulation/Distribution indicator**: Price-volume divergence
- **Bid-ask spread**: Liquidity cost indicator
- **On-balance volume (OBV)**: Cumulative volume pressure

**Why it helps**:
- Volume confirms price moves (momentum with volume = sustainable)
- Low liquidity stocks have higher slippage costs
- **Expected improvement**: +1-2% annual return

---

#### 7. **News Sentiment & Earnings Call Tone**
**Impact**: ⭐⭐⭐⭐ (High)
**Difficulty**: ⚡⚡⚡⚡⚡ (Very High)
**Cost**: $$$$ (Expensive - RavenPack, Bloomberg)

**What to add**:
- **News sentiment score**: Aggregate news polarity (positive/negative)
- **Earnings call transcript sentiment**: Management tone analysis
- **Social media sentiment**: Twitter/Reddit mentions + sentiment
- **News momentum**: Change in sentiment over 30/60 days

**Why it helps**:
- Sentiment predicts next 1-3 month returns
- Catches narrative shifts before fundamentals show up
- **Expected improvement**: +2-4% annual return (but high cost)

---

### 🎯 **TIER 3: Nice-to-Have (Lower Priority)**

#### 8. **Alternative Data**
- Satellite imagery (retail foot traffic)
- Credit card transaction data
- Web traffic / app downloads
- Supply chain data

**Impact**: ⭐⭐⭐ (Medium)
**Cost**: $$$$$$ (Very expensive)
**Best for**: Individual stock deep dives, not systematic strategies

---

#### 9. **Intraday Price Data**
- Minute/hourly bars
- Opening auction dynamics
- Closing auction volume

**Impact**: ⭐⭐ (Low-Medium)
**Reason**: Strategy is quarterly rebalance, intraday less useful

---

## Recommended Implementation Plan

### **Phase 1 (Next 3 months)**: Core Quality Enhancements
**Target**: +4-6% annual return improvement

1. **Add earnings quality metrics** (accruals, cash conversion)
   - Data source: FMP API (free tier sufficient)
   - Modify `calculate_growth_potential_score()` to include
   - Weight: Add 15% earnings quality to growth score

2. **Add sector/macro regime detection**
   - Data source: FRED API (free), Yahoo Finance sectors
   - Implement `detect_regime()` function
   - Adjust sector weights by ±15% based on regime

3. **Add short interest data**
   - Data source: FMP Premium ($20/month)
   - Flag high-short-interest momentum stocks for boost
   - Avoid heavily shorted names in weak momentum

### **Phase 2 (3-6 months)**: Forward-Looking Signals
**Target**: +2-4% additional annual return

4. **Add analyst revisions tracking**
   - Data source: FMP Earnings Calendar + Estimates
   - Weight upgrades/beats heavily in growth score
   - Track earnings surprise history

5. **Add insider trading signals**
   - Data source: FMP Insider Trading
   - Boost stocks with net insider buying
   - Penalize heavy insider selling

### **Phase 3 (6-12 months)**: Advanced Optimization
**Target**: +1-3% additional annual return

6. **Add options market signals** (if covered calls enabled)
   - Optimize covered call strike selection with IV percentile
   - Detect unusual options flow for momentum confirmation

7. **Add volume/liquidity metrics**
   - Filter out low-liquidity stocks
   - Confirm momentum with volume surge

---

## Expected Total Improvement

**Conservative estimate**:
- Phase 1: +4-6% annual return
- Phase 2: +2-4% annual return
- Phase 3: +1-3% annual return
- **Total potential**: +7-13% annual return improvement

**Target performance**:
- **Current**: 16% long-term average
- **With Phase 1-2**: 22-26% annual return
- **With all phases**: 23-29% annual return

**Sharpe ratio improvement**:
- Current: ~1.89 (2023-2024)
- With better quality filters: ~2.2-2.5
- (Higher returns + lower drawdowns from avoiding blow-ups)

---

## Data Sources & Costs

| Data Type | Source | Cost | Priority |
|-----------|--------|------|----------|
| Earnings quality | FMP Free API | Free | HIGH |
| Sector data | Yahoo Finance | Free | HIGH |
| Macro indicators | FRED API | Free | HIGH |
| Short interest | FMP Premium | $20/mo | HIGH |
| Analyst estimates | AlphaVantage | $50/mo | MEDIUM |
| Insider trading | FMP Premium | $20/mo | MEDIUM |
| Options data | CBOE | $100+/mo | LOW |
| News sentiment | RavenPack | $1000+/mo | LOW |

**Recommended Phase 1 budget**: $0-20/month (free APIs + FMP Premium)

---

## Key Risks & Considerations

### 1. **Overfitting Risk**
- **Problem**: Too many features = model fits historical noise
- **Solution**: Keep scoring simple, only add proven factors
- **Rule**: Each new data source must improve out-of-sample by >1%

### 2. **Data Quality Issues**
- **Problem**: Historical fundamental data can have survivorship bias
- **Solution**: Use point-in-time data (what was known at that date)
- **Current approach**: Already using `get_fa_data_at_date()`

### 3. **Cost vs Benefit**
- **Problem**: Expensive data doesn't always justify cost
- **Solution**: Start with free/cheap sources, measure incremental impact
- **Rule**: Data cost should be <10% of expected return improvement

### 4. **Latency & Reporting Delays**
- **Problem**: Fundamentals reported 45-90 days after quarter end
- **Solution**: Use estimates/revisions for forward-looking view
- **Note**: Earnings quality metrics already have built-in delay

---

## Quick Win: Phase 1 Implementation

### Immediate Action Items:

1. **Sign up for FMP API** (Free or Premium $20/mo)
   - Get API key from: https://financialmodelingprep.com/

2. **Add 3 new metrics to growth scoring**:
   ```python
   # In calculate_growth_potential_score():

   # 6. Earnings Quality (15 points) - NEW
   accruals = (net_income - operating_cf) / total_assets
   if accruals < 0: score += 15
   elif accruals < 0.03: score += 10
   elif accruals < 0.05: score += 5

   # 7. Short Interest (10 points) - NEW
   if short_pct_float > 20 and momentum > 0: score += 10
   elif short_pct_float > 15 and momentum > 0: score += 5

   # 8. Sector Regime Fit (10 points) - NEW
   regime = detect_regime(date)
   if sector_matches_regime(ticker_sector, regime):
       score += 10
   ```

3. **Test on 2010-2024 period**
   - Compare Tier 2 (current) vs Tier 2.1 (with new metrics)
   - Target: >20% annual return, <-15% max drawdown

4. **If successful, deploy to paper trading**
   - Run alongside Tier 2 for 3 months
   - Compare live performance before switching

---

## Conclusion

**Highest ROI data additions**:
1. ✅ Earnings quality metrics (accruals, cash conversion)
2. ✅ Sector/macro regime detection
3. ✅ Short interest & insider trading
4. ✅ Analyst revisions & earnings surprises

**Expected outcome**:
- **Phase 1**: 16% → 20-22% annual return (+4-6%)
- **Phase 2**: 22% → 24-26% annual return (+2-4%)
- **Phase 3**: 26% → 27-29% annual return (+1-3%)

**Cost**: $0-20/month for Phase 1, $50-100/month for Phase 2

**Start with Phase 1** - highest impact, lowest cost, lowest risk.
