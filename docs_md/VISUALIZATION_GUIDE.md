# Monte Carlo Visualization Guide - Detailed Explanation

## Overview
The Monte Carlo visualization has 4 interactive dashboards (tabs) with multiple charts analyzing strategy robustness.

---

## TAB 1: DISTRIBUTION ANALYSIS (4 Charts)

### Chart 1: Annual Return Histogram (Top Left)

**What it shows:**
- Distribution of annual returns across all simulations
- Each bar = number of simulations that achieved that return range

**Key Elements:**
```
Y-axis: Frequency (how many simulations)
X-axis: Annual Return (%)

Vertical Lines:
- Red dashed lines   = 5th and 95th percentile (worst/best 5%)
- Orange dashed lines = 25th and 75th percentile (middle 50% boundaries)
- Green dashed line  = Median (50th percentile - middle value)

Color Zones (background):
- Red zone (<5%):     Poor performance
- Yellow zone (5-8%): Below target
- Green zone (>8%):   Good performance
```

**How to interpret:**
- **Peak of histogram** = Most common return level
- **Width of distribution** = Return consistency
  - Narrow = Consistent returns (good\!)
  - Wide = Variable returns (less predictable)
- **95% Confidence Interval** = Range between red lines
  - "95% of simulations fall within this range"
- **Labels on lines** = Exact percentile values

**Example Reading:**
```
If histogram shows:
- 5th percentile: 7.2%
- Median: 9.3%
- 95th percentile: 11.8%

Interpretation:
"95% of parameter combinations produce returns between 7.2% and 11.8%,
with typical (median) performance around 9.3%"
```

---

### Chart 2: Max Drawdown Histogram (Top Right)

**What it shows:**
- Distribution of worst drawdowns across simulations
- Each bar = number of simulations with that drawdown level

**Key Elements:**
```
Y-axis: Frequency
X-axis: Max Drawdown (%) - NEGATIVE numbers (losses)

Vertical Lines:
- Dark red   = 5th percentile (WORST drawdowns)
- Orange     = Median drawdown
- Green      = 95th percentile (BEST drawdowns, least negative)

Color Zones:
- Red zone (<-20%):      Severe drawdowns
- Yellow zone (-20 to -15%): Moderate drawdowns
- Green zone (>-15%):    Manageable drawdowns
```

**How to interpret:**
- **Leftmost bars (more negative)** = Worse drawdowns
- **Rightmost bars (less negative)** = Better drawdowns
- **Width** = Drawdown consistency
  - Narrow = Predictable risk
  - Wide = Variable risk across parameters

**Example Reading:**
```
If histogram shows:
- 5th percentile (worst): -22.1%
- Median: -18.1%
- 95th percentile (best): -14.8%

Interpretation:
"In worst-case scenarios (5%), drawdown reaches -22.1%.
Typical drawdown is -18.1%. Best case is -14.8%.
Strategy is fairly consistent on downside risk."
```

---

### Chart 3: Sharpe Ratio Histogram (Bottom Left)

**What it shows:**
- Distribution of risk-adjusted returns (Sharpe Ratio)
- Higher Sharpe = Better return per unit of risk

**Key Elements:**
```
Y-axis: Frequency
X-axis: Sharpe Ratio (unitless)

Vertical Lines:
- Red    = 5th percentile (worst risk-adjusted)
- Blue   = Median
- Green  = 95th percentile (best risk-adjusted)

Color Zones:
- Red zone (<0.8):    Poor risk-adjusted returns
- Yellow (0.8-1.0):   Acceptable
- Green zone (>1.0):  Good risk-adjusted returns
```

**How to interpret:**
- **Sharpe > 1.0** = Excellent (institutional quality)
- **Sharpe 0.5-1.0** = Good
- **Sharpe < 0.5** = Poor (too much risk for the return)

**Example Reading:**
```
If histogram shows:
- 5th percentile: 0.87
- Median: 1.01
- 95th percentile: 1.14

Interpretation:
"Even in worst cases, Sharpe stays above 0.87 (acceptable).
Typical Sharpe is 1.01 (institutional quality).
Strategy has robust risk-adjusted returns."
```

---

### Chart 4: Risk-Return Scatter Plot (Bottom Right)

**What it shows:**
- Each dot = 1 simulation
- Position shows risk vs return trade-off
- Color indicates Sharpe ratio

**Key Elements:**
```
X-axis: Max Drawdown (%) - more negative = riskier
Y-axis: Annual Return (%) - higher = better
Color: Sharpe Ratio
  - Purple/Dark = Low Sharpe (poor risk-adjusted)
  - Yellow/Bright = High Sharpe (good risk-adjusted)

Size: All dots same size (can be modified to show final portfolio value)
```

**How to interpret:**

**Ideal Region (Top-Right quadrant):**
```
High Return (top) + Low Drawdown (right side, less negative) = BEST
```

**Poor Region (Bottom-Left):**
```
Low Return (bottom) + High Drawdown (left side, very negative) = WORST
```

**Color Meaning:**
- **Bright yellow dots** = High Sharpe (efficient - good return for the risk)
- **Dark purple dots** = Low Sharpe (inefficient - poor return for the risk)

**Hover Feature:**
Move mouse over any dot to see:
- Simulation number
- Exact return value
- Exact drawdown value
- Sharpe ratio
- Final portfolio value

**Example Reading:**
```
Cluster of bright yellow dots at:
- X = -17% (drawdown)
- Y = 9.5% (return)

Interpretation:
"Most simulations achieve 9.5% return with -17% drawdown.
These are colored bright yellow (high Sharpe), meaning
this risk-return profile is efficient."

Outlier dark purple dot at:
- X = -25% (drawdown)
- Y = 7% (return)

Interpretation:
"This parameter combination had worse drawdown AND lower return.
Dark color confirms poor Sharpe ratio - avoid this configuration."
```

**Pattern Analysis:**
- **Tight cluster** = Consistent performance across parameters (robust)
- **Scattered dots** = Highly parameter-sensitive (needs careful tuning)
- **Upward trend** = More risk = More return (expected)
- **Horizontal spread** = Same returns but different risks (some configs are better)

---

## TAB 2: PARAMETER SENSITIVITY

### Chart 1: Box Plots by Parameter

**What it shows:**
- How annual return varies with different parameter values
- Each "box" = distribution of returns for that parameter setting

**Key Elements:**
```
Each Box shows:
  Top of box    = 75th percentile
  Red line      = Median (50th percentile)
  Bottom of box = 25th percentile
  
Width of box = Spread of middle 50% of results
```

**X-axis Categories:**
```
Portfolio Size variations:
- "3" = Fixed 3-stock portfolio
- "5" = Fixed 5-stock portfolio  
- "7" = Fixed 7-stock portfolio
- "10" = Fixed 10-stock portfolio
- "Dynamic" = Regime-based sizing (3-10 stocks)
```

**How to interpret:**

**Taller boxes** = More variable returns with this parameter
**Shorter boxes** = More consistent returns

**Higher median line** = Better average performance

**Example Reading:**
```
Box for "3 stocks":
- Bottom: 8.5%
- Median (red line): 10.2%
- Top: 11.5%

Box for "10 stocks":
- Bottom: 7.5%
- Median: 8.1%
- Top: 8.8%

Interpretation:
"3-stock portfolio has higher median return (10.2% vs 8.1%)
but also more variability (box is taller).
10-stock portfolio is more consistent but lower return.
Trade-off between concentration and diversification."
```

---

### Chart 2: Sensitivity Summary Table

**What it shows:**
- Text summary of parameter impacts
- Currently placeholder (will show detailed sensitivity analysis in Phase 2)

**Will include:**
- Ranking of parameters by impact
- Optimal values for each parameter
- Sensitivity coefficients

---

## TAB 3: CONFIDENCE BANDS (Time Series)

**What it shows:**
- Portfolio value over time with uncertainty bands
- Shows how portfolio grows with confidence intervals

**Currently:**
- Placeholder text (requires full time series data from simulations)

**When fully implemented:**
```
Median line (thick blue):
  - Most likely portfolio value path over time

90% Confidence Band (light blue shade):
  - Range between 5th and 95th percentile
  - "90% of simulations fall within this band"
  
50% Confidence Band (darker blue shade):
  - Range between 25th and 75th percentile
  - "Middle 50% of simulations"

Min/Max envelope (thin lines):
  - Absolute worst and best cases
```

**How to interpret (once implemented):**
```
If band is:
- NARROW = Consistent performance regardless of parameters
- WIDE = High sensitivity to parameters/conditions
- WIDENING over time = Increasing uncertainty
- STABLE width = Consistent uncertainty level
```

---

## TAB 4: BOOTSTRAP ANALYSIS

**What it shows:**
- How performance varies with different start dates
- Tests whether timing your entry matters

**Key Elements:**
```
X-axis: Start Date Offset (days)
  - 0 = Start at earliest date in dataset
  - 60 = Start 60 days later
  - 120 = Start 120 days later
  - etc.

Y-axis: Annual Return (%)

Each dot = One simulation with that start offset

Hover shows:
- Exact offset
- Annual return
- Max drawdown
- Sharpe ratio
```

**How to interpret:**

**Flat pattern (dots at same Y-level):**
```
"Strategy is ROBUST to entry timing.
Doesn't matter when you start investing."
```

**Upward trend:**
```
"Later start dates perform better.
Early period was unfavorable."
```

**Downward trend:**
```
"Earlier start dates perform better.
Later period was unfavorable."
```

**Scattered dots:**
```
"Performance highly dependent on entry timing.
Market timing matters for this strategy."
```

**Example Reading:**
```
Dots at:
- 0 days: 9.1% return
- 60 days: 9.3% return
- 120 days: 9.2% return
- 180 days: 9.1% return

Interpretation:
"Returns cluster around 9.1-9.3% regardless of start date.
Very small variation (0.2%) means strategy is NOT sensitive
to entry timing. You can invest at any time."
```

---

## INTERACTIVE FEATURES (All Charts)

### Toolbar (Top right of each chart)

**Tools available:**

1. **Pan** (hand icon)
   - Click and drag to move around chart
   - Useful for examining specific regions

2. **Box Zoom** (square with arrows)
   - Drag to select area to zoom into
   - Click to activate, draw box, release

3. **Wheel Zoom** (magnifying glass)
   - Scroll wheel to zoom in/out
   - Centers on mouse position

4. **Reset** (circular arrow)
   - Returns chart to original view
   - Undoes all pan/zoom

5. **Save** (floppy disk icon)
   - Saves current view as PNG image
   - Good for reports/presentations

6. **Hover** (crosshair)
   - Auto-enabled
   - Shows tooltips when mouse over data

### Hover Tooltips

**Histograms:**
- Shows bar height (frequency count)
- Shows range of values in that bar

**Scatter plots:**
- Shows all metrics for that simulation
- Simulation ID
- Exact values for X and Y axes
- Additional metrics (Sharpe, final value, etc.)

---

## COMMON QUESTIONS

### Q1: Why are all my values the same in quick test?
**A:** The 5-simulation quick test uses identical parameters (baseline V28).
Run Phase 1 with parameter variations to see distribution.

### Q2: What's a "good" distribution shape?
**A:** 
- **Normal (bell curve)** = Most parameter combos work similarly (robust)
- **Bimodal (two peaks)** = Two distinct behaviors (parameter-dependent)
- **Skewed** = Asymmetric risk/reward profile

### Q3: How do I find optimal parameters?
**A:**
1. Look at scatter plot - find bright yellow dots (high Sharpe)
2. Hover over those dots to see their parameter values
3. Look at box plots - find tallest median lines
4. Check sensitivity table for parameter rankings

### Q4: What if my 95% CI is very wide?
**A:**
Wide confidence interval means:
- Strategy is parameter-sensitive
- Need careful parameter selection
- Consider more robust parameter values (those with narrow boxes)

### Q5: Should I aim for highest return or highest Sharpe?
**A:**
- **Highest Sharpe** = Best risk-adjusted (recommended)
- **Highest Return** = May come with unacceptable risk
- Look for bright yellow dots (high Sharpe) in upper-right area

---

## NEXT STEPS

After viewing visualizations:

1. **Identify patterns** - Are results clustered or scattered?
2. **Note optimal regions** - Which dots are bright yellow and top-right?
3. **Check robustness** - Is 95% CI acceptably narrow?
4. **Run Phase 1** - Get real data with 77 parameter variations
5. **Refine strategy** - Use insights to optimize V28 parameters

---

