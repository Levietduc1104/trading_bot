# Claude Skills for Trading Bot Repository

This directory contains custom Claude skills to help automate common trading bot workflows.

## 📚 Available Skills

### 1. **backtest** - Run Full Backtest
Executes the V22 production strategy and generates comprehensive reports.

**Usage:**
```
Hey Claude, use the backtest skill
```

**What it does:**
- Runs V22-Sqrt Kelly Position Sizing strategy
- Generates performance metrics
- Creates visualizations
- Saves to database

---

### 2. **strategy-compare** - Compare Strategies
Compare multiple strategy versions side-by-side.

**Usage:**
```
Claude, use strategy-compare to compare V13, V20, and V22
```

**What it does:**
- Runs multiple strategies
- Compares performance metrics
- Identifies winner
- Shows deltas vs baseline

---

### 3. **optimize** - Parameter Optimization
Run parameter sweeps to find optimal settings.

**Usage:**
```
Claude, use optimize skill to test portfolio sizes 3, 5, 7, 10
```

**What it does:**
- Tests parameter variations
- Compares results
- Identifies optimal value
- Provides sensitivity analysis

**Common optimizations:**
- Portfolio size (3, 5, 7, 10 stocks)
- Kelly exponent (0.3 to 0.7)
- VIX thresholds
- Drawdown control levels

---

### 4. **quick-test** - Fast Validation
Quick test on recent data (2-3 years) for rapid iteration.

**Usage:**
```
Claude, quick-test my recent changes
```

**What it does:**
- Detects code changes
- Runs abbreviated backtest
- Compares to baseline
- Provides fast feedback (<30s)

**Use when:**
- Testing new indicators
- Validating bug fixes
- Exploring parameter tweaks
- Quick sanity checks

---

### 5. **performance-report** - Detailed Analysis
Generate comprehensive performance analysis from database.

**Usage:**
```
Claude, generate a performance-report
```

**What it does:**
- Queries database for latest results
- Calculates advanced metrics (Sortino, Calmar)
- Analyzes yearly performance
- Identifies strengths/weaknesses
- Provides recommendations

---

## 🚀 How to Use Skills

### Method 1: Direct Invocation
```
Hey Claude, use the [skill-name] skill
```

### Method 2: Natural Language
```
Claude, can you run a backtest and show me the results?
```
(Claude will automatically use the backtest skill)

### Method 3: With Parameters
```
Claude, use optimize to test Kelly exponents from 0.3 to 0.7
```

---

## 📝 Creating Custom Skills

Skills are markdown files in `.claude/skills/` that describe:

1. **What the skill does** - Clear description
2. **Steps to execute** - Detailed workflow
3. **Expected output** - Format and content
4. **Success criteria** - How to validate results

### Skill Template:
```markdown
# [Skill Name] Skill

[Brief description]

## What this skill does:
1. Step 1
2. Step 2
3. Step 3

## Steps to execute:
1. Detailed step 1
2. Detailed step 2

## Expected output format:
[Show example output]

## Success criteria:
- Criterion 1
- Criterion 2
```

---

## 💡 Skill Best Practices

### ✅ DO:
- Keep skills focused on one task
- Provide clear success criteria
- Include example outputs
- Make skills composable (chain them)

### ❌ DON'T:
- Make skills too broad
- Assume file locations
- Skip error handling
- Forget to validate results

---

## 🎯 Common Workflows

### Workflow 1: Test New Strategy
```bash
1. Modify code
2. Claude, quick-test
3. (If promising) Claude, run backtest
4. Claude, performance-report
```

### Workflow 2: Find Optimal Parameters
```bash
1. Claude, optimize portfolio-size
2. (Note optimal value)
3. Update code with optimal parameter
4. Claude, backtest to confirm
```

### Workflow 3: Compare Approaches
```bash
1. Claude, strategy-compare V20 vs V22
2. (Analyze results)
3. Claude, performance-report for winner
```

---

## 🔧 Advanced: Skill Chaining

Skills can be chained together:

```
Claude, run quick-test, and if results look good,
run full backtest and generate performance-report
```

Claude will:
1. Execute quick-test skill
2. Evaluate results
3. Conditionally run backtest skill
4. Generate performance-report skill

---

## 📊 Metrics Calculated

Skills calculate these standard metrics:

**Return Metrics:**
- Annual Return (CAGR)
- Total Return
- Yearly Returns

**Risk Metrics:**
- Max Drawdown
- Volatility (annual)
- Downside Deviation

**Risk-Adjusted:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Performance:**
- Win Rate (% positive years)
- Final Portfolio Value
- Recovery Time

---

## 🐛 Troubleshooting

### Skill Not Found
Make sure the skill file exists in `.claude/skills/[skill-name].md`

### Skill Fails
Check:
1. Is the data directory accessible?
2. Are dependencies installed?
3. Is the database schema correct?

### Slow Execution
- Use `quick-test` for rapid iteration
- Full backtests take 2-3 minutes (19 years of data)

---

## 🎓 Learn More

- [Claude Skills Documentation](https://docs.anthropic.com/claude/docs/claude-skills)
- [Trading Bot README](../README.md)
- [Strategy Documentation](../docs/strategies.md)
