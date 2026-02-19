# Repository Cleanup Summary

## Production Files (KEEP in root)

### 🎯 Tier 2 Production Documentation
- ✅ `TIER2_PRODUCTION_DEPLOYMENT_GUIDE.md` - Complete deployment guide
- ✅ `HOW_TO_CHECK_POSITIONS.md` - Check buy/sell recommendations
- ✅ `LIVE_TRADING_MONITORING_CHECKLIST.md` - Daily/weekly/monthly monitoring
- ✅ `PAPER_TRADING_QUICK_START.md` - 3-month paper trading guide
- ✅ `TRADE_HISTORY_GUIDE.md` - Using trade CSV export
- ✅ `TIER2_MULTI_PERIOD_RESULTS.md` - 34-year validation results

### 🛠️ Tier 2 Production Utilities
- ✅ `check_tier2_positions.py` - Check current recommendations
- ✅ `export_tier2_recommendations.py` - Export recommendations to CSV

## Cleanup Actions

### 1. Archive Development Documentation
**Move to `docs/archive/development/`:**
- All `docs/*.md` files (67+ files)
- Development notes, experiments, analysis docs
- These are valuable history but not needed for production

### 2. Remove Test/Temporary Files
**Delete:**
- `*.bak` files (backups)
- Files with " 2.py" suffix (duplicates)
- `compare_*.py` test scripts
- `test_*.py` ad-hoc test scripts
- `run_v30_*.py` old test runners
- `visualize_*.py` (except in src/visualize/)
- `*_output.txt` files
- `simple_spy_comparison.py`
- `calculate_*.py` one-off analysis scripts

### 3. Clean Data Directory
**Archive to `sp500_data/archive/`:**
- `all_stock_data.csv` (consolidated file not used)
- `individual_stocks/` (if not needed)
- `sp500_filtered/` (if not needed)
- `alpaca_data_fetchers/` (old scripts)

### 4. Remove Old Code
**Delete from `src/`:**
- `src/core/execution_*.py` (old versions with suffixes)
- `src/strategies/*_backup.py` (backup files)
- `src/strategies/*.bak` (backup files)

### 5. Update .gitignore
**Add patterns:**
```
# Test/temporary files
*_output.txt
*.bak
*.bak2
*_backup.py
compare_*.py
test_*.py
run_v*.py
visualize_*.py (root level)
simple_*.py

# Data archives
sp500_data/archive/
sp500_data/individual_stocks/
sp500_data/sp500_filtered/

# Output artifacts (keep CSV/reports)
output/*.png
output/*.html
output/data/*.db
```

## Final Structure

```
trading_bot/
├── README.md (main documentation)
│
├── 📚 PRODUCTION DOCS (Root)
│   ├── TIER2_PRODUCTION_DEPLOYMENT_GUIDE.md
│   ├── HOW_TO_CHECK_POSITIONS.md
│   ├── LIVE_TRADING_MONITORING_CHECKLIST.md
│   ├── PAPER_TRADING_QUICK_START.md
│   ├── TRADE_HISTORY_GUIDE.md
│   └── TIER2_MULTI_PERIOD_RESULTS.md
│
├── 🛠️ UTILITIES (Root)
│   ├── check_tier2_positions.py
│   └── export_tier2_recommendations.py
│
├── docs/
│   ├── archive/
│   │   └── development/ (all historical docs)
│   └── (keep only essential docs)
│
├── src/
│   ├── core/
│   │   └── execution.py (production)
│   ├── strategies/
│   │   └── v31_tier2_growth_scoring.py (production)
│   └── visualize/
│       └── visualize_trades.py
│
├── sp500_data/
│   ├── stock_data_1990_2024/ (main data)
│   └── archive/ (old/unused data)
│
└── output/
    └── tier2_trades.csv (production output)
```

## Cleanup Commands

```bash
# 1. Create archive directories
mkdir -p docs/archive/development
mkdir -p sp500_data/archive

# 2. Move development docs to archive
mv docs/*.md docs/archive/development/ 2>/dev/null

# 3. Remove test scripts (from root)
rm -f compare_*.py test_*.py run_v30_*.py run_v31_*.py 2>/dev/null
rm -f visualize_*.py simple_*.py calculate_*.py 2>/dev/null
rm -f *_output.txt position_sizing_*.txt 2>/dev/null

# 4. Remove backup files
find . -name "*.bak" -delete
find . -name "*.bak2" -delete
find . -name "*backup.py" -delete
find . -name "* 2.py" -delete
find . -name "* 3.py" -delete
find . -name "* 4.py" -delete

# 5. Clean old code versions
rm -f src/core/execution_*.py 2>/dev/null
rm -f src/strategies/*.bak 2>/dev/null

# 6. Update gitignore
cat >> .gitignore << 'EOF'

# Test files
*_output.txt
*.bak
*.bak2
*_backup.py

# Root test scripts
compare_*.py
test_*.py
run_v*.py
/visualize_*.py
simple_*.py
calculate_*.py

# Archives
sp500_data/archive/
docs/archive/

# Output artifacts (keep CSVs)
output/*.png
output/*.html
output/plots/
output/reports/
EOF
```

## After Cleanup

Run these to verify:
```bash
# Check what's left untracked
git status --short | grep "^??"

# Should only see:
# - Production docs (6 files)
# - Utilities (2 files)
# - Essential new files

# Then commit
git add .
git commit -m "Tier 2 production deployment: Add trade CSV export + cleanup"
```
