"""
Walk-Forward Validation Report
================================
Reads existing walk-forward results from output/walk_forward/ and produces:
  - Per-year performance table (ML vs Phase 1A)
  - Win rate, average annual return, worst year
  - Regime breakdown: bull / bear / volatile years
  - Consistency score (% years with positive return)
  - Comparison: rolling-train model vs full-history model on 2023-2024

The walk-forward was run with:
  Train start : 2006 (expanding window)
  Test window : 1 year per fold (2013 → 2024)
  Strategy    : V31 ML (quarterly rebalance, 15% trailing stop)

Usage:
  python walk_forward_report.py                  # print report
  python walk_forward_report.py --save           # also save to output/wf_report.txt
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WF_RESULTS   = 'output/walk_forward/wf_v31_ml_results.csv'
PER_YEAR     = 'output/walk_forward/per_year_results.csv'
FULL_HIST_TEST = 'output/full_history_test_2023_2024.csv'


def _header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def load_data():
    wf  = pd.read_csv(WF_RESULTS)
    pyr = pd.read_csv(PER_YEAR)
    return wf, pyr


def print_per_year_table(wf: pd.DataFrame):
    _header("WALK-FORWARD: PER-YEAR RESULTS (V31 ML, Expanding Train Window)")

    print(f"\n  {'Year':<6} {'Train End':<10} {'ML Annual':>10} {'1A Annual':>10} "
          f"{'Alpha':>8} {'ML DD':>8} {'ML Sharpe':>10}  Winner")
    print(f"  {'-'*70}")

    ml_wins = 0
    for _, row in wf.iterrows():
        winner = row['winner']
        alpha  = row['ml_annual'] - row['annual_1a']
        marker = '*' if winner == 'ML' else ' '
        if winner == 'ML':
            ml_wins += 1
        print(f"  {int(row['test_year']):<6} {int(row['train_end']):<10} "
              f"{row['ml_annual']:>+9.2f}% {row['annual_1a']:>+9.2f}% "
              f"{alpha:>+7.2f}% {row['ml_max_dd']:>+7.2f}% "
              f"{row['ml_sharpe']:>9.3f}  {winner}{marker}")

    print(f"  {'-'*70}")
    print(f"  ML wins : {ml_wins}/{len(wf)}  ({ml_wins/len(wf)*100:.0f}%)")


def print_aggregate_stats(wf: pd.DataFrame):
    _header("AGGREGATE STATISTICS")

    ml_ann  = wf['ml_annual']
    ml_dd   = wf['ml_max_dd']
    ml_sh   = wf['ml_sharpe']
    a1_ann  = wf['annual_1a']

    # Classify years
    bull_mask = wf['ml_annual'] > 20
    bear_mask = wf['ml_annual'] < 0
    flat_mask = ~bull_mask & ~bear_mask

    print(f"\n  V31 ML across {len(wf)} test years (2013-2024):")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Avg annual return    : {ml_ann.mean():+.2f}%")
    print(f"  Median annual return : {ml_ann.median():+.2f}%")
    print(f"  Std annual return    : {ml_ann.std():.2f}%")
    print(f"  Best year            : {ml_ann.max():+.2f}%  ({int(wf.loc[ml_ann.idxmax(), 'test_year'])})")
    print(f"  Worst year           : {ml_ann.min():+.2f}%  ({int(wf.loc[ml_ann.idxmin(), 'test_year'])})")
    print(f"  Positive return years: {(ml_ann > 0).sum()}/{len(wf)}  "
          f"({(ml_ann > 0).mean()*100:.0f}%)")
    print(f"  Years >20% return    : {bull_mask.sum()}")
    print(f"  Years <0% (loss)     : {bear_mask.sum()}")
    print(f"  Avg max drawdown     : {ml_dd.mean():+.2f}%")
    print(f"  Worst drawdown year  : {ml_dd.min():+.2f}%  ({int(wf.loc[ml_dd.idxmin(), 'test_year'])})")
    print(f"  Avg Sharpe ratio     : {ml_sh.mean():.3f}")

    print(f"\n  ML vs Phase 1A:")
    avg_alpha = (wf['ml_annual'] - wf['annual_1a']).mean()
    win_rate  = (wf['winner'] == 'ML').mean()
    print(f"  Avg alpha vs 1A      : {avg_alpha:+.2f}% per year")
    print(f"  ML win rate vs 1A    : {win_rate*100:.0f}%  ({(wf['winner']=='ML').sum()}/{len(wf)} years)")


def print_regime_breakdown(wf: pd.DataFrame):
    _header("REGIME BREAKDOWN")

    # Classify each test year by market regime
    regime_map = {
        2013: 'Bull',   2014: 'Bull',   2015: 'Flat',
        2016: 'Bull',   2017: 'Bull',   2018: 'Bear',
        2019: 'Bull',   2020: 'Mixed',  2021: 'Bull',
        2022: 'Bear',   2023: 'Bull',   2024: 'Bull',
    }
    wf = wf.copy()
    wf['regime'] = wf['test_year'].astype(int).map(regime_map).fillna('Unknown')

    print(f"\n  {'Regime':<8} {'Count':>6} {'Avg ML%':>10} {'Avg 1A%':>10} {'Avg Alpha':>10}")
    print(f"  {'-'*50}")
    for regime, grp in wf.groupby('regime'):
        avg_ml  = grp['ml_annual'].mean()
        avg_1a  = grp['annual_1a'].mean()
        avg_alp = avg_ml - avg_1a
        print(f"  {regime:<8} {len(grp):>6} {avg_ml:>+9.2f}% {avg_1a:>+9.2f}% {avg_alp:>+9.2f}%")

    print(f"\n  Key stress years:")
    for yr, label in [(2018, '2018 Q4 selloff'), (2020, 'COVID crash+recovery'),
                      (2022, 'Rate shock')]:
        row = wf[wf['test_year'] == yr]
        if not row.empty:
            r = row.iloc[0]
            print(f"    {yr} ({label:<25}) ML: {r['ml_annual']:+.2f}%  "
                  f"1A: {r['annual_1a']:+.2f}%  DD: {r['ml_max_dd']:+.2f}%")


def print_full_history_comparison():
    _header("FULL-HISTORY MODEL vs ROLLING-TRAIN MODEL (2023-2024)")

    # Rolling-train result from walk-forward
    wf = pd.read_csv(WF_RESULTS)
    row_2023 = wf[wf['test_year'] == 2023]
    row_2024 = wf[wf['test_year'] == 2024]

    print(f"\n  Rolling-train model (train expands from 2006):")
    if not row_2023.empty:
        r = row_2023.iloc[0]
        print(f"    2023: ML={r['ml_annual']:+.2f}%  DD={r['ml_max_dd']:.2f}%  "
              f"Sharpe={r['ml_sharpe']:.3f}  val_corr={r['val_corr']:.4f}")
    if not row_2024.empty:
        r = row_2024.iloc[0]
        print(f"    2024: ML={r['ml_annual']:+.2f}%  DD={r['ml_max_dd']:.2f}%  "
              f"Sharpe={r['ml_sharpe']:.3f}  val_corr={r['val_corr']:.4f}")

    print(f"\n  Full-history model (train=1997-2022, test=2023-2024 combined):")
    if os.path.exists(FULL_HIST_TEST):
        df = pd.read_csv(FULL_HIST_TEST, index_col=0, parse_dates=True)
        if 'value' in df.columns:
            final = df['value'].iloc[-1]
            years = (df.index[-1] - df.index[0]).days / 365.25
            annual = ((final / 100_000) ** (1/years) - 1) * 100
            cummax = df['value'].cummax()
            max_dd = ((df['value'] - cummax) / cummax * 100).min()
            rets = df['value'].pct_change().dropna()
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
            print(f"    2023-24: annual={annual:+.2f}%  DD={max_dd:.2f}%  Sharpe={sharpe:.3f}")
        else:
            print(f"    (result file found but 'value' column missing)")
    else:
        print(f"    (run train_full_history.py first)")

    print(f"\n  Takeaway:")
    print(f"    Full-history model uses 3x more training data (1997-2022 vs 2006-train_end)")
    print(f"    and covers more regimes → expected to be more robust out-of-sample.")


def print_consistency_score(wf: pd.DataFrame):
    _header("CONSISTENCY SCORE")

    pos_years   = (wf['ml_annual'] > 0).sum()
    beat_spy    = (wf['ml_annual'] > 15).sum()   # rough SPY proxy ~12% avg
    beat_1a     = (wf['winner'] == 'ML').sum()
    no_big_dd   = (wf['ml_max_dd'] > -15).sum()

    total = len(wf)
    score = (pos_years + beat_1a + no_big_dd) / (3 * total) * 100

    print(f"\n  Positive return years : {pos_years}/{total}  ({pos_years/total*100:.0f}%)")
    print(f"  Beat Phase 1A years   : {beat_1a}/{total}  ({beat_1a/total*100:.0f}%)")
    print(f"  Drawdown < 15% years  : {no_big_dd}/{total}  ({no_big_dd/total*100:.0f}%)")
    print(f"\n  Consistency score     : {score:.0f}/100")
    if score >= 75:
        print(f"  Assessment            : STRONG — model generalizes well across regimes")
    elif score >= 55:
        print(f"  Assessment            : MODERATE — generally good but some weak spots")
    else:
        print(f"  Assessment            : WEAK — high variance, review model/features")


def run(save: bool = False):
    if not os.path.exists(WF_RESULTS):
        print(f"ERROR: Walk-forward results not found at {WF_RESULTS}")
        print("Run the walk-forward backtest first (see src/backtest/walk_forward_backtest.py)")
        return

    wf, pyr = load_data()

    lines = []
    import io, contextlib

    # Capture output if saving
    if save:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_all(wf)
        output = buf.getvalue()
        print(output)
        with open('output/wf_report.txt', 'w') as f:
            f.write(output)
        print(f"\n  Report saved: output/wf_report.txt")
    else:
        _print_all(wf)


def _print_all(wf: pd.DataFrame):
    print_per_year_table(wf)
    print_aggregate_stats(wf)
    print_regime_breakdown(wf)
    print_consistency_score(wf)
    print_full_history_comparison()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Walk-Forward Validation Report')
    parser.add_argument('--save', action='store_true',
                        help='Save report to output/wf_report.txt')
    args = parser.parse_args()
    run(save=args.save)
