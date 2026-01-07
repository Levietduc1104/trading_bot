"""
Phase 1 Monte Carlo Testing for V28 Strategy
Runs 77 simulations: 45 parameter sweep + 20 stress tests + 12 bootstrap
"""

import sys
import os
sys.path.append('src')

from monte_carlo.simulator import MonteCarloSimulator
from monte_carlo.statistics import calculate_statistics
from monte_carlo.report_generator import generate_monte_carlo_report
from visualize.monte_carlo_visualizer import MonteCarloVisualizer
from datetime import datetime
import pandas as pd

def run_phase1():
    """Run Phase 1 Monte Carlo analysis (77 simulations)"""

    print("="*80)
    print("PHASE 1 MONTE CARLO ANALYSIS - V28 MOMENTUM LEADERS")
    print("="*80)
    print(f"Total simulations: 77")
    print(f"  - Parameter sweep: 45")
    print(f"  - Stress tests: 20")
    print(f"  - Bootstrap: 12")
    print(f"Expected runtime: 15-25 minutes")
    print("="*80)
    print()

    # Initialize simulator
    simulator = MonteCarloSimulator(
        data_dir='sp500_data/daily',
        initial_capital=100000,
        strategy='V28'
    )

    sim_count = 0

    # =========================================================================
    # PART 1: PARAMETER SWEEP (45 simulations)
    # =========================================================================
    print("[1/3] Running Parameter Sweep (45 simulations)...")
    print("-"*80)

    portfolio_sizes = [None, 3, 5, 7, 10]  # None = dynamic
    kelly_exponents = [0.4, 0.5, 0.6]
    vix_multipliers = [0.85, 1.0, 1.15]

    for psize in portfolio_sizes:
        for kelly in kelly_exponents:
            for vix_mult in vix_multipliers:
                sim_count += 1
                print(f"  Sim {sim_count}/77: psize={psize}, kelly={kelly:.1f}, vix_mult={vix_mult:.2f}...", end='', flush=True)

                result = simulator.run_single_simulation(
                    sim_id=sim_count,
                    portfolio_size=psize,
                    kelly_exponent=kelly,
                    vix_multiplier=vix_mult,
                    fee_pct=0.001,
                    start_offset_days=0,
                    stress_scenario=None
                )

                print(f" ✓ {result['annual_return']:.2f}%, {result['max_drawdown']:.2f}%, {result['sharpe_ratio']:.2f}")

    print(f"✓ Parameter sweep complete ({sim_count} simulations)")
    print()

    # =========================================================================
    # PART 2: STRESS TESTS (20 simulations)
    # =========================================================================
    print("[2/3] Running Stress Tests (20 simulations)...")
    print("-"*80)

    stress_scenarios = [
        ('2008_crisis', 5),
        ('2020_covid', 5),
        ('prolonged_bear', 5),
        ('flash_crash', 5)
    ]

    for scenario, count in stress_scenarios:
        for i in range(count):
            sim_count += 1
            print(f"  Sim {sim_count}/77: {scenario} #{i+1}...", end='', flush=True)

            result = simulator.run_single_simulation(
                sim_id=sim_count,
                portfolio_size=None,
                kelly_exponent=0.5,
                vix_multiplier=1.0,
                fee_pct=0.001,
                start_offset_days=0,
                stress_scenario=scenario
            )

            print(f" ✓ {result['annual_return']:.2f}%, {result['max_drawdown']:.2f}%, {result['sharpe_ratio']:.2f}")

    print(f"✓ Stress tests complete (20 simulations)")
    print()

    # =========================================================================
    # PART 3: BOOTSTRAP (12 simulations)
    # =========================================================================
    print("[3/3] Running Bootstrap Analysis (12 simulations)...")
    print("-"*80)

    start_offsets = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660]

    for offset in start_offsets:
        sim_count += 1
        print(f"  Sim {sim_count}/77: start_offset={offset}d...", end='', flush=True)

        result = simulator.run_single_simulation(
            sim_id=sim_count,
            portfolio_size=None,
            kelly_exponent=0.5,
            vix_multiplier=1.0,
            fee_pct=0.001,
            start_offset_days=offset,
            stress_scenario=None
        )

        print(f" ✓ {result['annual_return']:.2f}%, {result['max_drawdown']:.2f}%, {result['sharpe_ratio']:.2f}")

    print(f"✓ Bootstrap complete (12 simulations)")
    print()

    # =========================================================================
    # ANALYSIS AND REPORTING
    # =========================================================================
    print("="*80)
    print("ANALYSIS")
    print("="*80)

    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(simulator.results)

    # Create DataFrame
    results_df = pd.DataFrame(simulator.results)

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('output/monte_carlo/reports', exist_ok=True)
    report_path = f'output/monte_carlo/reports/phase1_{timestamp}.txt'

    print("Generating report...")
    generate_monte_carlo_report(stats, simulator.results, report_path)
    print(f"✓ Report: {report_path}")

    # Generate visualizations
    os.makedirs('output/monte_carlo/plots', exist_ok=True)
    viz_path = f'output/monte_carlo/plots/phase1_{timestamp}.html'

    print("Generating visualizations...")
    visualizer = MonteCarloVisualizer(results_df, stats)
    visualizer.save_html(viz_path)

    # Print summary
    print()
    print("="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print()
    print("RESULTS SUMMARY:")
    print(f"  Simulations:       {len(simulator.results)}")
    print(f"  Annual Return:     {stats['annual_return']['mean']:.2f}% (median: {stats['annual_return']['median']:.2f}%)")
    print(f"  95% CI:            {stats['annual_return']['percentile_05']:.2f}% to {stats['annual_return']['percentile_95']:.2f}%")
    print(f"  Max Drawdown:      {stats['max_drawdown']['mean']:.2f}% (median: {stats['max_drawdown']['median']:.2f}%)")
    print(f"  95% CI:            {stats['max_drawdown']['percentile_05']:.2f}% to {stats['max_drawdown']['percentile_95']:.2f}%")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']['mean']:.2f} (median: {stats['sharpe_ratio']['median']:.2f})")
    print()
    print("OUTPUT FILES:")
    print(f"  📋 {report_path}")
    print(f"  📊 {viz_path}")
    print()
    print("="*80)

    return results_df, stats, viz_path

if __name__ == '__main__':
    results_df, stats, viz_path = run_phase1()
    print(f"\n✓ Phase 1 complete\! Open: {viz_path}")
