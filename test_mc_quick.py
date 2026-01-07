"""Quick Monte Carlo test - 5 simulations"""
import sys
sys.path.append('src')

from monte_carlo.simulator import MonteCarloSimulator
from monte_carlo.statistics import calculate_statistics
from monte_carlo.report_generator import generate_monte_carlo_report
from datetime import datetime
import os

print("="*80)
print("QUICK MONTE CARLO TEST (5 SIMULATIONS)")
print("="*80)

simulator = MonteCarloSimulator(
    data_dir='sp500_data/daily',
    initial_capital=100000,
    strategy='V28'
)

results = []
for i in range(1, 6):
    print(f"\nRunning simulation {i}/5...")
    result = simulator.run_single_simulation(sim_id=i)
    if result:
        results.append(result)
        print(f"  ✓ Return: {result['annual_return']:.2f}%, DD: {result['max_drawdown']:.2f}%, Sharpe: {result['sharpe_ratio']:.2f}")

print(f"\n{'='*80}")
print(f"Completed {len(results)} simulations")

if results:
    stats = calculate_statistics(results)
    
    print("\nRESULTS:")
    if 'annual_return' in stats:
        ar = stats['annual_return']
        print(f"  Annual Return: {ar['mean']:.2f}% (median: {ar['median']:.2f}%)")
        print(f"  Range: {ar['min_value']:.2f}% to {ar['max_value']:.2f}%")
    
    if 'max_drawdown' in stats:
        dd = stats['max_drawdown']
        print(f"  Max Drawdown: {dd['mean']:.2f}% (median: {dd['median']:.2f}%)")
    
    # Save report
    report_path = f'output/monte_carlo/reports/quick_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    generate_monte_carlo_report(stats, results, report_path)
    print(f"\n  Report saved: {report_path}")

print("="*80)
