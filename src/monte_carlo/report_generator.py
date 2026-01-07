"""
Generate Monte Carlo analysis reports
"""

from datetime import datetime
from typing import Dict, List
import os


def generate_monte_carlo_report(stats: Dict, results: List[Dict], output_path: str):
    """
    Generate comprehensive Monte Carlo analysis report
    
    Args:
        stats: Statistics dictionary from calculate_statistics()
        results: List of simulation results
        output_path: Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("MONTE CARLO ROBUSTNESS ANALYSIS - V28 MOMENTUM LEADERS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Simulations: {stats['total_simulations']}\n")
        f.write(f"Strategy: V28 Momentum Leaders\n")
        f.write("=" * 80 + "\n\n")
        
        # Annual Return Distribution
        if 'annual_return' in stats:
            ar = stats['annual_return']
            f.write("ANNUAL RETURN DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean:              {ar['mean']:>8.2f}%\n")
            f.write(f"Median:            {ar['median']:>8.2f}%\n")
            f.write(f"Std Dev:           {ar['std_dev']:>8.2f}%\n")
            f.write(f"Min:               {ar['min_value']:>8.2f}%\n")
            f.write(f"Max:               {ar['max_value']:>8.2f}%\n")
            f.write(f"5th Percentile:    {ar['percentile_05']:>8.2f}%\n")
            f.write(f"95th Percentile:   {ar['percentile_95']:>8.2f}%\n\n")
        
        # Max Drawdown Distribution
        if 'max_drawdown' in stats:
            dd = stats['max_drawdown']
            f.write("MAX DRAWDOWN DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean:              {dd['mean']:>8.2f}%\n")
            f.write(f"Median:            {dd['median']:>8.2f}%\n")
            f.write(f"Std Dev:           {dd['std_dev']:>8.2f}%\n")
            f.write(f"Best (min DD):     {dd['min_value']:>8.2f}%\n")
            f.write(f"Worst (max DD):    {dd['max_value']:>8.2f}%\n")
            f.write(f"5th Percentile:    {dd['percentile_05']:>8.2f}%\n")
            f.write(f"95th Percentile:   {dd['percentile_95']:>8.2f}%\n\n")
        
        # Sharpe Ratio Distribution
        if 'sharpe_ratio' in stats:
            sr = stats['sharpe_ratio']
            f.write("SHARPE RATIO DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean:              {sr['mean']:>8.2f}\n")
            f.write(f"Median:            {sr['median']:>8.2f}\n")
            f.write(f"Std Dev:           {sr['std_dev']:>8.2f}\n")
            f.write(f"Min:               {sr['min_value']:>8.2f}\n")
            f.write(f"Max:               {sr['max_value']:>8.2f}\n")
            f.write(f"5th Percentile:    {sr['percentile_05']:>8.2f}\n")
            f.write(f"95th Percentile:   {sr['percentile_95']:>8.2f}\n\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF MONTE CARLO ANALYSIS\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Report saved to {output_path}")
