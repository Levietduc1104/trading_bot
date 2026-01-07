"""Test Monte Carlo visualizations with quick test data"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import sqlite3
from monte_carlo.statistics import calculate_statistics
from visualize.monte_carlo_visualizer import create_visualizations

print("="*80)
print("TESTING MONTE CARLO VISUALIZATIONS")
print("="*80)

# Load results from database (if they exist) or create sample data
db_path = 'output/data/trading_results.db'

try:
    conn = sqlite3.connect(db_path)
    
    # Get latest MC run
    cursor = conn.cursor()
    cursor.execute("""
        SELECT mc_run_id, run_name, num_simulations 
        FROM monte_carlo_runs 
        ORDER BY mc_run_id DESC LIMIT 1
    """)
    
    run_info = cursor.fetchone()
    
    if run_info:
        mc_run_id, run_name, num_sims = run_info
        print(f"\nLoading results from: {run_name}")
        print(f"  MC Run ID: {mc_run_id}")
        print(f"  Simulations: {num_sims}")
        
        # Load results
        results_df = pd.read_sql(f"""
            SELECT * FROM monte_carlo_results 
            WHERE mc_run_id = {mc_run_id}
        """, conn)
        
        print(f"\n✓ Loaded {len(results_df)} simulation results")
        
        conn.close()
        
    else:
        print("\n⚠️  No Monte Carlo runs found in database")
        print("Creating sample data for visualization test...")
        
        # Create sample data
        np.random.seed(42)
        n_sims = 20
        
        results_df = pd.DataFrame({
            'simulation_number': range(1, n_sims+1),
            'annual_return': np.random.normal(9.0, 1.5, n_sims),
            'max_drawdown': np.random.normal(-18, 3, n_sims),
            'sharpe_ratio': np.random.normal(1.0, 0.15, n_sims),
            'sortino_ratio': np.random.normal(1.5, 0.2, n_sims),
            'final_value': np.random.normal(550000, 50000, n_sims),
            'parameters': [{'portfolio_size': None, 'kelly_exponent': 0.5, 
                          'vix_multiplier': 1.0, 'start_offset_days': i*30} 
                         for i in range(n_sims)]
        })
        
        print(f"✓ Created {len(results_df)} sample results")
        
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Calculate statistics
print("\nCalculating statistics...")
results_list = results_df.to_dict('records')
statistics = calculate_statistics(results_list)

print(f"✓ Statistics calculated for {statistics.get('total_simulations', 0)} simulations")

# Generate visualizations
print("\nGenerating visualizations...")
output_file = 'output/monte_carlo/plots/mc_analysis.html'

try:
    create_visualizations(results_df, statistics, output_file)
    print(f"\n✅ SUCCESS\! Visualizations created")
    print(f"   Open: {output_file}")
    
    # Show statistics summary
    print("\nQuick Statistics:")
    if 'annual_return' in statistics:
        ar = statistics['annual_return']
        print(f"  Annual Return: {ar['mean']:.2f}% ± {ar['std_dev']:.2f}%")
        print(f"  95% CI: [{ar['percentile_05']:.2f}%, {ar['percentile_95']:.2f}%]")
    
    if 'max_drawdown' in statistics:
        dd = statistics['max_drawdown']
        print(f"  Max Drawdown: {dd['mean']:.2f}% ± {dd['std_dev']:.2f}%")
        print(f"  95% CI: [{dd['percentile_05']:.2f}%, {dd['percentile_95']:.2f}%]")
    
except Exception as e:
    print(f"\n❌ Visualization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*80)
