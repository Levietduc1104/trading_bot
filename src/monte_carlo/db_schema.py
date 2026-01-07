"""
Database schema extension for Monte Carlo simulations
"""
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

def extend_database_schema(db_path):
    """Extend database with Monte Carlo tables"""
    logger.info(f"Extending database schema: {db_path}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Monte Carlo runs metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monte_carlo_runs (
                mc_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_type TEXT NOT NULL,
                simulation_type TEXT NOT NULL,
                num_simulations INTEGER NOT NULL,
                parameter_ranges TEXT,
                description TEXT,
                phase INTEGER DEFAULT 1
            )
        ''')
        
        # Individual simulation results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monte_carlo_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mc_run_id INTEGER NOT NULL,
                simulation_number INTEGER NOT NULL,
                parameters TEXT,
                annual_return REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                final_value REAL,
                win_rate REAL,
                start_date TEXT,
                end_date TEXT,
                years REAL,
                FOREIGN KEY (mc_run_id) REFERENCES monte_carlo_runs(mc_run_id)
            )
        ''')
        
        # Aggregated statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monte_carlo_statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mc_run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                mean REAL,
                median REAL,
                std_dev REAL,
                min_value REAL,
                max_value REAL,
                percentile_05 REAL,
                percentile_25 REAL,
                percentile_75 REAL,
                percentile_95 REAL,
                FOREIGN KEY (mc_run_id) REFERENCES monte_carlo_runs(mc_run_id)
            )
        ''')
        
        conn.commit()
        logger.info("✓ Monte Carlo schema extension complete")
        
    except Exception as e:
        logger.error(f"Schema extension failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(project_root, 'output', 'data', 'trading_results.db')
    
    print("=" * 80)
    print("MONTE CARLO DATABASE SCHEMA EXTENSION")
    print("=" * 80)
    print(f"Database: {db_path}\n")
    
    extend_database_schema(db_path)
    print("\n✓ Schema extension successful\!")
    print("=" * 80)
