import sys
import os

# Change to project root
os.chdir('/Users/levietduc/Documents/Documents - Le\'s MacBook Pro/Learning/ml_trading/trading_bot')
sys.path.insert(0, '.')

# Modify the SQL query in visualize_trades.py to use run_id 42
with open('src/visualize/visualize_trades.py', 'r') as f:
    script = f.read()

# Replace the query to get run_id 42 instead of latest
script = script.replace('ORDER BY run_id DESC LIMIT 1', 'WHERE run_id = 42')

# Execute the modified script
exec(script)
