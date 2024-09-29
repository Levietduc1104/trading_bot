import pandas as pd
import sys
sys.path.append(r'D:\BOT trading\trading_system')
import argparse
from data.mt5_data import MT5Data
from strategies.multiframe_strategy import MultiTimeFrameStrategy
from backtesting import Backtest
from utilities.helper_functions import convert_to_readable_datetime
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from bokeh.models import DatetimeTickFormatter
from bokeh.io import show
import argcomplete  # Import argcomplete


# Global default values (can be modified as needed)
LOGIN = 2432249
PASSWORD = 'Vinh12345@'
SERVER = "FivePercentOnline-Real"
SYMBOL = 'XAUUSD'
START_DATE = '2024-08-01'
END_DATE = '2024-08-31'
PRIMARY_TIMEFRAME = mt5.TIMEFRAME_M15
CONFIRM_TIMEFRAME = mt5.TIMEFRAME_H1


# Function to calculate and export key parameters
def calculate_and_export_stats(stats, trades):
    # Derive 'Direction' from 'EntryPrice' and 'ExitPrice'
    trades['Direction'] = trades.apply(lambda row: 'long' if row['EntryPrice'] < row['ExitPrice'] else 'short', axis=1)

    # Total trades
    total_trades = len(trades)
    
    # Separate winning and losing trades
    winning_trades = trades[trades['PnL'] > 0]
    losing_trades = trades[trades['PnL'] < 0]
    total_net_profit = trades['PnL'].sum()
    gross_profit = winning_trades['PnL'].sum()  # Gross Profit
    gross_loss = losing_trades['PnL'].sum()     # Gross Loss
    net_profit = gross_profit + gross_loss      # Net Profit (Note: gross_loss is negative)
    
    short_trades = trades[trades['Direction'] == 'short']  # Assuming 'Direction' is a column
    short_winning_trades = short_trades[short_trades['PnL'] > 0]
    short_positions_won_percent = (len(short_winning_trades) / len(short_trades)) * 100 if len(short_trades) > 0 else 0

    # Long positions won %
    long_trades = trades[trades['Direction'] == 'long']  # Assuming 'Direction' is a column
    long_winning_trades = long_trades[long_trades['PnL'] > 0]
    long_positions_won_percent = (len(long_winning_trades) / len(long_trades)) * 100 if len(long_trades) > 0 else 0
    
    
    profit_trades_percent = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    loss_trades_percent = (len(losing_trades) / total_trades) * 100 if total_trades > 0 else 0

    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    loss_rate = len(losing_trades) / total_trades * 100 if total_trades > 0 else 0
    
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    largest_profit_trade = winning_trades['PnL'].max() if len(winning_trades) > 0 else 0
    largest_loss_trade = losing_trades['PnL'].min() if len(losing_trades) > 0 else 0
    
    expected_payoff = net_profit / total_trades if total_trades > 0 else 0

    # Maximal Drawdown from backtesting stats
    max_drawdown = stats['Max. Drawdown [%]']
    relative_drawdown = stats['Max. Drawdown Duration']
    
    # Consecutive wins/losses (For this, you might need a custom implementation based on the trade history)
    consecutive_wins = (winning_trades['PnL'] > 0).sum()
    consecutive_losses = (losing_trades['PnL'] < 0).sum()
    
    # Compile all data into a dictionary
    summary_stats = {
        'Total Trades': total_trades,
        'Gross Profit': gross_profit,
        'Gross Loss': gross_loss,
        'Net Profit': net_profit,
        'Profit Factor': profit_factor,
        'Win Rate (%)': win_rate,
        'Loss Rate (%)': loss_rate,
        'Maximal Drawdown (%)': max_drawdown,
        'Relative Drawdown': relative_drawdown,
        'Largest Profit Trade': largest_profit_trade,
        'Largest Loss Trade': largest_loss_trade,
        'Expected Payoff': expected_payoff,
        'Consecutive Wins': consecutive_wins,
        'Consecutive Losses': consecutive_losses,
        'Total Net Profit': total_net_profit,
        'Short Positions Won %': short_positions_won_percent,
        'Long Positions Won %': long_positions_won_percent,
        'Profit Trades (% of total)': profit_trades_percent,
        'Loss Trades (% of total)': loss_trades_percent,
        
    }

    # Convert dictionary to DataFrame and export to Excel
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_excel('summary_stats.xlsx', index=False)
    print("Summary statistics have been exported to 'summary_stats.xlsx'")


def plot_entry_exit(bt, trades, primary_data):
    """
    Plot the entry and exit points of trades on the price chart.
    """
    fig = bt.plot()

    # Extracting entry and exit points from trades
    entry_times = pd.to_datetime(trades['EntryTime'], unit='s')
    exit_times = pd.to_datetime(trades['ExitTime'], unit='s')
    entry_prices = trades['EntryPrice']
    exit_prices = trades['ExitPrice']

    # Mark entries as green circles and exits as red circles
    fig.circle(entry_times, entry_prices, size=10, color='green', legend_label="Entry", fill_alpha=0.5)
    fig.circle(exit_times, exit_prices, size=10, color='red', legend_label="Exit", fill_alpha=0.5)

    # Show plot
    show(fig)

# Define main function to execute backtest
def main():
    global LOGIN, PASSWORD, SERVER, SYMBOL, START_DATE, END_DATE, PRIMARY_TIMEFRAME, CONFIRM_TIMEFRAME

    try:
        # Initialize MT5 data using the global variables
        mt5_data = MT5Data(
            login=LOGIN,
            password=PASSWORD,
            server=SERVER,
            symbol=SYMBOL,
            primary_timeframe=PRIMARY_TIMEFRAME,
            confirm_timeframe=CONFIRM_TIMEFRAME,
            start_date=START_DATE,
            end_date=END_DATE
        )
        primary_data = mt5_data.primary_data
        confirm_data = mt5_data.confirm_data

        # Create strategy instance
        strategy = create_strategy_class(primary_data, confirm_data)

        # Run the backtest
        bt = Backtest(primary_data, strategy, cash=10000, commission=0.0002)
        stats = bt.run()

        # Export equity curve and trade history to Excel
        equity_curve = stats['_equity_curve']
        equity_curve.to_excel('equity_curve.xlsx', index=True)
        
        trades = stats['_trades']
        trades = convert_to_readable_datetime(trades)
        trades.to_excel('trade_history.xlsx', index=True)
        calculate_and_export_stats(stats, trades)
        print("Results have been exported to 'equity_curve.xlsx' and 'trade_history.xlsx'.")
        print(stats)
        # Plot the results with entry and exit points
        plot_entry_exit(bt, trades, primary_data)
        # Plot the results
        fig = bt.plot()

        # Apply custom date formatting for better visuals
        fig.xaxis[0].formatter = DatetimeTickFormatter(
            days="%d %b",  # For example, '25 Sep'
            months="%b %Y",  # For example, 'Sep 2024'
            years="%Y"  # For example, '2024'
        )
        
        show(fig)
        
        
    except Exception as e:
        print(f"An error occurred: {e}")


def create_strategy_class(primary_data, confirm_data):
    class CustomMultiTimeFrameStrategy(MultiTimeFrameStrategy):
        def __init__(self, broker, *args, **kwargs):
            super().__init__(broker, primary_data, confirm_data, *args, **kwargs)
    return CustomMultiTimeFrameStrategy



if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run the trading bot with the --run command.")
    
    # Define the --run_cross_sma argument
    parser.add_argument('--run_cross_sma', action='store_true', help='Run the trading bot with cross SMA strategy.')

    # Initialize argcomplete for tab completion
    argcomplete.autocomplete(parser)

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Check if the --run_cross_sma argument is provided
    if args.run_cross_sma:
        print("Running the trading bot with cross SMA strategy...")
        main()
    else:
        print("Use --run_cross_sma argument to start the bot. Example: ./algobot --run_cross_sma")