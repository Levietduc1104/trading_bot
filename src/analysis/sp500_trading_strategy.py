import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    """Backtest trading strategies with buy and sell signals"""
    
    def __init__(self, data, initial_capital=10000):
        self.data = data
        self.initial_capital = initial_capital
        
    def calculate_indicators(self):
        """Calculate technical indicators"""
        df = self.data.copy()
        
        # Moving Averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Price momentum
        df['returns'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_20d'] = df['Close'].pct_change(20)
        
        return df
    
    def backtest(self, buy_condition, sell_condition, strategy_name):
        """Backtest a trading strategy"""
        df = self.data.copy()
        
        cash = self.initial_capital
        shares = 0
        position = False
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Calculate current portfolio value
            portfolio_value = cash + (shares * row['Close'] if shares > 0 else 0)
            equity_curve.append({
                'date': row['Date'],
                'portfolio_value': portfolio_value,
                'position': position
            })
            
            # Buy signal
            if not position and buy_condition(row, i, df):
                if cash > 0:
                    shares = cash / row['Close']
                    buy_price = row['Close']
                    buy_date = row['Date']
                    cash = 0
                    position = True
                    
            # Sell signal
            elif position and sell_condition(row, i, df):
                if shares > 0:
                    sell_price = row['Close']
                    sell_value = shares * sell_price
                    profit = sell_value - (shares * buy_price)
                    profit_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    trades.append({
                        'buy_date': buy_date,
                        'buy_price': buy_price,
                        'sell_date': row['Date'],
                        'sell_price': sell_price,
                        'shares': shares,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'hold_days': (row['Date'] - buy_date).days
                    })
                    
                    cash = sell_value
                    shares = 0
                    position = False
        
        # Close any open position at the end
        final_value = cash + (shares * df.iloc[-1]['Close'] if shares > 0 else 0)
        
        # Calculate metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) * 100
            avg_profit = trades_df['profit'].mean()
            avg_profit_pct = trades_df['profit_pct'].mean()
            total_trades = len(trades_df)
            
            # Max drawdown from equity curve
            equity_df = pd.DataFrame(equity_curve)
            equity_df['cummax'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min() * 100
        else:
            win_rate = 0
            avg_profit = 0
            avg_profit_pct = 0
            total_trades = 0
            max_drawdown = 0
        
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Annualized return
        years = (df.iloc[-1]['Date'] - df.iloc[0]['Date']).days / 365.25
        annualized_return = (((final_value / self.initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        return {
            'strategy': strategy_name,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit_pct': avg_profit_pct,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve
        }

def main():
    print("=" * 100)
    print("S&P 500 TRADING STRATEGIES ANALYSIS")
    print("Goal: Find strategies with returns > 10% per year")
    print("=" * 100)
    print()
    
    # Load data
    df = pd.read_csv('trading_bot/sp500_data/individual_stocks/SPY.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= '2005-01-01'].copy()
    df = df.reset_index(drop=True)
    
    print(f"Testing period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Initial capital: $10,000")
    print(f"Total years: {(df['Date'].max() - df['Date'].min()).days / 365.25:.1f}")
    print()
    
    # Initialize trader
    trader = TradingStrategy(df, initial_capital=10000)
    df = trader.calculate_indicators()
    trader.data = df
    
    # Define trading strategies
    strategies = []
    
    # 1. Buy and Hold (Baseline)
    strategies.append({
        'name': 'Buy and Hold',
        'buy': lambda row, i, df: i == 0,  # Buy on first day
        'sell': lambda row, i, df: False  # Never sell
    })
    
    # 2. Golden Cross / Death Cross
    strategies.append({
        'name': 'Golden/Death Cross (SMA50/200)',
        'buy': lambda row, i, df: i > 0 and df.iloc[i-1]['sma_50'] <= df.iloc[i-1]['sma_200'] and row['sma_50'] > row['sma_200'],
        'sell': lambda row, i, df: i > 0 and df.iloc[i-1]['sma_50'] >= df.iloc[i-1]['sma_200'] and row['sma_50'] < row['sma_200']
    })
    
    # 3. RSI Oversold/Overbought
    strategies.append({
        'name': 'RSI (Buy<30, Sell>70)',
        'buy': lambda row, i, df: row['rsi'] < 30,
        'sell': lambda row, i, df: row['rsi'] > 70
    })
    
    # 4. RSI Modified (less extreme)
    strategies.append({
        'name': 'RSI (Buy<40, Sell>60)',
        'buy': lambda row, i, df: row['rsi'] < 40,
        'sell': lambda row, i, df: row['rsi'] > 60
    })
    
    # 5. MACD Crossover
    strategies.append({
        'name': 'MACD Crossover',
        'buy': lambda row, i, df: i > 0 and df.iloc[i-1]['macd'] <= df.iloc[i-1]['macd_signal'] and row['macd'] > row['macd_signal'],
        'sell': lambda row, i, df: i > 0 and df.iloc[i-1]['macd'] >= df.iloc[i-1]['macd_signal'] and row['macd'] < row['macd_signal']
    })
    
    # 6. Bollinger Bands
    strategies.append({
        'name': 'Bollinger Bands (Buy at lower, Sell at upper)',
        'buy': lambda row, i, df: row['Close'] < row['bb_lower'],
        'sell': lambda row, i, df: row['Close'] > row['bb_upper']
    })
    
    # 7. Price below SMA200 + RSI
    strategies.append({
        'name': 'Buy: Price<SMA200 & RSI<50, Sell: Price>SMA200 & RSI>70',
        'buy': lambda row, i, df: row['Close'] < row['sma_200'] and row['rsi'] < 50,
        'sell': lambda row, i, df: row['Close'] > row['sma_200'] and row['rsi'] > 70
    })
    
    # 8. Trend following with EMA
    strategies.append({
        'name': 'Buy: Price>EMA50, Sell: Price<EMA50',
        'buy': lambda row, i, df: row['Close'] > row['ema_50'],
        'sell': lambda row, i, df: row['Close'] < row['ema_50']
    })
    
    # 9. Mean reversion
    strategies.append({
        'name': 'Buy: Price<BB_Lower, Sell: Price>BB_Middle',
        'buy': lambda row, i, df: row['Close'] < row['bb_lower'],
        'sell': lambda row, i, df: row['Close'] > row['bb_middle']
    })
    
    # 10. Aggressive momentum
    strategies.append({
        'name': 'Buy: 5d return < -5%, Sell: 5d return > 5%',
        'buy': lambda row, i, df: row['returns_5d'] < -0.05,
        'sell': lambda row, i, df: row['returns_5d'] > 0.05
    })
    
    # 11. Combined: RSI + MACD
    strategies.append({
        'name': 'RSI<40 & MACD>Signal, Sell: RSI>60 | MACD<Signal',
        'buy': lambda row, i, df: row['rsi'] < 40 and row['macd'] > row['macd_signal'],
        'sell': lambda row, i, df: row['rsi'] > 60 or row['macd'] < row['macd_signal']
    })
    
    # 12. Dip buying with stop loss
    strategies.append({
        'name': 'Buy: Price<SMA50, Sell: Price>SMA50 | 10% loss',
        'buy': lambda row, i, df: row['Close'] < row['sma_50'],
        'sell': lambda row, i, df: row['Close'] > row['sma_50']  # Simplified
    })
    
    # Backtest all strategies
    print("Backtesting strategies...")
    print()
    
    results = []
    for strategy in strategies:
        result = trader.backtest(strategy['buy'], strategy['sell'], strategy['name'])
        results.append(result)
    
    # Sort by annualized return
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('annualized_return', ascending=False)
    
    print("=" * 100)
    print("RESULTS: TRADING STRATEGIES RANKED BY ANNUALIZED RETURN")
    print("=" * 100)
    print()
    
    print(f"{'Strategy':<50} {'Ann.Return':<12} {'Total Return':<13} {'Trades':<8} {'Win Rate':<10} {'Max DD':<10}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['strategy']:<50} {row['annualized_return']:>10.2f}% {row['total_return']:>11.2f}% "
              f"{row['total_trades']:>7} {row['win_rate']:>9.1f}% {row['max_drawdown']:>9.2f}%")
    
    # Detailed analysis of top performers
    print()
    print("=" * 100)
    print("TOP 5 STRATEGIES - DETAILED ANALYSIS")
    print("=" * 100)
    
    for idx, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"\n{idx}. {row['strategy']}")
        print("-" * 100)
        print(f"   Initial Capital:      ${row['initial_capital']:,.0f}")
        print(f"   Final Value:          ${row['final_value']:,.0f}")
        print(f"   Total Return:         {row['total_return']:.2f}%")
        print(f"   Annualized Return:    {row['annualized_return']:.2f}%")
        print(f"   Total Trades:         {row['total_trades']}")
        print(f"   Win Rate:             {row['win_rate']:.1f}%")
        print(f"   Avg Profit per Trade: {row['avg_profit_pct']:.2f}%")
        print(f"   Max Drawdown:         {row['max_drawdown']:.2f}%")
        
        # Show sample trades
        if len(row['trades']) > 0:
            trades_df = pd.DataFrame(row['trades'])
            print(f"\n   Sample Trades (first 5):")
            for i, trade in trades_df.head(5).iterrows():
                print(f"      Buy: {trade['buy_date'].date()} at ${trade['buy_price']:.2f} -> "
                      f"Sell: {trade['sell_date'].date()} at ${trade['sell_price']:.2f} "
                      f"({trade['profit_pct']:+.2f}%, {trade['hold_days']} days)")
    
    # Comparison with DCA
    print()
    print("=" * 100)
    print("COMPARISON WITH DCA STRATEGIES")
    print("=" * 100)
    print()
    
    # Calculate what buy and hold would give with monthly DCA
    monthly_df = df.copy()
    monthly_df['year_month'] = monthly_df['Date'].dt.to_period('M')
    monthly_first = monthly_df.groupby('year_month').first().reset_index()
    
    monthly_investment = 500  # $500/month
    dca_shares = 0
    dca_invested = 0
    
    for _, row in monthly_first.iterrows():
        dca_shares += monthly_investment / row['Close']
        dca_invested += monthly_investment
    
    dca_final_value = dca_shares * df.iloc[-1]['Close']
    dca_return = ((dca_final_value - dca_invested) / dca_invested) * 100
    years = (df['Date'].max() - df['Date'].min()).days / 365.25
    dca_annualized = (((dca_final_value / dca_invested) ** (1 / years)) - 1) * 100
    
    print(f"DCA ($500/month):")
    print(f"   Total Invested:    ${dca_invested:,.0f}")
    print(f"   Final Value:       ${dca_final_value:,.0f}")
    print(f"   Total Return:      {dca_return:.2f}%")
    print(f"   Annualized Return: {dca_annualized:.2f}%")
    
    best_strategy = results_df.iloc[0]
    print()
    print(f"Best Trading Strategy ({best_strategy['strategy']}):")
    print(f"   Initial Investment: ${best_strategy['initial_capital']:,.0f} (lump sum)")
    print(f"   Final Value:        ${best_strategy['final_value']:,.0f}")
    print(f"   Total Return:       {best_strategy['total_return']:.2f}%")
    print(f"   Annualized Return:  {best_strategy['annualized_return']:.2f}%")
    
    print()
    print("=" * 100)
    print("IMPORTANT NOTES:")
    print("=" * 100)
    print("1. Past performance does not guarantee future results")
    print("2. Trading involves:")
    print("   - Tax implications (short-term vs long-term capital gains)")
    print("   - Transaction costs (even if commissions are $0, there's bid-ask spread)")
    print("   - Emotional discipline required")
    print("   - Time commitment to monitor positions")
    print("3. These backtests assume:")
    print("   - Perfect execution at close prices")
    print("   - No slippage")
    print("   - No transaction costs")
    print("   - No taxes")
    print("4. Real-world returns will be lower due to above factors")
    print("5. Strategies that traded more frequently will be more affected by costs")
    print("=" * 100)
    
    # Save results
    results_df.to_csv('sp500_trading_results.csv', index=False)
    print("\nResults saved to: sp500_trading_results.csv")

if __name__ == "__main__":
    main()
