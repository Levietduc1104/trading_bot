import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Calculate various technical indicators"""

    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD indicator"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def stochastic(high, low, close, period=14):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        return k, d

class DCABacktester:
    """Backtest DCA strategy with technical indicators"""

    def __init__(self, data, monthly_investment=1000):
        self.data = data
        self.monthly_investment = monthly_investment

    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        df = self.data.copy()

        # Moving Averages
        df['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema_50'] = TechnicalIndicators.ema(df['close'], 50)
        df['ema_200'] = TechnicalIndicators.ema(df['close'], 200)
        df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        df['sma_200'] = TechnicalIndicators.sma(df['close'], 200)

        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], 14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalIndicators.macd(df['close'])

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Stochastic
        df['stoch_k'], df['stoch_d'] = TechnicalIndicators.stochastic(
            df['high'], df['low'], df['close']
        )

        # Price vs Moving Averages
        df['price_above_ema50'] = (df['close'] > df['ema_50']).astype(int)
        df['price_above_ema200'] = (df['close'] > df['ema_200']).astype(int)
        df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        df['price_above_sma200'] = (df['close'] > df['sma_200']).astype(int)

        # Moving Average Crossovers
        df['ema_50_above_200'] = (df['ema_50'] > df['ema_200']).astype(int)
        df['sma_50_above_200'] = (df['sma_50'] > df['sma_200']).astype(int)

        return df

    def backtest_strategy(self, df, condition, name="Strategy"):
        """Backtest a specific strategy condition"""
        # Create monthly investment points (first trading day of each month)
        df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_days = df.groupby('year_month').first().index

        portfolio_value = 0
        shares = 0
        cash_invested = 0
        trades = []

        for period in monthly_days:
            # Find the first trading day of this month
            mask = df['year_month'] == period
            if not mask.any():
                continue

            first_day_idx = df[mask].index[0]

            if first_day_idx >= len(df):
                continue

            # Check if condition is met (buy signal)
            if condition(df.iloc[first_day_idx]):
                price = df.iloc[first_day_idx]['close']
                shares_to_buy = self.monthly_investment / price
                shares += shares_to_buy
                cash_invested += self.monthly_investment

                trades.append({
                    'date': df.iloc[first_day_idx]['date'],
                    'price': price,
                    'shares': shares_to_buy,
                    'invested': self.monthly_investment,
                    'total_shares': shares,
                    'total_invested': cash_invested
                })

        # Calculate final portfolio value
        if len(trades) > 0:
            final_price = df.iloc[-1]['close']
            portfolio_value = shares * final_price
            total_return = ((portfolio_value - cash_invested) / cash_invested * 100) if cash_invested > 0 else 0

            # Calculate additional metrics
            trade_df = pd.DataFrame(trades)
            trade_df['date'] = pd.to_datetime(trade_df['date'])

            # Time-weighted returns for Sharpe calculation
            trade_df['value'] = trade_df['total_shares'] * df.set_index('date')['close'].reindex(trade_df['date'].values).values
            trade_df['returns'] = trade_df['value'].pct_change()

            sharpe = (trade_df['returns'].mean() / trade_df['returns'].std() * np.sqrt(12)) if trade_df['returns'].std() > 0 else 0

            # Maximum drawdown
            trade_df['cummax'] = trade_df['value'].cummax()
            trade_df['drawdown'] = (trade_df['value'] - trade_df['cummax']) / trade_df['cummax']
            max_drawdown = trade_df['drawdown'].min() * 100

            return {
                'name': name,
                'trades': len(trades),
                'total_invested': cash_invested,
                'final_value': portfolio_value,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_drawdown,
                'avg_buy_price': cash_invested / shares if shares > 0 else 0,
                'final_price': final_price
            }
        else:
            return {
                'name': name,
                'trades': 0,
                'total_invested': 0,
                'final_value': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'avg_buy_price': 0,
                'final_price': df.iloc[-1]['close']
            }

    def test_strategy_combinations(self, df):
        """Test various indicator combinations"""
        strategies = []

        # 1. Buy and Hold (baseline - always buy)
        strategies.append({
            'name': 'Buy and Hold (Always)',
            'condition': lambda row: True
        })

        # 2. RSI-based strategies
        strategies.append({
            'name': 'RSI < 30 (Oversold)',
            'condition': lambda row: row['rsi'] < 30
        })
        strategies.append({
            'name': 'RSI < 40',
            'condition': lambda row: row['rsi'] < 40
        })
        strategies.append({
            'name': 'RSI < 50',
            'condition': lambda row: row['rsi'] < 50
        })

        # 3. Price vs EMA strategies
        strategies.append({
            'name': 'Price < EMA50',
            'condition': lambda row: row['close'] < row['ema_50']
        })
        strategies.append({
            'name': 'Price < EMA200',
            'condition': lambda row: row['close'] < row['ema_200']
        })

        # 4. Price vs SMA strategies
        strategies.append({
            'name': 'Price < SMA50',
            'condition': lambda row: row['close'] < row['sma_50']
        })
        strategies.append({
            'name': 'Price < SMA200',
            'condition': lambda row: row['close'] < row['sma_200']
        })

        # 5. Bollinger Bands strategies
        strategies.append({
            'name': 'Price < BB Lower',
            'condition': lambda row: row['close'] < row['bb_lower']
        })
        strategies.append({
            'name': 'BB Position < 0.3',
            'condition': lambda row: row['bb_position'] < 0.3
        })

        # 6. MACD strategies
        strategies.append({
            'name': 'MACD < Signal (Bearish)',
            'condition': lambda row: row['macd'] < row['macd_signal']
        })

        # 7. Stochastic strategies
        strategies.append({
            'name': 'Stochastic < 20',
            'condition': lambda row: row['stoch_k'] < 20
        })

        # 8. Combined strategies (multi-indicator)
        strategies.append({
            'name': 'RSI<50 AND Price<EMA50',
            'condition': lambda row: (row['rsi'] < 50) and (row['close'] < row['ema_50'])
        })
        strategies.append({
            'name': 'RSI<40 AND Price<SMA50',
            'condition': lambda row: (row['rsi'] < 40) and (row['close'] < row['sma_50'])
        })
        strategies.append({
            'name': 'RSI<50 AND BB_Pos<0.5',
            'condition': lambda row: (row['rsi'] < 50) and (row['bb_position'] < 0.5)
        })
        strategies.append({
            'name': 'Price<EMA50 AND MACD<Signal',
            'condition': lambda row: (row['close'] < row['ema_50']) and (row['macd'] < row['macd_signal'])
        })
        strategies.append({
            'name': 'RSI<40 AND Stoch<30',
            'condition': lambda row: (row['rsi'] < 40) and (row['stoch_k'] < 30)
        })
        strategies.append({
            'name': 'Price<SMA200 AND RSI<50',
            'condition': lambda row: (row['close'] < row['sma_200']) and (row['rsi'] < 50)
        })

        # 9. Triple indicator combinations
        strategies.append({
            'name': 'RSI<50 AND Price<EMA50 AND BB_Pos<0.5',
            'condition': lambda row: (row['rsi'] < 50) and (row['close'] < row['ema_50']) and (row['bb_position'] < 0.5)
        })
        strategies.append({
            'name': 'RSI<40 AND Price<SMA50 AND Stoch<30',
            'condition': lambda row: (row['rsi'] < 40) and (row['close'] < row['sma_50']) and (row['stoch_k'] < 30)
        })

        return strategies

def main():
    print("=" * 80)
    print("S&P 500 DCA STRATEGY OPTIMIZER")
    print("Testing indicator combinations to maximize long-term returns")
    print("=" * 80)
    print()

    # Load data
    print("Loading S&P 500 (SPY) data...")
    df = pd.read_csv('trading_bot/sp500_data/daily/SPY.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total trading days: {len(df)}")
    print()

    # Split data into train and test to avoid overfitting
    split_date = '2018-01-01'
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    print(f"Training period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Testing period: {test_df['date'].min()} to {test_df['date'].max()}")
    print()

    # Initialize backtester
    backtester = DCABacktester(df, monthly_investment=1000)

    # Calculate indicators for both periods
    print("Calculating technical indicators...")
    train_df = backtester.calculate_all_indicators()
    train_df = train_df[train_df['date'] < split_date].copy()

    test_df_full = backtester.calculate_all_indicators()
    test_df = test_df_full[test_df_full['date'] >= split_date].copy()
    print("Done\!")
    print()

    # Get all strategies
    strategies = backtester.test_strategy_combinations(train_df)

    # Test on training data
    print("=" * 80)
    print("TRAINING PERIOD RESULTS (2005-2017)")
    print("=" * 80)
    train_results = []

    for strategy in strategies:
        result = backtester.backtest_strategy(train_df, strategy['condition'], strategy['name'])
        train_results.append(result)

    train_results_df = pd.DataFrame(train_results)
    train_results_df = train_results_df.sort_values('total_return_pct', ascending=False)

    print("\nTop 10 Strategies (by Total Return):")
    print("-" * 80)
    for idx, row in train_results_df.head(10).iterrows():
        print(f"{row['name'][:40]:40} | Return: {row['total_return_pct']:6.2f}% | "
              f"Trades: {row['trades']:3} | Sharpe: {row['sharpe_ratio']:5.2f} | "
              f"MaxDD: {row['max_drawdown_pct']:6.2f}%")

    # Test on testing data
    print()
    print("=" * 80)
    print("TESTING PERIOD RESULTS (2018-2025)")
    print("=" * 80)
    test_results = []

    for strategy in strategies:
        result = backtester.backtest_strategy(test_df, strategy['condition'], strategy['name'])
        test_results.append(result)

    test_results_df = pd.DataFrame(test_results)
    test_results_df = test_results_df.sort_values('total_return_pct', ascending=False)

    print("\nTop 10 Strategies (by Total Return):")
    print("-" * 80)
    for idx, row in test_results_df.head(10).iterrows():
        print(f"{row['name'][:40]:40} | Return: {row['total_return_pct']:6.2f}% | "
              f"Trades: {row['trades']:3} | Sharpe: {row['sharpe_ratio']:5.2f} | "
              f"MaxDD: {row['max_drawdown_pct']:6.2f}%")

    # Full period analysis
    print()
    print("=" * 80)
    print("FULL PERIOD RESULTS (2005-2025)")
    print("=" * 80)
    full_results = []

    for strategy in strategies:
        result = backtester.backtest_strategy(
            backtester.calculate_all_indicators(),
            strategy['condition'],
            strategy['name']
        )
        full_results.append(result)

    full_results_df = pd.DataFrame(full_results)
    full_results_df = full_results_df.sort_values('total_return_pct', ascending=False)

    print("\nAll Strategies Ranked by Total Return:")
    print("-" * 80)
    print(f"{'Strategy':<40} | {'Return':<8} | {'Trades':<6} | {'Sharpe':<7} | {'MaxDD':<8} | {'Invested':<12} | {'Final Value':<12}")
    print("-" * 80)
    for idx, row in full_results_df.iterrows():
        print(f"{row['name']:<40} | {row['total_return_pct']:>6.2f}% | {row['trades']:>6} | "
              f"{row['sharpe_ratio']:>6.2f} | {row['max_drawdown_pct']:>6.2f}% | "
              f"${row['total_invested']:>11,.0f} | ${row['final_value']:>11,.0f}")

    # Summary and recommendations
    print()
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    baseline = full_results_df[full_results_df['name'] == 'Buy and Hold (Always)'].iloc[0]
    print(f"\nBaseline (Buy and Hold):")
    print(f"  Total Invested: ${baseline['total_invested']:,.0f}")
    print(f"  Final Value: ${baseline['final_value']:,.0f}")
    print(f"  Total Return: {baseline['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {baseline['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {baseline['max_drawdown_pct']:.2f}%")

    best_strategy = full_results_df.iloc[0]
    print(f"\nBest Strategy: {best_strategy['name']}")
    print(f"  Total Invested: ${best_strategy['total_invested']:,.0f}")
    print(f"  Final Value: ${best_strategy['final_value']:,.0f}")
    print(f"  Total Return: {best_strategy['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {best_strategy['max_drawdown_pct']:.2f}%")
    print(f"  Number of Purchases: {best_strategy['trades']}")

    if best_strategy['total_return_pct'] > baseline['total_return_pct']:
        improvement = best_strategy['final_value'] - baseline['final_value']
        print(f"\n  Improvement over baseline: ${improvement:,.0f} ({(improvement/baseline['final_value']*100):.2f}%)")

    print("\n" + "=" * 80)
    print("IMPORTANT NOTES:")
    print("=" * 80)
    print("1. These results are based on historical data and past performance doesn't")
    print("   guarantee future results.")
    print("2. The analysis uses train/test split to reduce overfitting.")
    print("3. Consider strategies that perform well in BOTH training and testing periods.")
    print("4. Lower number of trades may miss some opportunities but reduces timing risk.")
    print("5. For long-term DCA, consistency is often more important than optimization.")
    print("6. Buy and Hold is hard to beat for tax efficiency and simplicity.")
    print("=" * 80)

    # Save detailed results
    full_results_df.to_csv('sp500_dca_results.csv', index=False)
    print("\nDetailed results saved to: sp500_dca_results.csv")

if __name__ == "__main__":
    main()
