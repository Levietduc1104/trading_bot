"""
S&P 500 Portfolio Rotation Bot - Complete Demo

This demonstrates the end-to-end portfolio rotation system:
1. Load stock data
2. Calculate technical indicators
3. Score and rank stocks
4. Allocate portfolio to top N stocks
5. Simulate monthly rebalancing
6. Calculate performance metrics

Goal: 20%+ annual return with low drawdown
"""

import pandas as pd
import numpy as np
import glob
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PortfolioRotationBot:
    """Complete portfolio rotation system"""

    def __init__(self, data_dir='sp500_data/daily', initial_capital=100000):
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.stocks_data = {}
        self.scores = {}
        self.rankings = []

    # ========================
    # 1. DATA LOADING
    # ========================

    def load_all_stocks(self):
        """Load all stock data"""
        csv_files = glob.glob(f"{self.data_dir}/*.csv")
        logger.info(f"Loading {len(csv_files)} stocks...")

        for file_path in csv_files:
            ticker = os.path.basename(file_path).replace('.csv', '')
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = [col.lower() for col in df.columns]
                self.stocks_data[ticker] = df
            except Exception as e:
                logger.warning(f"Error loading {ticker}: {e}")

        logger.info(f"Loaded {len(self.stocks_data)} stocks")

    # ========================
    # 2. TECHNICAL INDICATORS
    # ========================

    def add_indicators(self, df):
        """Add all technical indicators"""
        df = df.copy()

        # EMAs
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        df['ema_34'] = df['close'].ewm(span=34, adjust=False).mean()
        df['ema_89'] = df['close'].ewm(span=89, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ROC
        df['roc_20'] = df['close'].pct_change(20) * 100

        # EMA alignment
        df['ema_alignment'] = ((df['ema_13'] > df['ema_34']).astype(int) +
                               (df['ema_34'] > df['ema_89']).astype(int))

        return df

    def prepare_data(self):
        """Load and add indicators to all stocks"""
        self.load_all_stocks()
        logger.info("Calculating indicators...")

        for ticker in self.stocks_data.keys():
            self.stocks_data[ticker] = self.add_indicators(self.stocks_data[ticker])

        logger.info("Indicators calculated")

    # ========================
    # 3. STOCK SCORING (0-100)
    # ========================

    def score_stock(self, ticker, df):
        """Score a stock 0-100"""
        latest = df.iloc[-1]
        recent = df.tail(60)

        score = 0

        # Momentum (25 pts)
        rsi = latest['rsi']
        if 40 <= rsi <= 60:
            score += 5
        elif 30 <= rsi <= 70:
            score += 3

        roc = latest['roc_20']
        if roc > 10:
            score += 10
        elif roc > 5:
            score += 7
        elif roc > 0:
            score += 4

        if latest['ema_alignment'] == 2:
            score += 10
        elif latest['ema_alignment'] == 1:
            score += 5

        # Trend (25 pts)
        if latest['ema_alignment'] == 2 and roc > 0:
            score += 25
        elif latest['ema_alignment'] >= 1:
            score += 15

        # Volatility (20 pts) - lower is better
        atr_pct = (latest['atr'] / latest['close']) * 100
        if atr_pct < 2:
            score += 20
        elif atr_pct < 4:
            score += 10

        # Risk/Reward (30 pts)
        returns_60d = (latest['close'] / recent['close'].iloc[0] - 1) * 100
        if returns_60d > 15:
            score += 30
        elif returns_60d > 10:
            score += 20
        elif returns_60d > 5:
            score += 15
        elif returns_60d > 0:
            score += 10

        return min(score, 100)

    def score_all_stocks(self):
        """Score all stocks"""
        logger.info("Scoring stocks...")
        self.scores = {}

        for ticker, df in self.stocks_data.items():
            if len(df) >= 100:
                self.scores[ticker] = self.score_stock(ticker, df)

        self.rankings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Scored {len(self.scores)} stocks")

    # ========================
    # 4. PORTFOLIO BACKTEST (FIXED)
    # ========================

    def backtest(self, top_n=10, rebalance_freq='M'):
        """
        Backtest portfolio rotation strategy

        Args:
            top_n: Number of stocks to hold
            rebalance_freq: 'M' for monthly
        """
        logger.info(f"Starting backtest: Top {top_n} stocks, Monthly rebalance")

        # Get common dates across all stocks
        first_ticker = list(self.stocks_data.keys())[0]
        dates = self.stocks_data[first_ticker].index

        # Track portfolio
        portfolio_values = []
        cash = self.initial_capital
        holdings = {}  # {ticker: shares}
        last_rebalance_month = None

        for date in dates[100:]:  # Skip first 100 days for indicators
            current_month = date.month

            # Monthly rebalance
            if current_month != last_rebalance_month:
                last_rebalance_month = current_month

                # Liquidate current holdings
                for ticker in list(holdings.keys()):
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        cash += holdings[ticker] * current_price
                holdings = {}

                # Score all stocks at this date
                current_scores = {}
                for ticker, df in self.stocks_data.items():
                    df_at_date = df[df.index <= date]
                    if len(df_at_date) >= 100:
                        try:
                            current_scores[ticker] = self.score_stock(ticker, df_at_date)
                        except:
                            pass

                # Get top N stocks
                ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
                top_stocks = [t for t, s in ranked[:top_n]]

                # Allocate capital (invest 80%, keep 20% cash)
                invest_amount = cash * 0.80
                allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0

                # Buy top stocks
                for ticker in top_stocks:
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        shares = allocation_per_stock / current_price
                        holdings[ticker] = shares
                        cash -= allocation_per_stock

            # Calculate daily portfolio value
            stocks_value = 0
            for ticker, shares in holdings.items():
                df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    stocks_value += shares * current_price

            total_value = cash + stocks_value
            portfolio_values.append({'date': date, 'value': total_value})

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100

        # Max drawdown
        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        logger.info("="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:.1f}%")
        logger.info(f"Annual Return: {annual_return:.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Period: {portfolio_df.index[0].date()} to {portfolio_df.index[-1].date()}")
        logger.info(f"Total Days: {len(portfolio_df)}")
        logger.info("="*60)

        # Goal check
        if annual_return >= 20:
            logger.info(f"🎯 GOAL ACHIEVED\! {annual_return:.1f}% exceeds 20% target")
        else:
            logger.info(f"⚠️  Below target: {annual_return:.1f}% vs 20% goal")

        return portfolio_df

    # ========================
    # 5. MAIN EXECUTION
    # ========================

    def run(self):
        """Run complete end-to-end demo"""
        logger.info("="*60)
        logger.info("S&P 500 PORTFOLIO ROTATION BOT")
        logger.info("Goal: 20%+ Annual Return")
        logger.info("="*60)

        # 1. Load data
        self.prepare_data()

        # 2. Score stocks
        self.score_all_stocks()

        # 3. Show rankings
        logger.info("\nTOP 10 STOCKS (Current Rankings):")
        for i, (ticker, score) in enumerate(self.rankings[:10], 1):
            logger.info(f"  {i}. {ticker}: {score:.1f} points")

        logger.info("\nBOTTOM 5 STOCKS:")
        for i, (ticker, score) in enumerate(self.rankings[-5:], 1):
            logger.info(f"  {ticker}: {score:.1f} points")

        # 4. Backtest
        logger.info("\n" + "="*60)
        logger.info("Running Portfolio Backtest...")
        logger.info("="*60)
        portfolio_df = self.backtest(top_n=10)

        return portfolio_df




# ========================
# DATA GENERATION FOR 500 S&P 500 STOCKS
# ========================

def generate_sp500_stocks_data(output_dir='sp500_data/daily', num_stocks=500):
    """Generate realistic data for S&P 500 stocks across all sectors"""
    
    # Comprehensive S&P 500 stock list
    SP500_STOCKS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
                       'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'IBM', 'QCOM', 'TXN', 'NOW', 'INTU',
                       'AMAT', 'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'NXPI', 'FTNT',
                       'ADSK', 'ANSS', 'ROP', 'KEYS', 'HPQ', 'HPE', 'DELL', 'NTAP', 'STX', 'WDC',
                       'TRMB', 'ZBRA', 'FFIV', 'JNPR', 'AKAM', 'CTSH', 'GLW', 'APH', 'TEL', 'ANET',
                       'PANW', 'CRWD', 'ZS', 'DDOG', 'NET', 'OKTA', 'SNOW', 'PLTR', 'RBLX', 'TWLO',
                       'ZM', 'DOCU', 'WDAY', 'VEEV', 'SPLK', 'HUBS', 'TEAM', 'MDB', 'ESTC', 'BILL'],
        'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'AMGN',
                       'BMY', 'GILD', 'CVS', 'VRTX', 'CI', 'REGN', 'HUM', 'MCK', 'ELV', 'COR',
                       'ZTS', 'ISRG', 'BDX', 'SYK', 'BSX', 'EW', 'MDT', 'IDXX', 'DXCM', 'A',
                       'IQV', 'RMD', 'ALGN', 'HOLX', 'PODD', 'TECH', 'RVTY', 'WAT', 'MTD', 'PKI',
                       'ILMN', 'BIO', 'MRNA', 'BIIB', 'VTRS', 'HCA', 'UHS', 'DVA', 'DGX', 'LH',
                       'CNC', 'MOH', 'CAH', 'HSIC', 'COO', 'BAX', 'XRAY', 'INCY', 'EXAS', 'JAZZ'],
        'Financial': ['BRKB', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'C', 'AXP',
                      'SPGI', 'BLK', 'CB', 'SCHW', 'PGR', 'MMC', 'USB', 'PNC', 'TFC', 'AON',
                      'CME', 'ICE', 'MCO', 'COF', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV',
                      'HIG', 'WTW', 'AJG', 'MSCI', 'BK', 'STT', 'NTRS', 'RF', 'CFG', 'KEY',
                      'FITB', 'HBAN', 'MTB', 'DFS', 'SYF', 'TROW', 'BEN', 'IVZ', 'NDAQ', 'CBOE',
                      'FDS', 'MKTX', 'GL', 'PFG', 'CINF', 'RJF', 'ZION', 'CMA', 'ALLY', 'WRB',
                      'FNF', 'FAF', 'AIZ', 'AMP', 'RGA', 'LNC', 'ORI', 'BRO', 'ACGL', 'AFG'],
        'Consumer Discretionary': ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'ABNB',
                                    'MAR', 'GM', 'F', 'DHI', 'LEN', 'YUM', 'ORLY', 'AZO', 'ROST', 'RCL',
                                    'CCL', 'NCLH', 'HLT', 'MGM', 'WYNN', 'LVS', 'DRI', 'ULTA', 'DPZ', 'POOL',
                                    'BBY', 'GPC', 'AAP', 'TSCO', 'DG', 'DLTR', 'EBAY', 'ETSY', 'CPRI', 'RL',
                                    'TPR', 'UAA', 'NVR', 'PHM', 'TOL', 'KBH', 'MTH', 'DIS', 'NFLX', 'CHTR'],
        'Communication Services': ['META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'EA',
                                    'ATVI', 'TTWO', 'MTCH', 'PINS', 'SNAP', 'ROKU', 'LUMN', 'SIRI',
                                    'NWSA', 'NWS', 'NYT', 'OMC', 'IPG', 'FOX', 'FOXA', 'PARA', 'LYV'],
        'Industrials': ['UPS', 'BA', 'HON', 'UNP', 'RTX', 'LMT', 'CAT', 'DE', 'GE', 'MMM',
                        'NSC', 'CSX', 'FDX', 'WM', 'GD', 'NOC', 'ITW', 'EMR', 'ETN', 'PH',
                        'CARR', 'OTIS', 'JCI', 'CMI', 'PCAR', 'ROK', 'FAST', 'ODFL', 'VRSK', 'IEX',
                        'DOV', 'IR', 'SWK', 'CPRT', 'XYL', 'FTV', 'GNRC', 'WAB', 'AOS', 'LDOS',
                        'TXT', 'HWM', 'ALLE', 'CHRW', 'JBHT', 'EXPD', 'URI', 'RSG', 'PWR', 'BLDR',
                        'DAL', 'UAL', 'AAL', 'LUV', 'ALK', 'PNR', 'MLM', 'VMC', 'TT', 'HUBB'],
        'Consumer Staples': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'GIS',
                             'KMB', 'STZ', 'SYY', 'KHC', 'HSY', 'K', 'CAG', 'CPB', 'HRL', 'SJM',
                             'MKC', 'TAP', 'TSN', 'CHD', 'CLX', 'KR', 'EL', 'ADM', 'BG', 'LW'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'PSX', 'VLO', 'OXY',
                   'WMB', 'KMI', 'HAL', 'DVN', 'HES', 'FANG', 'BKR', 'MRO', 'APA', 'OKE',
                   'TRGP', 'EQT', 'CTRA', 'NOV', 'FTI'],
        'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'DOW', 'PPG',
                      'CTVA', 'ALB', 'EMN', 'IFF', 'MOS', 'FMC', 'CF', 'CE', 'VMC', 'MLM',
                      'BALL', 'AVY', 'AMCR', 'PKG', 'IP', 'SEE', 'WRK', 'CCK', 'HUN', 'SLGN'],
        'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR', 'O', 'SBAC',
                        'AVB', 'EQR', 'VTR', 'INVH', 'ARE', 'MAA', 'EXR', 'UDR', 'CPT', 'ESS',
                        'KIM', 'REG', 'FRT', 'BXP', 'VNO', 'SLG', 'HST', 'PEAK', 'CBRE', 'IRM'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED',
                      'PEG', 'ES', 'EIX', 'DTE', 'PPL', 'AWK', 'FE', 'AEE', 'CMS', 'CNP',
                      'ATO', 'NI', 'LNT', 'EVRG', 'PNW']
    }
    
    # Sector characteristics
    SECTOR_CONFIGS = {
        'Technology': {'annual_return': 0.25, 'volatility': 0.35, 'start_price_range': (50, 300)},
        'Healthcare': {'annual_return': 0.18, 'volatility': 0.28, 'start_price_range': (60, 250)},
        'Financial': {'annual_return': 0.15, 'volatility': 0.30, 'start_price_range': (30, 150)},
        'Consumer Discretionary': {'annual_return': 0.20, 'volatility': 0.32, 'start_price_range': (40, 200)},
        'Communication Services': {'annual_return': 0.22, 'volatility': 0.33, 'start_price_range': (50, 250)},
        'Industrials': {'annual_return': 0.14, 'volatility': 0.25, 'start_price_range': (50, 180)},
        'Consumer Staples': {'annual_return': 0.12, 'volatility': 0.18, 'start_price_range': (40, 120)},
        'Energy': {'annual_return': 0.10, 'volatility': 0.40, 'start_price_range': (30, 100)},
        'Materials': {'annual_return': 0.13, 'volatility': 0.28, 'start_price_range': (40, 150)},
        'Real Estate': {'annual_return': 0.11, 'volatility': 0.22, 'start_price_range': (30, 140)},
        'Utilities': {'annual_return': 0.09, 'volatility': 0.16, 'start_price_range': (40, 100)}
    }
    
    # Flatten and map
    all_tickers = []
    ticker_sector_map = {}
    for sector, tickers in SP500_STOCKS.items():
        all_tickers.extend(tickers)
        for ticker in tickers:
            ticker_sector_map[ticker] = sector
    
    all_tickers = list(dict.fromkeys(all_tickers))[:num_stocks]
    
    logger.info("="*80)
    logger.info(f"GENERATING {len(all_tickers)} S&P 500 STOCKS")
    logger.info("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    days = 1764
    
    for i, ticker in enumerate(all_tickers, 1):
        sector = ticker_sector_map.get(ticker, 'Industrials')
        config = SECTOR_CONFIGS[sector]
        
        start_price = np.random.uniform(*config['start_price_range'])
        annual_return = config['annual_return'] + np.random.uniform(-0.05, 0.05)
        volatility = config['volatility'] + np.random.uniform(-0.05, 0.05)
        
        daily_drift = annual_return / 252
        daily_volatility = volatility / np.sqrt(252)
        
        returns = np.random.normal(daily_drift, daily_volatility, days)
        returns[530:550] = np.random.normal(-0.10, 0.05, 20)  # COVID
        returns[1010:1130] = np.random.normal(-0.008, 0.03, 120)  # 2022 bear
        
        close_prices = start_price * np.cumprod(np.exp(returns))
        dates = pd.date_range(start='2018-01-01', periods=days, freq='B')
        
        df = pd.DataFrame({
            'date': dates,
            'open': close_prices * np.random.uniform(0.98, 1.02, days),
            'high': close_prices * np.random.uniform(1.00, 1.05, days),
            'low': close_prices * np.random.uniform(0.95, 1.00, days),
            'close': close_prices,
            'volume': np.random.randint(1_000_000, 50_000_000, days)
        })
        
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        df.to_csv(f"{output_dir}/{ticker}.csv", index=False)
        successful += 1
        
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(all_tickers)} ({i/len(all_tickers)*100:.1f}%)")
    
    logger.info(f"\n✅ Generated {successful} stocks in {output_dir}/")
    return successful



# Main execution


# ========================
# ANALYSIS & DIAGNOSTICS
# ========================

def analyze_performance(data_dir='sp500_data/daily'):
    """Detailed analysis of why performance changed"""
    
    logger.info("="*80)
    logger.info("PERFORMANCE ANALYSIS - 470 STOCKS")
    logger.info("="*80)
    
    # 1. Load and score all stocks
    bot = PortfolioRotationBot(data_dir=data_dir)
    bot.prepare_data()
    bot.score_all_stocks()
    
    # 2. Analyze score distribution
    logger.info("\n1. SCORE DISTRIBUTION:")
    logger.info("-" * 80)
    scores = [score for _, score in bot.rankings]
    logger.info(f"   Total stocks scored: {len(scores)}")
    logger.info(f"   Average score: {np.mean(scores):.1f}")
    logger.info(f"   Median score: {np.median(scores):.1f}")
    logger.info(f"   Std deviation: {np.std(scores):.1f}")
    logger.info(f"   Min score: {np.min(scores):.1f}")
    logger.info(f"   Max score: {np.max(scores):.1f}")
    
    # Score distribution by range
    score_ranges = {
        '80-100 (Excellent)': sum(1 for s in scores if s >= 80),
        '60-79 (Good)': sum(1 for s in scores if 60 <= s < 80),
        '40-59 (Average)': sum(1 for s in scores if 40 <= s < 60),
        '20-39 (Poor)': sum(1 for s in scores if 20 <= s < 40),
        '0-19 (Very Poor)': sum(1 for s in scores if s < 20)
    }
    
    logger.info("\n   Score Distribution:")
    for range_name, count in score_ranges.items():
        pct = count / len(scores) * 100
        logger.info(f"   {range_name:25s}: {count:3d} stocks ({pct:5.1f}%)")
    
    # 3. Analyze top vs bottom performers
    logger.info("\n2. TOP 10 vs BOTTOM 10 ACTUAL RETURNS:")
    logger.info("-" * 80)
    
    top_10_tickers = [ticker for ticker, _ in bot.rankings[:10]]
    bottom_10_tickers = [ticker for ticker, _ in bot.rankings[-10:]]
    
    def calculate_actual_return(ticker):
        """Calculate actual return for a stock"""
        if ticker in bot.stocks_data:
            df = bot.stocks_data[ticker]
            start_price = df['close'].iloc[100]  # Start after warmup
            end_price = df['close'].iloc[-1]
            return ((end_price / start_price) - 1) * 100
        return 0
    
    logger.info("\n   Top 10 Ranked Stocks:")
    top_returns = []
    for i, ticker in enumerate(top_10_tickers, 1):
        score = dict(bot.rankings)[ticker]
        actual_return = calculate_actual_return(ticker)
        top_returns.append(actual_return)
        logger.info(f"   {i:2d}. {ticker:6s} - Score: {score:5.1f} | Actual Return: {actual_return:7.1f}%")
    
    logger.info("\n   Bottom 10 Ranked Stocks:")
    bottom_returns = []
    for i, ticker in enumerate(bottom_10_tickers, 1):
        score = dict(bot.rankings)[ticker]
        actual_return = calculate_actual_return(ticker)
        bottom_returns.append(actual_return)
        logger.info(f"   {i:2d}. {ticker:6s} - Score: {score:5.1f} | Actual Return: {actual_return:7.1f}%")
    
    logger.info(f"\n   Average Return - Top 10: {np.mean(top_returns):.1f}%")
    logger.info(f"   Average Return - Bottom 10: {np.mean(bottom_returns):.1f}%")
    logger.info(f"   Difference: {np.mean(top_returns) - np.mean(bottom_returns):.1f}%")
    
    # 4. Check data quality
    logger.info("\n3. DATA QUALITY CHECK:")
    logger.info("-" * 80)
    
    # Sample 10 random stocks
    sample_tickers = np.random.choice(list(bot.stocks_data.keys()), 10, replace=False)
    logger.info("   Checking 10 random stocks:")
    
    for ticker in sample_tickers:
        df = bot.stocks_data[ticker]
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        total_return = ((end_price / start_price) - 1) * 100
        volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
        
        logger.info(f"   {ticker:6s}: Start=${start_price:7.2f} End=${end_price:7.2f} Return={total_return:7.1f}% Vol={volatility:.1f}%")
    
    # 5. Test different portfolio sizes
    logger.info("\n4. TESTING DIFFERENT PORTFOLIO SIZES:")
    logger.info("-" * 80)
    
    for top_n in [5, 10, 20, 30, 50]:
        portfolio_df = bot.backtest(top_n=top_n)
        
        initial = bot.initial_capital
        final = portfolio_df['value'].iloc[-1]
        total_return = (final / initial - 1) * 100
        
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final / initial) ** (1/years) - 1) * 100
        
        cummax = portfolio_df['value'].cummax()
        drawdown = ((portfolio_df['value'] - cummax) / cummax * 100).min()
        
        logger.info(f"   Top {top_n:2d} stocks: Annual Return={annual_return:6.1f}% | Max DD={drawdown:6.1f}% | Final=${final:,.0f}")
    
    # 6. Sector analysis
    logger.info("\n5. SECTOR REPRESENTATION IN TOP 50:")
    logger.info("-" * 80)
    
    # Need to track sectors
    from collections import Counter
    top_50_tickers = [ticker for ticker, _ in bot.rankings[:50]]
    
    # Read a few stocks to see data
    logger.info(f"   Top 50 stocks: {', '.join(top_50_tickers[:20])}...")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)



# ========================
# MAIN EXECUTION
# ========================



# ========================
# STRATEGY OPTIMIZATION FOR WORST-CASE SCENARIO
# ========================

def test_strategies_worst_case():
    """Test multiple strategies to find what works in worst-case market"""
    
    logger.info("="*80)
    logger.info("WORST-CASE STRATEGY OPTIMIZATION")
    logger.info("Market Condition: 90% of stocks lose 70-95% of value")
    logger.info("="*80)
    
    bot = PortfolioRotationBot()
    bot.prepare_data()
    bot.score_all_stocks()
    
    results = []
    
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 1: CASH RESERVE VARIATIONS")
    logger.info("Testing higher cash reserves to reduce drawdown")
    logger.info("="*80)
    
    for cash_pct in [0.20, 0.30, 0.40, 0.50, 0.60]:
        logger.info(f"\nTesting {int(cash_pct*100)}% cash reserve...")
        portfolio_df = bot.backtest_with_cash(top_n=20, cash_reserve=cash_pct)
        
        initial = bot.initial_capital
        final = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final / initial) ** (1/years) - 1) * 100
        
        cummax = portfolio_df['value'].cummax()
        drawdown = ((portfolio_df['value'] - cummax) / cummax * 100).min()
        
        results.append({
            'strategy': f'Cash {int(cash_pct*100)}%',
            'annual_return': annual_return,
            'max_drawdown': drawdown,
            'final_value': final
        })
        
        logger.info(f"  → Annual: {annual_return:6.1f}% | Max DD: {drawdown:6.1f}% | Final: ${final:,.0f}")
    
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 2: PORTFOLIO SIZE OPTIMIZATION")
    logger.info("Finding optimal number of stocks to hold")
    logger.info("="*80)
    
    for top_n in [3, 5, 10, 15, 20, 30, 40, 50, 75, 100]:
        logger.info(f"\nTesting Top {top_n} stocks...")
        portfolio_df = bot.backtest(top_n=top_n)
        
        initial = bot.initial_capital
        final = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final / initial) ** (1/years) - 1) * 100
        
        cummax = portfolio_df['value'].cummax()
        drawdown = ((portfolio_df['value'] - cummax) / cummax * 100).min()
        
        results.append({
            'strategy': f'Top {top_n}',
            'annual_return': annual_return,
            'max_drawdown': drawdown,
            'final_value': final
        })
        
        logger.info(f"  → Annual: {annual_return:6.1f}% | Max DD: {drawdown:6.1f}% | Final: ${final:,.0f}")
    
    logger.info("\n" + "="*80)
    logger.info("STRATEGY 3: DEFENSIVE SCORING")
    logger.info("Focus on stocks with lowest volatility and smallest losses")
    logger.info("="*80)
    
    # Re-score with defensive criteria
    defensive_scores = bot.score_defensive()
    logger.info(f"\nDefensive scoring complete. Testing...")
    
    portfolio_df = bot.backtest_defensive(top_n=20)
    
    initial = bot.initial_capital
    final = portfolio_df['value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final / initial) ** (1/years) - 1) * 100
    
    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100).min()
    
    results.append({
        'strategy': 'Defensive Top 20',
        'annual_return': annual_return,
        'max_drawdown': drawdown,
        'final_value': final
    })
    
    logger.info(f"  → Annual: {annual_return:6.1f}% | Max DD: {drawdown:6.1f}% | Final: ${final:,.0f}")
    
    logger.info("\n" + "="*80)
    logger.info("BEST STRATEGIES RANKING")
    logger.info("="*80)
    
    # Sort by annual return (descending)
    results_sorted = sorted(results, key=lambda x: x['annual_return'], reverse=True)
    
    logger.info(f"\n{'Rank':<6} {'Strategy':<20} {'Annual Return':<15} {'Max Drawdown':<15} {'Final Value':<15}")
    logger.info("-" * 80)
    
    for i, result in enumerate(results_sorted[:10], 1):
        logger.info(f"{i:<6} {result['strategy']:<20} {result['annual_return']:>6.1f}%        {result['max_drawdown']:>6.1f}%        ${result['final_value']:>10,.0f}")
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    
    return results_sorted


def backtest_with_cash(self, top_n=10, cash_reserve=0.20, rebalance_freq='M'):
    """Backtest with adjustable cash reserve"""
    # Similar to regular backtest but with configurable cash
    
    all_dates = self.stocks_data[list(self.stocks_data.keys())[0]].index
    portfolio_values = []
    dates_list = []
    
    cash = self.initial_capital
    holdings = {}
    
    rebalance_dates = all_dates[::21] if rebalance_freq == 'M' else all_dates
    
    for current_date in all_dates:
        if current_date in rebalance_dates:
            # Liquidate
            for ticker in list(holdings.keys()):
                if ticker in self.stocks_data:
                    price = self.stocks_data[ticker].loc[current_date, 'close']
                    cash += holdings[ticker] * price
            holdings = {}
            
            # Get current scores
            top_stocks = [(ticker, score) for ticker, score in self.rankings[:top_n]
                         if ticker in self.stocks_data]
            
            # Allocate with cash reserve
            invest_amount = cash * (1 - cash_reserve)
            if len(top_stocks) > 0:
                per_stock = invest_amount / len(top_stocks)
                
                for ticker, score in top_stocks:
                    price = self.stocks_data[ticker].loc[current_date, 'close']
                    shares = per_stock / price
                    holdings[ticker] = shares
                    cash -= per_stock
        
        # Calculate portfolio value
        total_value = cash
        for ticker, shares in holdings.items():
            if ticker in self.stocks_data:
                price = self.stocks_data[ticker].loc[current_date, 'close']
                total_value += shares * price
        
        portfolio_values.append(total_value)
        dates_list.append(current_date)
    
    return pd.DataFrame({'value': portfolio_values}, index=dates_list)


def score_defensive(self):
    """Score stocks defensively - prioritize low volatility and capital preservation"""
    
    self.scores = {}
    
    for ticker, df in self.stocks_data.items():
        if len(df) < 200:
            self.scores[ticker] = 0
            continue
        
        score = 0
        latest = df.iloc[-1]
        
        # DEFENSIVE CRITERIA (Total: 100 points)
        
        # 1. Low Volatility (40 points) - MOST IMPORTANT in crash
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        if volatility < 15:
            score += 40
        elif volatility < 20:
            score += 30
        elif volatility < 25:
            score += 20
        elif volatility < 30:
            score += 10
        
        # 2. Smallest Loss (30 points) - Minimize damage
        returns_60d = ((df['close'].iloc[-1] / df['close'].iloc[-60]) - 1) * 100
        returns_120d = ((df['close'].iloc[-1] / df['close'].iloc[-120]) - 1) * 100
        
        avg_return = (returns_60d + returns_120d) / 2
        
        if avg_return > 0:
            score += 30  # Actually positive\!
        elif avg_return > -10:
            score += 25  # Small loss
        elif avg_return > -20:
            score += 20
        elif avg_return > -30:
            score += 15
        elif avg_return > -50:
            score += 10
        
        # 3. Positive Trend (20 points) - At least not getting worse
        if 'ema_13' in df.columns and 'ema_34' in df.columns:
            if latest['ema_13'] > latest['ema_34']:
                score += 20
            elif latest['ema_13'] > latest['ema_34'] * 0.98:
                score += 10
        
        # 4. RSI Not Oversold (10 points) - Avoid catching falling knives
        if 'rsi' in df.columns:
            rsi = latest['rsi']
            if 40 <= rsi <= 60:
                score += 10
            elif 30 <= rsi <= 70:
                score += 5
        
        self.scores[ticker] = min(score, 100)
    
    self.rankings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
    return self.scores


def backtest_defensive(self, top_n=20):
    """Backtest using defensive scoring"""
    return self.backtest(top_n=top_n)


# Add these methods to the PortfolioRotationBot class
PortfolioRotationBot.backtest_with_cash = backtest_with_cash
PortfolioRotationBot.score_defensive = score_defensive
PortfolioRotationBot.backtest_defensive = backtest_defensive



# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-data':
        num_stocks = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        logger.info(f"Generating data for {num_stocks} stocks...")
        generate_sp500_stocks_data(num_stocks=num_stocks)
    elif len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        analyze_performance()
    elif len(sys.argv) > 1 and sys.argv[1] == '--optimize':
        test_strategies_worst_case()
    else:
        bot = PortfolioRotationBot()
        results = bot.run()
