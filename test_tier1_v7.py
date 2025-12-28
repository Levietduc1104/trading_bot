"""V7 Tier 1 Optimizations Test"""
import sys
sys.path.insert(0, 'src/backtest')
import pandas as pd
import numpy as np
import logging
from portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Simplified sector map for top stocks
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
    'META': 'Technology', 'TSLA': 'Technology', 'AVGO': 'Technology', 'ORCL': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology', 'QCOM': 'Technology',
    'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials', 'BAC': 'Financials',
    'WFC': 'Financials', 'MS': 'Financials', 'GS': 'Financials', 'BLK': 'Financials',
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'PFE': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'AMZN': 'Consumer Discretionary', 'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials', 'RTX': 'Industrials',
}

class V7Bot(PortfolioRotationBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sector_map = SECTOR_MAP
    
    def get_seasonal_multiplier(self, date):
        month = date.month
        if month in [11, 12, 1, 2, 3, 4]:  # Winter: aggressive
            return 0.85
        else:  # Summer: defensive
            return 1.15
    
    def should_rebalance_midmonth(self, date, last_date):
        if last_date is None:
            return True
        current = (date.year, date.month)
        last = (last_date.year, last_date.month)
        return current != last and 7 <= date.day <= 10
    
    def calc_sector_rs(self, ticker, df_at_date):
        if len(df_at_date) < 60:
            return 0
        sector = self.sector_map.get(ticker, 'Other')
        tick_ret = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1) * 100
        sect_rets = []
        for peer, psect in self.sector_map.items():
            if psect == sector and peer != ticker and peer in self.stocks_data:
                pdf = self.stocks_data[peer]
                pdate = pdf[pdf.index <= df_at_date.index[-1]]
                if len(pdate) >= 60:
                    pret = (pdate['close'].iloc[-1] / pdate['close'].iloc[-60] - 1) * 100
                    sect_rets.append(pret)
        if not sect_rets:
            return 0
        rs = tick_ret - np.mean(sect_rets)
        if rs > 10:
            return 15
        elif rs > 5:
            return 10
        elif rs > 2:
            return 5
        elif rs < -5:
            return -10
        return 0
    
    def score_v7(self, ticker, df):
        latest = df.iloc[-1]
        score = 0
        close = latest['close']
        ema_13, ema_34, ema_89 = latest['ema_13'], latest['ema_34'], latest['ema_89']
        
        if close > ema_13 > ema_34:
            score += 20
        elif close > ema_13:
            score += 10
        if close > ema_89:
            score += 30
            if ema_34 > ema_89:
                score += 10
        elif close > ema_89 * 0.95:
            score += 15
        
        roc = latest['roc_20']
        if roc > 15:
            score += 30
        elif roc > 10:
            score += 20
        elif roc > 5:
            score += 15
        elif roc > 0:
            score += 10
        
        atr_pct = (latest['atr'] / close) * 100
        if atr_pct < 2:
            score += 20
        elif atr_pct < 3:
            score += 15
        elif atr_pct < 4:
            score += 10
        elif atr_pct < 5:
            score += 5
        
        # V6 filters
        if close < ema_89:
            return 0
        if roc < 2:
            return 0
        rsi = latest['rsi']
        if rsi > 75:
            score *= 0.7
        if rsi < 30:
            score *= 0.5
        
        # V7 sector bonus
        score += self.calc_sector_rs(ticker, df)
        return min(score, 150)
    
    def backtest_v7(self, top_n=10):
        logger.info("V7 backtest with Tier 1 optimizations")
        first = list(self.stocks_data.keys())[0]
        dates = self.stocks_data[first].index
        
        values = []
        cash = self.initial_capital
        holdings = {}
        last_reb = None
        
        for date in dates[100:]:
            if self.should_rebalance_midmonth(date, last_reb):
                last_reb = date
                
                # Liquidate
                for t in list(holdings.keys()):
                    df = self.stocks_data[t][self.stocks_data[t].index <= date]
                    if len(df) > 0:
                        cash += holdings[t] * df.iloc[-1]['close']
                holdings = {}
                
                # Score
                scores = {}
                for t, df in self.stocks_data.items():
                    dfd = df[df.index <= date]
                    if len(dfd) >= 100:
                        try:
                            scores[t] = self.score_v7(t, dfd)
                        except:
                            pass
                
                # Cash reserve with seasonal adjustment
                base_cash = self.calculate_adaptive_regime(date)
                seasonal = self.get_seasonal_multiplier(date)
                cash_res = max(0.05, min(0.70, base_cash * seasonal))
                
                # Select top
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top = [t for t, s in ranked if s > 0][:top_n]
                
                # Allocate
                invest = cash * (1 - cash_res)
                alloc = invest / len(top) if top else 0
                
                for t in top:
                    df = self.stocks_data[t][self.stocks_data[t].index <= date]
                    if len(df) > 0:
                        holdings[t] = alloc / df.iloc[-1]['close']
                        cash -= alloc
            
            # Value
            sv = sum(holdings[t] * self.stocks_data[t][self.stocks_data[t].index <= date].iloc[-1]['close']
                     for t in holdings if len(self.stocks_data[t][self.stocks_data[t].index <= date]) > 0)
            values.append({'date': date, 'value': cash + sv})
        
        df = pd.DataFrame(values).set_index('date')
        final = df['value'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        annual = ((final / self.initial_capital) ** (1/years) - 1) * 100
        cummax = df['value'].cummax()
        dd = ((df['value'] - cummax) / cummax * 100).min()
        rets = df['value'].pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        
        logger.info("="*60)
        logger.info("V7 RESULTS")
        logger.info(f"Annual: {annual:.1f}%, DD: {dd:.1f}%, Sharpe: {sharpe:.2f}")
        logger.info(f"Final: ${final:,.0f}")
        logger.info("="*60)
        
        return {'annual_return': annual, 'max_drawdown': dd, 'sharpe': sharpe, 'final_value': final}

# Run test
logger.info("="*60)
logger.info("TIER 1 TEST: Seasonal + MidMonth + SectorRS")
logger.info("="*60)

logger.info("\n>>> V6 Baseline")
from test_momentum_filters import MomentumFilteredBot
v6 = MomentumFilteredBot(data_dir='sp500_data/daily', initial_capital=100000)
v6.prepare_data()
r6 = v6.backtest_v6(top_n=10)

logger.info("\n>>> V7 Tier 1")
v7 = V7Bot(data_dir='sp500_data/daily', initial_capital=100000)
v7.prepare_data()
r7 = v7.backtest_v7(top_n=10)

logger.info("\n" + "="*60)
logger.info("COMPARISON")
logger.info("="*60)
logger.info(f"{'Metric':<20} {'V6':<10} {'V7':<10} {'Change':<10}")
logger.info("-"*50)
logger.info(f"{'Annual':<20} {r6['annual_return']:>5.1f}%     {r7['annual_return']:>5.1f}%     {r7['annual_return']-r6['annual_return']:>+5.1f}%")
logger.info(f"{'Drawdown':<20} {r6['max_drawdown']:>5.1f}%     {r7['max_drawdown']:>5.1f}%     {r7['max_drawdown']-r6['max_drawdown']:>+5.1f}%")
logger.info(f"{'Sharpe':<20} {r6['sharpe']:>5.2f}      {r7['sharpe']:>5.2f}      {r7['sharpe']-r6['sharpe']:>+5.2f}")
logger.info("-"*50)

ch = r7['annual_return'] - r6['annual_return']
if ch >= 1.2:
    logger.info("✅ EXCELLENT\! Target hit")
elif ch >= 0.5:
    logger.info("✅ SUCCESS\!")
else:
    logger.info("⚠️  Modest gain")
