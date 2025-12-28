"""V8: Add VIX Regime Detection to V7"""
import sys
sys.path.insert(0, 'src/backtest')
import pandas as pd
import numpy as np
import logging
from test_tier1_v7 import V7Bot, SECTOR_MAP

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class V8VIXBot(V7Bot):
    """
    V8: VIX-Based Regime Detection
    
    Adds VIX volatility index for better regime detection
    VIX is forward-looking (implied volatility) vs SPY 200 MA (lagging)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vix_data = None
    
    def load_vix_data(self):
        """Load VIX proxy data"""
        try:
            vix = pd.read_csv('sp500_data/daily/VIX.csv', index_col=0, parse_dates=True)
            vix.columns = [c.lower() for c in vix.columns]
            self.vix_data = vix
            logger.info(f"Loaded VIX data: {len(vix)} days")
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")
            self.vix_data = None
    
    def calculate_vix_regime(self, date):
        """
        VIX-Based Regime Detection
        
        Uses volatility levels to determine cash reserve:
        - Low VIX = low fear = aggressive (low cash)
        - High VIX = high fear = defensive (high cash)
        
        VIX regimes (adjusted for our proxy which runs higher):
        - VIX < 30: Very low fear -> 5% cash
        - VIX 30-40: Low fear -> 15% cash
        - VIX 40-50: Moderate fear -> 30% cash
        - VIX 50-70: High fear -> 50% cash
        - VIX > 70: Panic -> 70% cash
        """
        if self.vix_data is None:
            return self.calculate_adaptive_regime(date)
        
        # Get VIX value at this date
        vix_at_date = self.vix_data[self.vix_data.index <= date]
        if len(vix_at_date) == 0:
            return self.calculate_adaptive_regime(date)
        
        vix = vix_at_date.iloc[-1]['close']
        
        # VIX-based regimes
        if vix < 30:
            cash_reserve = 0.05  # Very low fear: aggressive
        elif vix < 40:
            cash_reserve = 0.15  # Low fear
        elif vix < 50:
            cash_reserve = 0.30  # Moderate fear
        elif vix < 70:
            cash_reserve = 0.50  # High fear
        else:
            cash_reserve = 0.70  # Panic mode
        
        return cash_reserve
    
    def backtest_v8(self, top_n=10):
        """
        V8 Backtest with VIX regime
        Combines V7 optimizations + VIX-based regime detection
        """
        logger.info("V8 backtest with VIX regime detection")
        
        # Load VIX data first
        self.load_vix_data()
        
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
                
                # Score with V7
                scores = {}
                for t, df in self.stocks_data.items():
                    dfd = df[df.index <= date]
                    if len(dfd) >= 100:
                        try:
                            scores[t] = self.score_v7(t, dfd)
                        except:
                            pass
                
                # VIX-based regime (NEW\!)
                base_cash = self.calculate_vix_regime(date)
                
                # Apply seasonal adjustment (from V7)
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
            
            # Calculate value
            sv = sum(holdings[t] * self.stocks_data[t][self.stocks_data[t].index <= date].iloc[-1]['close']
                     for t in holdings if len(self.stocks_data[t][self.stocks_data[t].index <= date]) > 0)
            values.append({'date': date, 'value': cash + sv})
        
        # Metrics
        df = pd.DataFrame(values).set_index('date')
        final = df['value'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        annual = ((final / self.initial_capital) ** (1/years) - 1) * 100
        cummax = df['value'].cummax()
        dd = ((df['value'] - cummax) / cummax * 100).min()
        rets = df['value'].pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        
        logger.info("="*60)
        logger.info("V8 VIX REGIME RESULTS")
        logger.info(f"Annual: {annual:.1f}%, DD: {dd:.1f}%, Sharpe: {sharpe:.2f}")
        logger.info(f"Final: ${final:,.0f}")
        logger.info("="*60)
        
        return {'annual_return': annual, 'max_drawdown': dd, 'sharpe': sharpe, 'final_value': final}

# Run comparison
logger.info("="*60)
logger.info("V8 TEST: V7 + VIX Regime Detection")
logger.info("="*60)

logger.info("\n>>> V7 Tier 1 Baseline")
v7 = V7Bot(data_dir='sp500_data/daily', initial_capital=100000)
v7.prepare_data()
r7 = v7.backtest_v7(top_n=10)

logger.info("\n>>> V8 with VIX Regime")
v8 = V8VIXBot(data_dir='sp500_data/daily', initial_capital=100000)
v8.prepare_data()
r8 = v8.backtest_v8(top_n=10)

logger.info("\n" + "="*60)
logger.info("COMPARISON: V7 vs V8")
logger.info("="*60)
logger.info(f"{'Metric':<20} {'V7':<10} {'V8':<10} {'Change':<10}")
logger.info("-"*50)
logger.info(f"{'Annual':<20} {r7['annual_return']:>5.1f}%     {r8['annual_return']:>5.1f}%     {r8['annual_return']-r7['annual_return']:>+5.1f}%")
logger.info(f"{'Drawdown':<20} {r7['max_drawdown']:>5.1f}%     {r8['max_drawdown']:>5.1f}%     {r8['max_drawdown']-r7['max_drawdown']:>+5.1f}%")
logger.info(f"{'Sharpe':<20} {r7['sharpe']:>5.2f}      {r8['sharpe']:>5.2f}      {r8['sharpe']-r7['sharpe']:>+5.2f}")
logger.info(f"{'Final Value':<20} ${r7['final_value']:>7,.0f}  ${r8['final_value']:>7,.0f}  ${r8['final_value']-r7['final_value']:>+7,.0f}")
logger.info("-"*50)

ch_annual = r8['annual_return'] - r7['annual_return']
ch_dd = r8['max_drawdown'] - r7['max_drawdown']

logger.info("")
if ch_annual >= 0.5:
    logger.info("✅ SUCCESS\! VIX regime improved returns")
elif ch_annual > 0:
    logger.info("⚠️  Modest gain from VIX")
else:
    logger.info("❌ VIX regime did not improve returns")

if ch_dd > 0:
    logger.info(f"⚠️  Drawdown worse by {ch_dd:.1f}%")
else:
    logger.info(f"✅ Drawdown improved by {abs(ch_dd):.1f}%\!")

logger.info("")
logger.info("="*60)
