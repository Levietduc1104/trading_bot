"""
Monte Carlo Simulator for V28 Strategy Robustness Testing

Tests strategy performance across:
- Parameter variations (portfolio size, Kelly exponent, VIX multipliers, fees)
- Bootstrap resampling (different entry timing)
- Stress testing (extreme market conditions)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import sqlite3
from typing import Dict, List, Tuple, Optional

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation framework for trading strategies
    
    Runs multiple simulations with parameter variations to assess:
    - Performance robustness
    - Parameter sensitivity
    - Risk distributions
    - Confidence intervals
    """
    
    def __init__(self,
                 data_dir='sp500_data/daily',
                 initial_capital=100000,
                 strategy='V28'):
        """
        Initialize Monte Carlo simulator
        
        Args:
            data_dir: Path to stock data
            initial_capital: Starting capital
            strategy: Strategy name (V28, V35, etc.)
        """
        self.data_dir = os.path.join(project_root, data_dir)
        self.initial_capital = initial_capital
        self.strategy = strategy
        self.results = []
        
    def run_single_simulation(self,
                             sim_id: int,
                             portfolio_size: Optional[int] = None,
                             kelly_exponent: float = 0.5,
                             vix_multiplier: float = 1.0,
                             fee_pct: float = 0.001,
                             start_offset_days: int = 0,
                             stress_scenario: Optional[str] = None) -> Optional[Dict]:
        """
        Run a single Monte Carlo simulation
        
        Args:
            sim_id: Simulation identifier
            portfolio_size: Number of stocks (None = dynamic)
            kelly_exponent: Kelly weighting exponent (0.5 = sqrt)
            vix_multiplier: VIX cash reserve multiplier
            fee_pct: Trading fee percentage
            start_offset_days: Days to offset start date
            stress_scenario: Stress test scenario name
            
        Returns:
            Dictionary with simulation results
        """
        try:
            # Initialize bot
            bot = PortfolioRotationBot(
                data_dir=self.data_dir,
                initial_capital=self.initial_capital
            )
            
            # Load data
            bot.prepare_data()
            bot.score_all_stocks()
            
            # Run backtest variant
            portfolio_df = self._run_backtest_variant(
                bot=bot,
                portfolio_size=portfolio_size,
                kelly_exponent=kelly_exponent,
                vix_multiplier=vix_multiplier,
                fee_pct=fee_pct,
                start_offset_days=start_offset_days,
                stress_scenario=stress_scenario
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(portfolio_df, self.initial_capital)
            metrics['sim_id'] = sim_id
            metrics['parameters'] = {
                'portfolio_size': portfolio_size,
                'kelly_exponent': kelly_exponent,
                'vix_multiplier': vix_multiplier,
                'fee_pct': fee_pct,
                'start_offset_days': start_offset_days,
                'stress_scenario': stress_scenario
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Simulation {sim_id} failed: {e}")
            return None
    
    def _run_backtest_variant(self,
                             bot,
                             portfolio_size: Optional[int],
                             kelly_exponent: float,
                             vix_multiplier: float,
                             fee_pct: float,
                             start_offset_days: int,
                             stress_scenario: Optional[str]) -> pd.DataFrame:
        """Run V28 backtest with parameter variations"""
        
        first_ticker = list(bot.stocks_data.keys())[0]
        dates = bot.stocks_data[first_ticker].index
        
        # Apply start offset
        if start_offset_days > 0:
            dates = dates[start_offset_days:]
        
        portfolio_values = []
        holdings = {}
        cash = bot.initial_capital
        last_rebalance_date = None
        
        for date in dates[100:]:
            # Monthly rebalancing
            is_rebalance_day = (
                last_rebalance_date is None or
                (
                    (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                    7 <= date.day <= 10
                )
            )
            
            if is_rebalance_day:
                # Liquidate holdings
                for ticker in list(holdings.keys()):
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        cash += holdings[ticker] * current_price
                holdings = {}
                last_rebalance_date = date
                
                # Get VIX
                vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
                if vix_at_date is not None and len(vix_at_date) > 0:
                    vix = vix_at_date.iloc[-1]['close']
                else:
                    vix = 20
                
                # Apply stress scenario if specified
                if stress_scenario:
                    vix = self._apply_stress_scenario(stress_scenario, vix, date)
                
                # Score stocks
                current_scores = {}
                for ticker, df in bot.stocks_data.items():
                    df_at_date = df[df.index <= date]
                    if len(df_at_date) >= 100:
                        try:
                            current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                        except:
                            pass
                
                # Determine portfolio size
                if portfolio_size is None:
                    top_n = bot.determine_portfolio_size(date)
                else:
                    top_n = portfolio_size
                
                # Get top N stocks
                ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
                top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]
                
                if not top_stocks:
                    portfolio_values.append({'date': date, 'value': cash})
                    continue
                
                # VIX-based cash reserve with multiplier
                if vix < 30:
                    cash_reserve = 0.05 + (vix - 10) * 0.005
                else:
                    cash_reserve = 0.15 + (vix - 30) * 0.0125
                cash_reserve = np.clip(cash_reserve * vix_multiplier, 0.05, 0.70)
                
                invest_amount = cash * (1 - cash_reserve)
                
                # Kelly position sizing with custom exponent
                kelly_weights = self._calculate_kelly_weights(top_stocks, kelly_exponent)
                
                allocations = {
                    ticker: invest_amount * weight
                    for ticker, weight in kelly_weights.items()
                }
                
                # Drawdown control
                portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
                if portfolio_df is not None and len(portfolio_df) > 1:
                    drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                    allocations = {
                        ticker: amount * drawdown_multiplier
                        for ticker, amount in allocations.items()
                    }
                
                # Buy stocks
                for ticker, _ in top_stocks:
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        allocation_amount = allocations.get(ticker, 0)
                        shares = allocation_amount / current_price
                        holdings[ticker] = shares
                        fee = allocation_amount * fee_pct
                        cash -= (allocation_amount + fee)
            
            # Calculate daily portfolio value
            stocks_value = 0
            for ticker, shares in holdings.items():
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    stocks_value += shares * current_price
            
            total_value = cash + stocks_value
            portfolio_values.append({'date': date, 'value': total_value})
        
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
        portfolio_df = portfolio_df.rename(columns={'value': 'portfolio_value'})
        
        return portfolio_df
    
    def _calculate_kelly_weights(self, scored_stocks, exponent):
        """Calculate Kelly weights with custom exponent"""
        tickers = [t for t, s in scored_stocks]
        scores = [s for t, s in scored_stocks]
        
        # Apply exponent (0.5 = sqrt, 1.0 = linear)
        weighted_scores = [max(0, score) ** exponent for score in scores]
        total_weighted = sum(weighted_scores)
        
        if total_weighted > 0:
            weights = {
                ticker: (max(0, score) ** exponent) / total_weighted
                for ticker, score in scored_stocks
            }
        else:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        return weights
    
    def _apply_stress_scenario(self, scenario: str, vix: float, date) -> float:
        """Apply stress test scenario modifications"""
        if scenario == '2008_crisis':
            return np.clip(vix * 2.5, 40, 80)
        elif scenario == '2020_covid':
            return np.clip(vix * 2.0, 30, 85)
        elif scenario == 'flash_crash':
            return np.clip(vix * 3.0, 50, 90)
        elif scenario == 'prolonged_bear':
            return np.clip(vix * 1.5, 25, 40)
        else:
            return vix
    
    def _calculate_metrics(self, portfolio_df, initial_capital):
        """Calculate performance metrics"""
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        years = (end_date - start_date).days / 365.25
        annual_return = (((final_value / initial_capital) ** (1 / years)) - 1) * 100
        
        cummax = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        returns = portfolio_df['portfolio_value'].pct_change()
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = (returns.mean() / downside_std) * (252 ** 0.5) if downside_std > 0 else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'years': years,
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date())
        }
    
    def save_results_to_database(self, mc_run_id: int, db_path: str):
        """Save Monte Carlo results to database"""
        logger.info(f"Saving {len(self.results)} results to database...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            for result in self.results:
                cursor.execute('''
                    INSERT INTO monte_carlo_results (
                        mc_run_id, simulation_number, parameters,
                        annual_return, max_drawdown, sharpe_ratio,
                        sortino_ratio, calmar_ratio, final_value,
                        start_date, end_date, years
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    mc_run_id,
                    result['sim_id'],
                    json.dumps(result['parameters']),
                    result['annual_return'],
                    result['max_drawdown'],
                    result['sharpe_ratio'],
                    result['sortino_ratio'],
                    result['calmar_ratio'],
                    result['final_value'],
                    result['start_date'],
                    result['end_date'],
                    result['years']
                ))
            
            conn.commit()
            logger.info("✓ Results saved to database")
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
