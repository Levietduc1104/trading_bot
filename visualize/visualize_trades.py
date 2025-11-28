"""
Interactive Trading Visualization with Bokeh - Multi-Tab Dashboard
Tab 1: Trading Analysis (entry/exit, holdings)
Tab 2: Principal Investment Performance (PnL, returns, drawdown)
"""

import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, TabPanel, Tabs, Div, BoxAnnotation, Label, Select, CustomJS, ColumnDataSource
from bokeh.palettes import Category20_20, Viridis256
import sys
sys.path.append("..")
from portfolio_bot_demo import PortfolioRotationBot
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_trade_visualizations():
    """Create comprehensive multi-tab trading visualizations"""
    
    logger.info("="*80)
    logger.info("CREATING MULTI-TAB TRADING DASHBOARD")
    logger.info("="*80)
    
    # Initialize bot
    data_dir = '../sp500_data/daily'
    bot = PortfolioRotationBot(data_dir=data_dir)
    bot.prepare_data()
    bot.score_all_stocks()
    
    # Run backtest
    logger.info("\nRunning backtest...")
    portfolio_df = bot.backtest(top_n=10)
    trades_log = track_trades_detailed(bot, top_n=10)
    
    # Create output file
    output_file("trading_analysis.html")
    
    # TAB 1: Trading Analysis
    logger.info("\n" + "="*80)
    logger.info("TAB 1: Creating Trading Analysis Charts...")
    logger.info("="*80)
    tab1_content = create_trading_analysis_tab(trades_log, bot)
    
    # TAB 2: Principal Investment Performance
    logger.info("\n" + "="*80)
    logger.info("TAB 2: Creating Investment Performance Charts...")
    logger.info("="*80)
    tab2_content = create_performance_tab(portfolio_df, bot.initial_capital)
    
    # Create tabs
    tab1 = TabPanel(child=tab1_content, title="🔄 Trading Analysis")
    
    # TAB 3: Interactive Stock Price Viewer
    logger.info("\n" + "="*80)
    logger.info("TAB 3: Creating Interactive Stock Price Viewer...")
    logger.info("="*80)
    tab3_content = create_stock_selector_tab(bot, trades_log)
    
    # Create tabs
    tab1 = TabPanel(child=tab1_content, title="🔄 Trading Analysis")
    tab2 = TabPanel(child=tab2_content, title="📈 Investment Performance (PnL)")
    tab3 = TabPanel(child=tab3_content, title="📊 Stock Price Viewer")
    
    tabs = Tabs(tabs=[tab1, tab2, tab3])
    # Save
    save(tabs)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Multi-tab dashboard saved to: visualize/trading_analysis.html")
    logger.info("="*80)
    logger.info("\nTabs created:")
    logger.info("  Tab 1: Trading Analysis (entry/exit points, holdings timeline)")


def create_trading_analysis_tab(trades_log, bot):
    """Create Tab 1: Trading analysis content"""
    
    plots = []
    
    # Portfolio composition
    logger.info("  - Portfolio composition chart")
    p1 = create_portfolio_composition_chart(trades_log, bot)
    plots.append(p1)
    
    # Individual stock charts
    logger.info("  - Individual stock charts (top 6)")
    most_traded = get_most_traded_stocks(trades_log, top_n=6)
    stock_plots = []
    for ticker in most_traded:
        p = create_stock_price_chart(ticker, trades_log, bot)
        stock_plots.append(p)
    
    grid = gridplot(stock_plots, ncols=2, width=600, height=400)
    plots.append(grid)
    
    # Holdings timeline
    logger.info("  - Holdings timeline")
    p3 = create_holdings_timeline(trades_log)
    plots.append(p3)
    
    # Trade frequency
    logger.info("  - Trade frequency chart")
    p4 = create_trade_frequency_chart(trades_log)
    plots.append(p4)
    
    return column(*plots)


def create_performance_tab(portfolio_df, initial_capital):
    """Create Tab 2: Principal investment performance content"""
    
    plots = []
    
    # Calculate metrics
    logger.info("  - Calculating performance metrics")
    metrics = calculate_metrics(portfolio_df, initial_capital)
    
    # 1. Portfolio value over time (MAIN CHART)
    logger.info("  - Portfolio value chart (PnL)")
    p1 = create_portfolio_value_chart(portfolio_df, initial_capital, metrics)
    plots.append(p1)
    
    # 2. Returns analysis (2 charts)
    logger.info("  - Cumulative returns & daily returns")
    p2a = create_cumulative_returns_chart(portfolio_df, initial_capital)
    p2b = create_daily_returns_histogram(portfolio_df)
    plots.append(row(p2a, p2b))
    
    # 3. Drawdown chart
    logger.info("  - Drawdown analysis")
    p3 = create_drawdown_chart(portfolio_df)
    plots.append(p3)
    
    # 4. Monthly returns
    logger.info("  - Monthly returns")
    p4 = create_monthly_returns_chart(portfolio_df)
    plots.append(p4)
    
    # 5. Metrics summary
    logger.info("  - Metrics summary table")
    p5 = create_metrics_summary(metrics, portfolio_df)
    plots.append(p5)
    
    return column(*plots)


# ============================================================================
# TAB 2: PERFORMANCE CHARTS
# ============================================================================





def create_stock_selector_tab(bot, trades_log):
    """Create Tab 3: Interactive stock price selector with CANDLESTICK chart"""
    
    # Get all available stocks
    all_tickers = sorted(bot.stocks_data.keys())
    
    logger.info(f"  - Creating interactive candlestick chart for {len(all_tickers)} stocks")
    
    # Default stock
    default_ticker = all_tickers[0]
    
    # Create data sources for all stocks
    sources = {}
    trade_sources = {}
    
    for ticker in all_tickers:
        df = bot.stocks_data[ticker]
        
        # Prepare candlestick data
        inc = df['close'] > df['open']
        dec = df['open'] > df['close']
        
        sources[ticker] = {
            'date': df.index.values,
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'inc': inc.values,
            'dec': dec.values,
        }
        
        # Trade data
        stock_trades = [t for t in trades_log if t.get('ticker') == ticker and t.get('action') in ['BUY', 'SELL']]
        
        buy_dates = [t['date'] for t in stock_trades if t['action'] == 'BUY']
        buy_prices = [t['price'] for t in stock_trades if t['action'] == 'BUY']
        sell_dates = [t['date'] for t in stock_trades if t['action'] == 'SELL']
        sell_prices = [t['price'] for t in stock_trades if t['action'] == 'SELL']
        
        trade_sources[ticker] = {
            'buy': ColumnDataSource(data={'date': buy_dates, 'price': buy_prices}),
            'sell': ColumnDataSource(data={'date': sell_dates, 'price': sell_prices})
        }
    
    # Create the plot
    p = figure(x_axis_type="datetime", width=1400, height=600,
               title=f"Stock Price (Candlestick): {default_ticker}",
               tools="pan,wheel_zoom,box_zoom,reset,save")
    
    # Create candlesticks using ColumnDataSource
    source = ColumnDataSource(sources[default_ticker])
    
    # Width for candlesticks (1 day in milliseconds)
    w = 12*60*60*1000  # half day in ms
    
    # Increasing candles (green)
    inc_source = ColumnDataSource(data={
        'date': sources[default_ticker]['date'][sources[default_ticker]['inc']],
        'open': sources[default_ticker]['open'][sources[default_ticker]['inc']],
        'close': sources[default_ticker]['close'][sources[default_ticker]['inc']],
        'high': sources[default_ticker]['high'][sources[default_ticker]['inc']],
        'low': sources[default_ticker]['low'][sources[default_ticker]['inc']],
    })
    
    # Decreasing candles (red)
    dec_source = ColumnDataSource(data={
        'date': sources[default_ticker]['date'][sources[default_ticker]['dec']],
        'open': sources[default_ticker]['open'][sources[default_ticker]['dec']],
        'close': sources[default_ticker]['close'][sources[default_ticker]['dec']],
        'high': sources[default_ticker]['high'][sources[default_ticker]['dec']],
        'low': sources[default_ticker]['low'][sources[default_ticker]['dec']],
    })
    
    # Plot increasing candles
    inc_candles = p.segment('date', 'high', 'date', 'low', source=inc_source, color="green", line_width=1)
    inc_bars = p.vbar('date', w, 'open', 'close', source=inc_source, fill_color="green", line_color="green", alpha=0.8)
    
    # Plot decreasing candles
    dec_candles = p.segment('date', 'high', 'date', 'low', source=dec_source, color="red", line_width=1)
    dec_bars = p.vbar('date', w, 'open', 'close', source=dec_source, fill_color="red", line_color="red", alpha=0.8)
    
    # Buy markers (triangles)
    buy_markers = p.scatter('date', 'price', source=trade_sources[default_ticker]['buy'],
                           size=15, color='blue', alpha=0.9, marker='triangle',
                           legend_label='Buy', line_color='white', line_width=2)
    
    # Sell markers (inverted triangles)
    sell_markers = p.scatter('date', 'price', source=trade_sources[default_ticker]['sell'],
                            size=15, color='orange', alpha=0.9, marker='inverted_triangle',
                            legend_label='Sell', line_color='white', line_width=2)
    
    p.yaxis.axis_label = "Price ($)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Date", "@date{%F}"),
        ("Open", "$@open{0.2f}"),
        ("High", "$@high{0.2f}"),
        ("Low", "$@low{0.2f}"),
        ("Close", "$@close{0.2f}"),
    ], formatters={'@date': 'datetime'})
    p.add_tools(hover)
    
    # Create dropdown selector
    select = Select(title="Choose Stock:", value=default_ticker, options=all_tickers, width=300)
    
    # JavaScript callback to update the chart
    callback = CustomJS(args=dict(
        p=p,
        inc_source=inc_source,
        dec_source=dec_source,
        buy_markers=buy_markers,
        sell_markers=sell_markers,
        sources=sources,
        trade_sources=trade_sources
    ), code="""
        const ticker = cb_obj.value;
        const data = sources[ticker];
        
        // Update title
        p.title.text = 'Stock Price (Candlestick): ' + ticker;
        
        // Filter increasing and decreasing candles
        const inc_dates = [];
        const inc_opens = [];
        const inc_closes = [];
        const inc_highs = [];
        const inc_lows = [];
        
        const dec_dates = [];
        const dec_opens = [];
        const dec_closes = [];
        const dec_highs = [];
        const dec_lows = [];
        
        for (let i = 0; i < data.date.length; i++) {
            if (data.inc[i]) {
                inc_dates.push(data.date[i]);
                inc_opens.push(data.open[i]);
                inc_closes.push(data.close[i]);
                inc_highs.push(data.high[i]);
                inc_lows.push(data.low[i]);
            } else if (data.dec[i]) {
                dec_dates.push(data.date[i]);
                dec_opens.push(data.open[i]);
                dec_closes.push(data.close[i]);
                dec_highs.push(data.high[i]);
                dec_lows.push(data.low[i]);
            }
        }
        
        // Update increasing candles
        inc_source.data = {
            date: inc_dates,
            open: inc_opens,
            close: inc_closes,
            high: inc_highs,
            low: inc_lows
        };
        
        // Update decreasing candles
        dec_source.data = {
            date: dec_dates,
            open: dec_opens,
            close: dec_closes,
            high: dec_highs,
            low: dec_lows
        };
        
        // Update buy markers
        buy_markers.data_source.data = trade_sources[ticker]['buy'].data;
        
        // Update sell markers  
        sell_markers.data_source.data = trade_sources[ticker]['sell'].data;
    """)
    
    select.js_on_change('value', callback)
    
    # Info text
    info_html = f"""
    <div style="font-family: Arial; padding: 15px; background-color: #e8f4f8; border-radius: 5px; margin-bottom: 10px;">
        <h3 style="margin-top: 0; color: #2c3e50;">📊 Interactive Candlestick Chart Viewer</h3>
        <p><strong>Total Stocks Available:</strong> {len(all_tickers)}</p>
        <p><strong>How to Read Candlesticks:</strong></p>
        <ul>
            <li><strong style="color: green;">Green candles:</strong> Price went UP that day (close > open)</li>
            <li><strong style="color: red;">Red candles:</strong> Price went DOWN that day (close < open)</li>
            <li><strong>Candle body:</strong> Shows open and close prices</li>
            <li><strong>Wicks (lines):</strong> Show high and low prices for the day</li>
        </ul>
        <p><strong>Trading Signals:</strong></p>
        <ul>
            <li><strong style="color: blue;">Blue triangles (▲):</strong> We BOUGHT the stock at this price</li>
            <li><strong style="color: orange;">Orange triangles (▼):</strong> We SOLD the stock at this price</li>
        </ul>
        <p><strong>How to Use:</strong></p>
        <ul>
            <li>Select a stock from the dropdown menu</li>
            <li>Hover over candles to see OHLC prices</li>
            <li>Use zoom and pan tools to explore different time periods</li>
            <li>Click legend items to hide/show elements</li>
        </ul>
        <p><strong>Available Stocks:</strong> {len(all_tickers)} S&P 500 companies (2018-2024, 1,764 days each)</p>
    </div>
    """
    
    info_div = Div(text=info_html, width=1400)
    
    layout = column(info_div, select, p)
    
    return layout


    """Create Tab 3: Interactive stock price selector with dropdown"""
    
    # Get all available stocks
    all_tickers = sorted(bot.stocks_data.keys())
    
    logger.info(f"  - Creating interactive selector for {len(all_tickers)} stocks")
    
    # Default stock
    default_ticker = all_tickers[0]
    
    # Create data sources for all stocks
    sources = {}
    trade_sources = {}
    
    for ticker in all_tickers:
        df = bot.stocks_data[ticker]
        
        # Price data
        sources[ticker] = ColumnDataSource(data={
            'date': df.index.values,
            'close': df['close'].values,
        })
        
        # Trade data
        stock_trades = [t for t in trades_log if t.get('ticker') == ticker and t.get('action') in ['BUY', 'SELL']]
        
        buy_dates = [t['date'] for t in stock_trades if t['action'] == 'BUY']
        buy_prices = [t['price'] for t in stock_trades if t['action'] == 'BUY']
        sell_dates = [t['date'] for t in stock_trades if t['action'] == 'SELL']
        sell_prices = [t['price'] for t in stock_trades if t['action'] == 'SELL']
        
        trade_sources[ticker] = {
            'buy': ColumnDataSource(data={'date': buy_dates, 'price': buy_prices}),
            'sell': ColumnDataSource(data={'date': sell_dates, 'price': sell_prices})
        }
    
    # Create the plot
    p = figure(x_axis_type="datetime", width=1400, height=600,
               title=f"Stock Price: {default_ticker}")
    
    # Price line
    price_line = p.line('date', 'close', source=sources[default_ticker],
                        line_width=2, color='navy', alpha=0.8, legend_label='Price')
    
    # Buy markers
    buy_markers = p.scatter('date', 'price', source=trade_sources[default_ticker]['buy'],
                           size=12, color='green', alpha=0.8, marker='triangle',
                           legend_label='Buy')
    
    # Sell markers
    sell_markers = p.scatter('date', 'price', source=trade_sources[default_ticker]['sell'],
                            size=10, color='red', alpha=0.8, legend_label='Sell')
    
    p.yaxis.axis_label = "Price ($)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "top_left"
    
    # Add hover
    hover = HoverTool(tooltips=[("Date", "@date{%F}"), ("Price", "$@close{0.2f}")],
                      formatters={'@date': 'datetime'})
    p.add_tools(hover)
    
    # Create dropdown selector
    select = Select(title="Choose Stock:", value=default_ticker, options=all_tickers, width=300)
    
    # JavaScript callback to update the chart
    callback = CustomJS(args=dict(
        p=p,
        price_line=price_line,
        buy_markers=buy_markers,
        sell_markers=sell_markers,
        sources=sources,
        trade_sources=trade_sources
    ), code="""
        const ticker = cb_obj.value;
        
        // Update title
        p.title.text = 'Stock Price: ' + ticker;
        
        // Update price line data
        price_line.data_source.data = sources[ticker].data;
        
        // Update buy markers
        buy_markers.data_source.data = trade_sources[ticker]['buy'].data;
        
        // Update sell markers  
        sell_markers.data_source.data = trade_sources[ticker]['sell'].data;
    """)
    
    select.js_on_change('value', callback)
    
    # Info text
    info_html = f"""
    <div style="font-family: Arial; padding: 15px; background-color: #e8f4f8; border-radius: 5px; margin-bottom: 10px;">
        <h3 style="margin-top: 0; color: #2c3e50;">📊 Interactive Stock Price Viewer</h3>
        <p><strong>Total Stocks Available:</strong> {len(all_tickers)}</p>
        <p><strong>How to Use:</strong></p>
        <ul>
            <li>Select a stock from the dropdown above</li>
            <li>Green triangles (▲) show when we bought the stock</li>
            <li>Red circles (●) show when we sold the stock</li>
            <li>Hover over the chart to see exact prices and dates</li>
            <li>Use zoom and pan tools in the toolbar</li>
        </ul>
        <p><strong>Available Stocks:</strong> {len(all_tickers)} S&P 500 companies from 2018-2024</p>
    </div>
    """
    
    info_div = Div(text=info_html, width=1400)
    
    layout = column(info_div, select, p)
    
    return layout


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate all performance metrics"""
    
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
    
    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
    max_drawdown = drawdown.min()
    
    daily_returns = portfolio_df['value'].pct_change().dropna()
    
    rf_rate = 0.02
    excess_returns = daily_returns - (rf_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['month'] = portfolio_df_copy.index.to_period('M')
    monthly_returns = portfolio_df_copy.groupby('month')['value'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 0 else 0
    )
    
    winning_months = (monthly_returns > 0).sum()
    losing_months = (monthly_returns < 0).sum()
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'best_day': daily_returns.max() * 100,
        'worst_day': daily_returns.min() * 100,
        'best_month': monthly_returns.max(),
        'worst_month': monthly_returns.min(),
        'winning_months': winning_months,
        'losing_months': losing_months,
        'total_days': len(portfolio_df)
    }


def create_portfolio_value_chart(portfolio_df, initial_capital, metrics):
    """Main PnL chart"""
    
    p = figure(x_axis_type="datetime", width=1400, height=500,
               title="Portfolio Value Over Time - Principal Investment Performance (PnL)")
    
    p.line(portfolio_df.index, portfolio_df['value'], 
           line_width=3, color='navy', alpha=0.8, legend_label='Portfolio Value')
    
    p.line(portfolio_df.index, [initial_capital] * len(portfolio_df),
           line_width=2, color='green', alpha=0.5, line_dash='dashed',
           legend_label=f'Initial Capital (${initial_capital:,.0f})')
    
    # Highlight COVID crash
    if len(portfolio_df) > 530:
        covid_date = portfolio_df.index[530]
        covid_end = portfolio_df.index[min(550, len(portfolio_df)-1)]
        covid_box = BoxAnnotation(left=covid_date, right=covid_end,
                                   fill_alpha=0.1, fill_color='red')
        p.add_layout(covid_box)
    
    # Highlight 2022 bear
    if len(portfolio_df) > 1010:
        bear_date = portfolio_df.index[1010]
        bear_end = portfolio_df.index[min(1130, len(portfolio_df)-1)]
        bear_box = BoxAnnotation(left=bear_date, right=bear_end,
                                 fill_alpha=0.1, fill_color='orange')
        p.add_layout(bear_box)
    
    p.yaxis.axis_label = "Portfolio Value ($)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "top_left"
    
    hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Value", "$@y{0,0}")],
                      formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    return p


def create_cumulative_returns_chart(portfolio_df, initial_capital):
    """Cumulative returns %"""
    
    cumulative_returns = (portfolio_df['value'] / initial_capital - 1) * 100
    
    p = figure(x_axis_type="datetime", width=700, height=400,
               title="Cumulative Returns (%)")
    
    p.line(portfolio_df.index, cumulative_returns,
           line_width=2, color='darkgreen', alpha=0.8)
    
    p.line(portfolio_df.index, [0] * len(portfolio_df),
           line_width=1, color='gray', alpha=0.5, line_dash='dashed')
    
    p.yaxis.axis_label = "Cumulative Return (%)"
    p.xaxis.axis_label = "Date"
    
    hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Return", "@y{0.1f}%")],
                      formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    return p


def create_daily_returns_histogram(portfolio_df):
    """Daily returns histogram"""
    
    daily_returns = portfolio_df['value'].pct_change().dropna() * 100
    
    hist, edges = np.histogram(daily_returns, bins=50)
    
    p = figure(width=700, height=400, title="Daily Returns Distribution")
    
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color='steelblue', line_color='white', alpha=0.7)
    
    mean_return = daily_returns.mean()
    p.line([mean_return, mean_return], [0, hist.max()],
           line_width=2, color='red', line_dash='dashed',
           legend_label=f'Mean: {mean_return:.2f}%')
    
    p.xaxis.axis_label = "Daily Return (%)"
    p.yaxis.axis_label = "Frequency"
    p.legend.location = "top_right"
    
    return p


def create_drawdown_chart(portfolio_df):
    """Drawdown chart"""
    
    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
    
    p = figure(x_axis_type="datetime", width=1400, height=400,
               title="Drawdown Analysis - How Much Below Peak (%)")
    
    p.line(portfolio_df.index, drawdown,
           line_width=2, color='red', alpha=0.8)
    
    p.varea(x=portfolio_df.index, y1=0, y2=drawdown,
            fill_color='red', fill_alpha=0.2)
    
    p.line(portfolio_df.index, [0] * len(portfolio_df),
           line_width=1, color='gray', alpha=0.5, line_dash='dashed')
    
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    
    p.scatter([max_dd_idx], [max_dd_val], size=10, color='darkred',
             legend_label=f'Max Drawdown: {max_dd_val:.1f}%')
    
    p.yaxis.axis_label = "Drawdown (%)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "bottom_left"
    
    hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Drawdown", "@y{0.1f}%")],
                      formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    return p


def create_monthly_returns_chart(portfolio_df):
    """Monthly returns bar chart"""
    
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['month'] = portfolio_df_copy.index.to_period('M')
    monthly_returns = portfolio_df_copy.groupby('month')['value'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 0 else 0
    )
    
    months = [str(m) for m in monthly_returns.index]
    returns = monthly_returns.values
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    p = figure(x_range=months, width=1400, height=400,
               title="Monthly Returns (%)")
    
    p.vbar(x=months, top=returns, width=0.8, color=colors, alpha=0.7)
    
    p.line(months, [0] * len(months), line_width=1, color='gray', line_dash='dashed')
    
    p.xaxis.axis_label = "Month"
    p.yaxis.axis_label = "Return (%)"
    p.xaxis.major_label_orientation = 0.785
    
    p.xaxis.major_label_overrides = {months[i]: months[i] if i % 6 == 0 else "" 
                                     for i in range(len(months))}
    
    return p


def create_metrics_summary(metrics, portfolio_df):
    """Metrics summary table"""
    
    start_date = portfolio_df.index[0].strftime('%Y-%m-%d')
    end_date = portfolio_df.index[-1].strftime('%Y-%m-%d')
    win_rate = metrics['winning_months']/(metrics['winning_months']+metrics['losing_months'])*100
    
    goal_status = 'GOAL ACHIEVED\!' if metrics['annual_return'] >= 20 else 'BELOW TARGET'
    goal_color = '#d4edda' if metrics['annual_return'] >= 20 else '#f8d7da'
    border_color = '#28a745' if metrics['annual_return'] >= 20 else '#dc3545'
    emoji = '🎯' if metrics['annual_return'] >= 20 else '⚠️'
    
    html = f"""
    <div style="font-family: Arial; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
        <h2 style="color: #2c3e50;">📊 Performance Metrics Summary</h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px;">
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #3498db; margin-top: 0;">💰 Capital</h3>
                <p><strong>Initial:</strong> ${metrics['initial_capital']:,.0f}</p>
                <p><strong>Final:</strong> ${metrics['final_value']:,.0f}</p>
                <p><strong>Change:</strong> ${metrics['final_value'] - metrics['initial_capital']:,.0f}</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #2ecc71; margin-top: 0;">📈 Returns</h3>
                <p><strong>Total:</strong> {metrics['total_return']:.1f}%</p>
                <p><strong>Annual:</strong> {metrics['annual_return']:.1f}%</p>
                <p><strong>Best Month:</strong> {metrics['best_month']:.1f}%</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #e74c3c; margin-top: 0;">⚠️ Risk</h3>
                <p><strong>Max Drawdown:</strong> {metrics['max_drawdown']:.1f}%</p>
                <p><strong>Volatility:</strong> {metrics['volatility']:.1f}%</p>
                <p><strong>Sharpe Ratio:</strong> {metrics['sharpe_ratio']:.2f}</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #9b59b6; margin-top: 0;">📊 Best/Worst</h3>
                <p><strong>Best Day:</strong> {metrics['best_day']:.2f}%</p>
                <p><strong>Worst Day:</strong> {metrics['worst_day']:.2f}%</p>
                <p><strong>Worst Month:</strong> {metrics['worst_month']:.1f}%</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #f39c12; margin-top: 0;">📅 Monthly Stats</h3>
                <p><strong>Winning:</strong> {metrics['winning_months']}</p>
                <p><strong>Losing:</strong> {metrics['losing_months']}</p>
                <p><strong>Win Rate:</strong> {win_rate:.1f}%</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #34495e; margin-top: 0;">⏱️ Timeline</h3>
                <p><strong>Total Days:</strong> {metrics['total_days']}</p>
                <p><strong>Start:</strong> {start_date}</p>
                <p><strong>End:</strong> {end_date}</p>
            </div>
            
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: {goal_color}; 
                    border-radius: 5px; border-left: 4px solid {border_color};">
            <h3 style="margin-top: 0;">{emoji} {goal_status}</h3>
            <p>Target: 20% annual return | Actual: {metrics['annual_return']:.1f}% annual return</p>
        </div>
    </div>
    """
    
    div = Div(text=html, width=1400, height=500)
    return div


# ============================================================================
# TAB 1: TRADING CHARTS
# ============================================================================

def track_trades_detailed(bot, top_n=10):
    """Track all trades during backtest"""
    
    all_dates = bot.stocks_data[list(bot.stocks_data.keys())[0]].index
    
    cash = bot.initial_capital
    holdings = {}
    trades = []
    rebalance_dates = all_dates[::21]
    
    for rebal_idx, current_date in enumerate(rebalance_dates):
        sell_trades = []
        for ticker in list(holdings.keys()):
            if ticker in bot.stocks_data:
                price = bot.stocks_data[ticker].loc[current_date, 'close']
                value = holdings[ticker] * price
                cash += value
                
                sell_trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': price,
                    'shares': holdings[ticker],
                    'value': value
                })
        
        trades.extend(sell_trades)
        holdings = {}
        
        top_stocks = [(ticker, score) for ticker, score in bot.rankings[:top_n]
                     if ticker in bot.stocks_data]
        
        invest_amount = cash * 0.80
        if len(top_stocks) > 0:
            per_stock = invest_amount / len(top_stocks)
            
            for ticker, score in top_stocks:
                price = bot.stocks_data[ticker].loc[current_date, 'close']
                shares = per_stock / price
                holdings[ticker] = shares
                cash -= per_stock
                
                trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares,
                    'value': per_stock,
                    'score': score
                })
        
        trades.append({
            'date': current_date,
            'ticker': 'PORTFOLIO',
            'action': 'HOLDINGS',
            'holdings': list(holdings.keys()),
            'cash': cash,
            'rebalance_num': rebal_idx
        })
    
    return trades


def create_portfolio_composition_chart(trades_log, bot):
    """Portfolio composition over time"""
    
    holdings_over_time = [t for t in trades_log if t.get('action') == 'HOLDINGS']
    
    dates = [h['date'] for h in holdings_over_time]
    all_tickers = set()
    for h in holdings_over_time:
        all_tickers.update(h['holdings'])
    
    ticker_values = {ticker: [] for ticker in all_tickers}
    
    for h in holdings_over_time:
        current_holdings = h['holdings']
        total_value = 0
        
        values = {}
        for ticker in current_holdings:
            if ticker in bot.stocks_data:
                price = bot.stocks_data[ticker].loc[h['date'], 'close']
                buy_trade = next((t for t in trades_log 
                                if t.get('ticker') == ticker 
                                and t.get('action') == 'BUY' 
                                and t.get('date') == h['date']), None)
                if buy_trade:
                    values[ticker] = buy_trade['value']
                    total_value += buy_trade['value']
        
        for ticker in all_tickers:
            if ticker in values:
                ticker_values[ticker].append(values[ticker] / total_value * 100 if total_value > 0 else 0)
            else:
                ticker_values[ticker].append(0)
    
    p = figure(x_axis_type="datetime", width=1200, height=400,
               title="Portfolio Composition Over Time (Top 10 Holdings)",
               toolbar_location="above")
    
    colors = Category20_20[:len(all_tickers)]
    
    for i, (ticker, color) in enumerate(zip(sorted(all_tickers), colors)):
        if i == 0:
            base = [0] * len(dates)
        else:
            base = [sum(ticker_values[t][j] for t in sorted(all_tickers)[:i]) 
                   for j in range(len(dates))]
        
        top = [base[j] + ticker_values[ticker][j] for j in range(len(dates))]
        
        p.varea(x=dates, y1=base, y2=top, alpha=0.7, color=color, legend_label=ticker)
    
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.yaxis.axis_label = "Portfolio Allocation (%)"
    p.xaxis.axis_label = "Date"
    
    return p


def create_stock_price_chart(ticker, trades_log, bot):
    """Stock price with entry/exit markers"""
    
    df = bot.stocks_data[ticker]
    
    stock_trades = [t for t in trades_log if t.get('ticker') == ticker and t.get('action') in ['BUY', 'SELL']]
    
    buy_dates = [t['date'] for t in stock_trades if t['action'] == 'BUY']
    buy_prices = [t['price'] for t in stock_trades if t['action'] == 'BUY']
    
    sell_dates = [t['date'] for t in stock_trades if t['action'] == 'SELL']
    sell_prices = [t['price'] for t in stock_trades if t['action'] == 'SELL']
    
    p = figure(x_axis_type="datetime", width=600, height=400,
               title=f"{ticker} - Price with Entry/Exit Points")
    
    p.line(df.index, df['close'], line_width=2, color='navy', alpha=0.8)
    
    if buy_dates:
        p.scatter(buy_dates, buy_prices, size=12, color='green', alpha=0.8, 
                 marker='triangle', legend_label='Buy')
    
    if sell_dates:
        p.scatter(sell_dates, sell_prices, size=10, color='red', alpha=0.8, legend_label='Sell')
    
    p.yaxis.axis_label = "Price ($)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "top_left"
    
    hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Price", "$@y{0.2f}")],
                      formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    return p


def create_holdings_timeline(trades_log):
    """Holdings timeline"""
    
    holdings_events = [t for t in trades_log if t.get('action') == 'HOLDINGS']
    
    timeline_data = []
    for i, event in enumerate(holdings_events):
        for ticker in event['holdings']:
            timeline_data.append({
                'ticker': ticker,
                'rebalance': i,
                'date': event['date']
            })
    
    df = pd.DataFrame(timeline_data)
    
    tickers = sorted(df['ticker'].unique())
    
    p = figure(width=1200, height=600, 
               title="Stock Holdings Timeline (Which stocks held at each rebalance)",
               y_range=tickers,
               x_axis_type="datetime")
    
    colors = Viridis256
    color_mapper = {ticker: colors[int(i * len(colors) / len(tickers))] 
                   for i, ticker in enumerate(tickers)}
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        y = [ticker] * len(ticker_data)
        p.scatter(ticker_data['date'], y, size=8, color=color_mapper[ticker], alpha=0.6)
    
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Stock"
    
    return p


def create_trade_frequency_chart(trades_log):
    """Trade frequency chart"""
    
    buy_trades = [t for t in trades_log if t.get('action') == 'BUY']
    
    trade_counts = {}
    for trade in buy_trades:
        ticker = trade['ticker']
        trade_counts[ticker] = trade_counts.get(ticker, 0) + 1
    
    sorted_trades = sorted(trade_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    tickers = [t[0] for t in sorted_trades]
    counts = [t[1] for t in sorted_trades]
    
    p = figure(x_range=tickers, width=1200, height=400,
               title="Top 20 Most Frequently Traded Stocks")
    
    p.vbar(x=tickers, top=counts, width=0.8, color='steelblue', alpha=0.8)
    
    p.xaxis.axis_label = "Stock"
    p.yaxis.axis_label = "Number of Times Bought"
    p.xaxis.major_label_orientation = 0.785
    
    return p


def get_most_traded_stocks(trades_log, top_n=6):
    """Get most traded stocks"""
    
    buy_trades = [t for t in trades_log if t.get('action') == 'BUY']
    
    trade_counts = {}
    for trade in buy_trades:
        ticker = trade['ticker']
        trade_counts[ticker] = trade_counts.get(ticker, 0) + 1
    
    sorted_trades = sorted(trade_counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_trades[:top_n]]


if __name__ == "__main__":
    create_trade_visualizations()
