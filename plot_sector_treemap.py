"""
Portfolio Sector Treemap
=========================
Generates an S&P-500-style sector treemap of the current (or last backtest)
portfolio holdings.

  - Rectangles sized by dollar allocation
  - Coloured by return since entry (red = loss, green = gain)
  - Grouped by sector with sector label headers
  - Sub-labels show ticker + return %

Usage:
  python plot_sector_treemap.py                        # last rebalance from v32 trades
  python plot_sector_treemap.py --trades output/v32_sector_balanced_2015_2024.csv
  python plot_sector_treemap.py --date 2022-01-10      # snapshot on specific date
  python plot_sector_treemap.py --strategy v31         # use v31 trades instead
"""

import os
import sys
import argparse
import json
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import RdYlGn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SECTOR_COLORS = {
    'Technology':       '#1a1a2e',
    'Financials':       '#16213e',
    'Healthcare':       '#0f3460',
    'Consumer Disc.':   '#533483',
    'Consumer Staples': '#2b4162',
    'Industrials':      '#1b4332',
    'Energy':           '#7c3626',
    'Comm. Services':   '#3d405b',
    'Real Estate':      '#2d4739',
    'Materials':        '#4a3728',
    'Utilities':        '#2c3e50',
}

OUTPUT_PATH = 'output/sector_treemap.png'


# ── Squarified treemap layout (pure Python / numpy) ───────────────────────────

def _squarify(values: List[float], x: float, y: float, w: float, h: float
              ) -> List[Tuple[float, float, float, float]]:
    """
    Squarified treemap layout algorithm.
    Returns list of (x, y, width, height) for each value, in input order.
    Values must be pre-normalised to fill the w*h area.
    """
    if not values:
        return []
    if len(values) == 1:
        return [(x, y, w, h)]

    total = sum(values)
    rects = []

    def _worst_ratio(row, width):
        s = sum(row)
        return max(
            max(width * width * v / (s * s), s * s / (width * width * v))
            for v in row
        )

    def _layout_row(row, x, y, w, h):
        s = sum(row)
        if w >= h:
            rw = s / h if h > 0 else 0
            rects_row = []
            cy = y
            for v in row:
                rh = v / rw if rw > 0 else 0
                rects_row.append((x, cy, rw, rh))
                cy += rh
            return rects_row, x + rw, y, w - rw, h
        else:
            rh = s / w if w > 0 else 0
            rects_row = []
            cx = x
            for v in row:
                rw = v / rh if rh > 0 else 0
                rects_row.append((cx, y, rw, rh))
                cx += rw
            return rects_row, x, y + rh, w, h - rh

    remaining = list(values)
    while remaining:
        if len(remaining) == 1:
            rects.append((x, y, w, h))
            break
        row = [remaining[0]]
        i = 1
        while i < len(remaining):
            candidate = row + [remaining[i]]
            if len(row) > 1 and _worst_ratio(candidate, min(w, h)) > _worst_ratio(row, min(w, h)):
                break
            row = candidate
            i += 1
        row_rects, x, y, w, h = _layout_row(row, x, y, w, h)
        rects.extend(row_rects)
        remaining = remaining[i:]

    return rects


# ── build portfolio snapshot from trade CSV ───────────────────────────────────

def _load_portfolio_snapshot(trades_path: str, date_str: str = None,
                              sector_map: Dict[str, str] = None
                              ) -> List[Dict]:
    """
    Reconstruct portfolio holdings on `date_str` from a trades CSV.
    Returns list of {ticker, sector, allocation, entry_price, current_price, return_pct}
    """
    df = pd.read_csv(trades_path, parse_dates=['date'])
    df = df.sort_values('date')

    if date_str:
        cutoff = pd.Timestamp(date_str)
        df = df[df['date'] <= cutoff]

    # Find the last rebalance date
    rebal_sells = df[df['action'] == 'SELL']
    if rebal_sells.empty:
        last_rebal = df['date'].min()
    else:
        last_rebal = rebal_sells['date'].max()

    # Holdings = BUYs on or after last rebalance
    buys = df[(df['date'] >= last_rebal) & (df['action'] == 'BUY')]
    # Subtract any trailing-stop sells after that
    stops = df[(df['date'] >= last_rebal) & (df['action'] == 'SELL') &
               (df['reason'] == 'trailing_stop')]
    stopped_tickers = set(stops['ticker'])

    holdings = []
    for _, row in buys.iterrows():
        ticker = row['ticker']
        if ticker in stopped_tickers:
            continue
        holdings.append({
            'ticker':       ticker,
            'sector':       (sector_map or {}).get(ticker, 'Unknown'),
            'allocation':   float(row['value']),
            'entry_price':  float(row['price']),
            'current_price': float(row['price']),  # placeholder — updated below
            'return_pct':   0.0,
            'date':         row['date'],
        })

    return holdings, last_rebal


def _enrich_with_prices(holdings: List[Dict], snapshot_date: str) -> List[Dict]:
    """Try to load current prices from the stock data directory."""
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    try:
        bot = PortfolioRotationBot(
            data_dir='sp500_data/stock_data_1990_2024',
            initial_capital=100_000,
        )
        bot.load_all_stocks()
        ts = pd.Timestamp(snapshot_date)
        for h in holdings:
            ticker = h['ticker']
            if ticker in bot.stocks_data:
                df = bot.stocks_data[ticker]
                df_at = df[df.index <= ts]
                if not df_at.empty:
                    h['current_price'] = float(df_at.iloc[-1]['close'])
                    ep = h['entry_price']
                    if ep > 0:
                        h['return_pct'] = (h['current_price'] / ep - 1) * 100
    except Exception as e:
        print(f"  Warning: could not load prices for enrichment: {e}")
    return holdings


# ── drawing ───────────────────────────────────────────────────────────────────

def _draw_treemap(holdings: List[Dict], title: str, save_path: str):
    """Draw the sector treemap and save to save_path."""
    if not holdings:
        print("  No holdings to plot.")
        return

    # Group by sector
    by_sector: Dict[str, List[Dict]] = defaultdict(list)
    for h in holdings:
        by_sector[h['sector']].append(h)

    # Sort sectors by total allocation (largest first)
    sector_totals = {s: sum(h['allocation'] for h in hs)
                     for s, hs in by_sector.items()}
    sectors_sorted = sorted(sector_totals, key=sector_totals.get, reverse=True)

    # Normalise sector sizes to fill canvas
    total_alloc = sum(sector_totals.values())
    sector_sizes = [sector_totals[s] / total_alloc for s in sectors_sorted]

    # Canvas
    fig_w, fig_h = 20, 11
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    # Colour normalisation for returns
    all_returns = [h['return_pct'] for h in holdings]
    vmin = max(min(all_returns), -20)
    vmax = min(max(all_returns),  20)
    if vmin == vmax:
        vmin, vmax = -5, 5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = RdYlGn

    # Sector-level layout
    sector_areas = [s * fig_w * fig_h for s in sector_sizes]
    sector_rects = _squarify(sector_areas, 0, 0, fig_w, fig_h)

    for sector, (sx, sy, sw, sh), area in zip(sectors_sorted, sector_rects, sector_areas):
        sector_holdings = sorted(by_sector[sector],
                                 key=lambda h: h['allocation'], reverse=True)
        n = len(sector_holdings)

        # Draw sector background
        bg = SECTOR_COLORS.get(sector, '#1a1a1a')
        bg_rect = mpatches.FancyBboxPatch(
            (sx + 0.02, sy + 0.02), sw - 0.04, sh - 0.04,
            boxstyle='round,pad=0.01',
            linewidth=1.5, edgecolor='#333333', facecolor=bg,
        )
        ax.add_patch(bg_rect)

        # Sector label at top
        label_h = min(0.45, sh * 0.12)
        ax.text(sx + sw * 0.5, sy + sh - label_h * 0.5,
                sector.upper(), ha='center', va='center',
                fontsize=max(5.5, min(9, sw * 0.55)),
                fontweight='bold', color='#cccccc',
                clip_on=True)

        # Stock-level layout within sector (leave top strip for sector label)
        inner_y      = sy + 0.04
        inner_h      = sh - label_h - 0.06
        inner_x      = sx + 0.04
        inner_w      = sw - 0.08

        if inner_h <= 0 or inner_w <= 0:
            continue

        stock_allocs = [h['allocation'] for h in sector_holdings]
        stock_areas  = [a / total_alloc * fig_w * fig_h
                        for a in stock_allocs]
        stock_rects  = _squarify(stock_areas, inner_x, inner_y, inner_w, inner_h)

        for h, (tx, ty, tw, th) in zip(sector_holdings, stock_rects):
            ret = h['return_pct']
            face = cmap(norm(ret))

            # Stock rectangle
            pad = 0.015
            stock_patch = mpatches.FancyBboxPatch(
                (tx + pad, ty + pad), max(tw - 2*pad, 0.01), max(th - 2*pad, 0.01),
                boxstyle='round,pad=0.005',
                linewidth=0.8, edgecolor='#111111', facecolor=face,
            )
            ax.add_patch(stock_patch)

            # Ticker label
            cx, cy = tx + tw / 2, ty + th / 2
            area_px = tw * th
            if area_px > 0.15:
                fsize_ticker = max(5, min(16, tw * 2.2))
                ax.text(cx, cy + th * 0.08, h['ticker'],
                        ha='center', va='center',
                        fontsize=fsize_ticker, fontweight='bold',
                        color='white', clip_on=True)
                if area_px > 0.3:
                    sign = '+' if ret >= 0 else ''
                    ax.text(cx, cy - th * 0.18,
                            f"{sign}{ret:.2f}%",
                            ha='center', va='center',
                            fontsize=max(4.5, fsize_ticker * 0.68),
                            color='white', alpha=0.9, clip_on=True)

    # Title
    fig.text(0.5, 0.985, title, ha='center', va='top',
             fontsize=13, fontweight='bold', color='white')

    # Colour scale legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.03, 0.10, 0.015])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, colors='white')
    cb.outline.set_edgecolor('#555555')
    cbar_ax.set_xlabel('Return %', fontsize=7, color='white')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Treemap saved: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def run(trades_path: str = None, date_str: str = None, strategy: str = 'v32_balanced'):
    # Default trade file
    if trades_path is None:
        candidates = {
            'v32_balanced': 'output/v32_balanced_trades.csv',
            'v32_loose':    'output/v32_loose_trades.csv',
            'v32_equal':    'output/v32_equal_trades.csv',
            'v31':          'output/ml_trades.csv',
        }
        trades_path = candidates.get(strategy, 'output/v32_balanced_trades.csv')

    if not os.path.exists(trades_path):
        print(f"  Trade file not found: {trades_path}")
        print("  Run backtest_sector_diversified.py first.")
        return

    # Load sector map
    from src.strategies.v32_sector_diversified import _load_sector_map
    sector_map = _load_sector_map('sp500_data/metadata')

    print(f"  Reading trades: {trades_path}")
    holdings, last_rebal = _load_portfolio_snapshot(trades_path, date_str, sector_map)

    if not holdings:
        print("  No holdings found in trade file.")
        return

    snapshot_date = date_str or pd.Timestamp.now().strftime('%Y-%m-%d')
    print(f"  Portfolio snapshot: {len(holdings)} positions as of {snapshot_date}")
    print(f"  Last rebalance: {last_rebal.date()}")

    # Enrich with current prices
    holdings = _enrich_with_prices(holdings, snapshot_date)

    # Print table
    print(f"\n  {'Ticker':<8} {'Sector':<18} {'Alloc $':>10} {'Entry':>8} {'Current':>9} {'Return':>8}")
    print(f"  {'-'*65}")
    for h in sorted(holdings, key=lambda x: -x['allocation']):
        sign = '+' if h['return_pct'] >= 0 else ''
        print(f"  {h['ticker']:<8} {h['sector']:<18} "
              f"${h['allocation']:>9,.0f} "
              f"${h['entry_price']:>7.2f} "
              f"${h['current_price']:>8.2f} "
              f"{sign}{h['return_pct']:>6.2f}%")

    title = (f"Portfolio Sector Map — {strategy.upper()}  |  "
             f"Last rebalance: {last_rebal.date()}  |  "
             f"{len(holdings)} positions")
    save_path = f"output/sector_treemap_{strategy}.png"
    _draw_treemap(holdings, title, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio Sector Treemap')
    parser.add_argument('--trades',    default=None,
                        help='Path to trades CSV (default: v32_balanced_trades.csv)')
    parser.add_argument('--date',      default=None,
                        help='Snapshot date YYYY-MM-DD (default: latest)')
    parser.add_argument('--strategy',  default='v32_balanced',
                        choices=['v32_loose', 'v32_balanced', 'v32_equal', 'v31'],
                        help='Strategy to visualise')
    args = parser.parse_args()
    run(trades_path=args.trades, date_str=args.date, strategy=args.strategy)
