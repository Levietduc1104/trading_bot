"""
FMP Ultimate Plan - Comprehensive Historical Data Downloader
============================================================
Downloads all premium/historical data needed for ML features:
  - Earnings surprises (actual vs estimated EPS)
  - Analyst estimates (forward EPS/revenue consensus)
  - Analyst upgrades/downgrades
  - Insider trading transactions
  - Income statements (quarterly, full history)
  - Balance sheets (quarterly, full history)
  - Cash flow statements (quarterly, full history)

Key design choices:
  - Bypasses system proxy (required for FMP access)
  - Per-symbol JSON files for easy resume / partial download
  - Unified progress tracking in download_progress.json
  - Skips already-downloaded symbols automatically
  - Rate limiting: ~0.25s delay → ~4 req/sec (well within Ultimate limits)

Usage:
  python3 download_fmp_ultimate.py                  # download all data types
  python3 download_fmp_ultimate.py --types earnings analyst insider income balance cashflow
  python3 download_fmp_ultimate.py --types earnings --symbols AAPL MSFT GOOGL
  python3 download_fmp_ultimate.py --reset           # clear progress, re-download all
"""

import os
import json
import time
import argparse
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
FMP_API_KEY  = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"
BASE_STABLE  = "https://financialmodelingprep.com/stable"
OUTPUT_DIR   = "sp500_data/fmp_fundamentals_premium"
PROGRESS_FILE = f"{OUTPUT_DIR}/download_progress.json"
SYMBOLS_FILE = "sp500_data/sp500_official_list.json"

ALL_TYPES = ["earnings", "analyst_estimates", "income", "balance", "cashflow"]

# ── HTTP session — use system proxy (required for DNS in this environment) ────
def make_session():
    s = requests.Session()
    # trust_env=True (default) so the system proxy is used for DNS + routing
    # The proxy allowlist must include financialmodelingprep.com
    return s

SESSION = make_session()

def fmp_get(url, params=None, retries=3, delay=2.0):
    """GET with automatic retry. Uses system proxy for DNS resolution."""
    p = params or {}
    p['apikey'] = FMP_API_KEY
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=p, timeout=20)
            if r.status_code == 200:
                data = r.json()
                # FMP returns {"Error Message": "..."} on plan restriction
                if isinstance(data, dict) and 'Error Message' in data:
                    print(f"  [API error] {data['Error Message'][:80]}")
                    return []
                return data
            elif r.status_code == 429:
                print(f"  [429 rate limit] waiting {delay*2}s...")
                time.sleep(delay * 2)
            else:
                print(f"  [HTTP {r.status_code}]")
                return []
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"  [ConnectionError] {e}")
                return []
        except Exception as e:
            print(f"  [Exception] {e}")
            return []
    return []

# ── Symbol loading ─────────────────────────────────────────────────────────────
def load_symbols():
    with open(SYMBOLS_FILE) as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [d['symbol'] for d in data if 'symbol' in d]
    elif isinstance(data, dict) and 'tickers' in data:
        return data['tickers']
    return data

# ── Progress tracking ─────────────────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {t: [] for t in ALL_TYPES}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

# ── Per-data-type fetchers ────────────────────────────────────────────────────

def fetch_earnings_surprises(symbol):
    """Actual vs estimated EPS per quarter — strong forward-looking signal.
    New stable endpoint: /stable/earnings (includes epsActual + epsEstimated)
    """
    data = fmp_get(f"{BASE_STABLE}/earnings", {'symbol': symbol, 'limit': 80})
    # Only keep records that have actual results (not future estimates)
    if isinstance(data, list):
        return [r for r in data if r.get('epsActual') is not None]
    return []

def fetch_analyst_estimates(symbol, limit=40):
    """Forward EPS/revenue consensus estimates per quarter."""
    data = fmp_get(f"{BASE_STABLE}/analyst-estimates", {'symbol': symbol, 'limit': limit, 'period': 'quarter'})
    return data if isinstance(data, list) else []

def fetch_analyst_upgrades(symbol):
    """Analyst rating upgrades/downgrades — momentum signal."""
    data = fmp_get(f"{BASE_STABLE}/upgrades-downgrades", {'symbol': symbol, 'limit': 200})
    return data if isinstance(data, list) else []

def fetch_insider_trading(symbol, limit=500):
    """Insider buy/sell transactions — insider sentiment signal."""
    data = fmp_get(f"{BASE_STABLE}/insider-trading", {'symbol': symbol, 'limit': limit})
    return data if isinstance(data, list) else []

def fetch_income_statement(symbol, limit=80):
    """Quarterly income statement — revenue, earnings, margins."""
    data = fmp_get(f"{BASE_STABLE}/income-statement", {'symbol': symbol, 'period': 'quarter', 'limit': limit})
    return data if isinstance(data, list) else []

def fetch_balance_sheet(symbol, limit=80):
    """Quarterly balance sheet — debt, assets, equity."""
    data = fmp_get(f"{BASE_STABLE}/balance-sheet-statement", {'symbol': symbol, 'period': 'quarter', 'limit': limit})
    return data if isinstance(data, list) else []

def fetch_cashflow(symbol, limit=80):
    """Quarterly cash flow — free cash flow, capex, buybacks."""
    data = fmp_get(f"{BASE_STABLE}/cash-flow-statement", {'symbol': symbol, 'period': 'quarter', 'limit': limit})
    return data if isinstance(data, list) else []

FETCHER_MAP = {
    "earnings":          (fetch_earnings_surprises,  "earnings_surprises"),
    "analyst_estimates": (fetch_analyst_estimates,   "analyst_estimates"),
    "income":            (fetch_income_statement,    "income_statements"),
    "balance":           (fetch_balance_sheet,       "balance_sheets"),
    "cashflow":          (fetch_cashflow,             "cashflow_statements"),
}

# ── Main download logic ───────────────────────────────────────────────────────

def download_type(data_type, symbols, progress, delay=0.25):
    """Download one data type for all symbols, skipping already done."""
    fetcher_fn, folder_name = FETCHER_MAP[data_type]
    out_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    done_set = set(progress.get(data_type, []))
    todo = [s for s in symbols if s not in done_set]

    print(f"\n{'='*70}")
    print(f"  {data_type.upper()} → {out_dir}")
    print(f"  Total: {len(symbols)}  |  Done: {len(done_set)}  |  Remaining: {len(todo)}")
    print(f"{'='*70}")

    if not todo:
        print("  Already complete — skipping.")
        return

    new_done = 0
    for i, symbol in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {symbol}...", end=" ", flush=True)

        data = fetcher_fn(symbol)
        out_file = os.path.join(out_dir, f"{symbol}.json")

        if data:
            with open(out_file, 'w') as f:
                json.dump(data, f)
            print(f"✓ {len(data)} records")
        else:
            # Save empty file so we don't retry (may be genuinely no data)
            with open(out_file, 'w') as f:
                json.dump([], f)
            print("- no data")

        # Mark done and persist every 10 symbols
        progress[data_type].append(symbol)
        new_done += 1
        if new_done % 10 == 0:
            save_progress(progress)

        if i < len(todo):
            time.sleep(delay)

    save_progress(progress)
    print(f"\n  Done: {new_done} new symbols downloaded for {data_type}")


def consolidate_to_json(data_type):
    """
    Merge all per-symbol JSON files into one combined dict file.
    Output: sp500_data/fmp_fundamentals_premium/{folder_name}.json
    """
    _, folder_name = FETCHER_MAP[data_type]
    in_dir  = os.path.join(OUTPUT_DIR, folder_name)
    out_file = os.path.join(OUTPUT_DIR, f"{folder_name}.json")

    combined = {}
    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith('.json'):
            continue
        symbol = fname.replace('.json', '')
        with open(os.path.join(in_dir, fname)) as f:
            data = json.load(f)
        if data:
            combined[symbol] = data

    with open(out_file, 'w') as f:
        json.dump(combined, f)

    total_records = sum(len(v) for v in combined.values())
    print(f"  Consolidated {len(combined)} symbols / {total_records} records → {out_file}")
    return combined


def main():
    parser = argparse.ArgumentParser(description='FMP Ultimate — bulk historical data downloader')
    parser.add_argument('--types', nargs='+', default=ALL_TYPES,
                        choices=ALL_TYPES,
                        help='Data types to download (default: all)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Specific symbols (default: all 466 S&P 500)')
    parser.add_argument('--delay', type=float, default=0.25,
                        help='Seconds between requests (default: 0.25)')
    parser.add_argument('--reset', action='store_true',
                        help='Clear progress file and re-download everything')
    parser.add_argument('--consolidate-only', action='store_true',
                        help='Just re-consolidate per-symbol files into combined JSONs')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = load_symbols()
    print(f"\nFMP Ultimate Data Downloader")
    print(f"API key : {FMP_API_KEY[:12]}...")
    print(f"Symbols : {len(symbols)}")
    print(f"Types   : {args.types}")
    print(f"Output  : {OUTPUT_DIR}/")

    # Reset if requested
    if args.reset:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        print("Progress reset.")

    progress = load_progress()

    if args.consolidate_only:
        print("\nConsolidating per-symbol files...")
        for dt in args.types:
            consolidate_to_json(dt)
        return

    # Download
    start = datetime.now()
    for dt in args.types:
        download_type(dt, symbols, progress, delay=args.delay)

    # Consolidate all into combined JSON files
    print(f"\n{'='*70}")
    print("  Consolidating per-symbol files...")
    print(f"{'='*70}")
    for dt in args.types:
        consolidate_to_json(dt)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nAll done in {elapsed/60:.1f} min.")
    print(f"Data saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
