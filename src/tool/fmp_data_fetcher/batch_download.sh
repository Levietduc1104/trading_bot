#!/bin/bash
# Batch download using curl with proxy bypass

API_KEY="yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"

# Disable proxy
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""

echo "================================================================================"
echo "BATCH DOWNLOAD - MISSING STOCKS"
echo "================================================================================"

count=0
success=0
total=$(wc -l < missing_stocks.txt)

echo "Total to download: $total"
echo "================================================================================"

while IFS= read -r symbol; do
    ((count++))

    echo -n "[$count/$total] $symbol... "

    # Download with curl
    result=$(curl -s --noproxy "*" "https://financialmodelingprep.com/stable/profile?symbol=${symbol}&apikey=${API_KEY}")

    # Check if we got valid JSON
    if echo "$result" | grep -q "mktCap"; then
        # Extract market cap and show
        mktcap=$(echo "$result" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['mktCap']/1e9 if d and len(d)>0 else 0)" 2>/dev/null)
        echo "✅ \$${mktcap}B"

        # Append to CSV
        echo "$result" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data and len(data) > 0:
        s = data[0]
        print(f\"{s.get('symbol','')},{s.get('price','')},{s.get('mktCap','')},{s.get('companyName','').replace(',', ' ')},{s.get('sector','')},{s.get('exchange','')}\")
except: pass
" >> new_downloads.csv
        ((success++))
    else
        echo "❌"
    fi

    # Progress save every 50
    if [ $((count % 50)) -eq 0 ]; then
        echo "    💾 Progress: $success/$count successful"
    fi

    # Rate limit
    sleep 0.2

done < missing_stocks.txt

echo "================================================================================"
echo "✅ Download complete: $success/$total successful"
echo "================================================================================"

# Merge with existing data
if [ -f "new_downloads.csv" ] && [ $success -gt 0 ]; then
    python3 << 'EOF'
import pandas as pd

# Load existing
existing = pd.read_csv('current_market_cap_progress.csv')

# Load new downloads
new = pd.read_csv('new_downloads.csv', names=['symbol','current_price','current_market_cap','company_name','sector','exchange'])

# Combine
combined = pd.concat([existing, new], ignore_index=True)
combined = combined.drop_duplicates(subset=['symbol'], keep='last')

# Save
combined.to_csv('current_market_cap_progress.csv', index=False)

print(f"\n✅ Merged: {len(combined)} total stocks")

# Check Mag 7
mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
have = [s for s in mag7 if s in combined['symbol'].values]
missing = [s for s in mag7 if s not in combined['symbol'].values]

print(f"\nMAG 7 STATUS:")
print(f"✅ Have: {have}")
if missing:
    print(f"❌ Missing: {missing}")
else:
    print("🎉 ALL MAG 7 AVAILABLE!")
EOF
fi

echo ""
echo "📊 Next step: python3 calculate_historical_market_cap_200.py"
