"""
Debug version - show actual errors
"""
import requests
import pandas as pd

FMP_API_KEY = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"

# Test just one stock with full error details
symbol = "MSFT"
url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_API_KEY}"

print("="*80)
print("DEBUG: Testing single stock download")
print("="*80)
print(f"Symbol: {symbol}")
print(f"URL: {url[:80]}...")
print()

# Try different methods
print("Method 1: requests with trust_env=False")
try:
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, timeout=10, proxies={'http': None, 'https': None})
    print(f"✅ Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")
    data = response.json()
    if data and len(data) > 0:
        print(f"✅ Market Cap: ${data[0]['mktCap']/1e9:.1f}B")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")

print("\n" + "="*80)
print("Method 2: requests with no proxy in session")
try:
    response = requests.get(url, timeout=10, proxies={"http": None, "https": None})
    print(f"✅ Status: {response.status_code}")
    data = response.json()
    if data and len(data) > 0:
        print(f"✅ Market Cap: ${data[0]['mktCap']/1e9:.1f}B")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")

print("\n" + "="*80)
print("Method 3: Check environment variables")
import os
print(f"http_proxy: {os.environ.get('http_proxy', 'Not set')}")
print(f"https_proxy: {os.environ.get('https_proxy', 'Not set')}")
print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'Not set')}")
print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")

print("\n" + "="*80)
print("Method 4: requests with explicit empty proxy dict")
try:
    response = requests.get(url, timeout=10, proxies={})
    print(f"✅ Status: {response.status_code}")
    data = response.json()
    if data and len(data) > 0:
        print(f"✅ Market Cap: ${data[0]['mktCap']/1e9:.1f}B")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")

print("\n" + "="*80)
