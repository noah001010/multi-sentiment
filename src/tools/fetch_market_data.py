import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

def fetch_usd_jpy(date_str, output_path):
    """
    Fetch USD/JPY 1m data for a specific date (JST).
    date_str: YYYY-MM-DD
    """
    # yfinance uses UTC for start/end if not specified.
    # BoJ conferences are 15:30 JST = 06:30 UTC.
    # We fetch a window around that.
    
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Fetch 2 days to be safe with timezones
    start_date = date_obj.strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Fetching USD/JPY for {date_str}...")
    ticker = "JPY=X" # USD/JPY in yfinance
    
    # 1m data is only available for the last 7 days normally, 
    # but for historical, we might need 2m or 5m if 1m is not available.
    # However, for 2023, yfinance 1m might be unavailable.
    # Let's try 1m first, fallback to 2m, 5m.
    
    for interval in ["1m", "2m", "5m"]:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if not data.empty:
            print(f"Success with interval {interval}")
            break
    
    if data.empty:
        print("Error: No data found.")
        return
    
    # Convert index to JST
    data.index = data.index.tz_convert("Asia/Tokyo")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"Market data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2023-06-16", help="Date in YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="output/market_data.csv", help="Output CSV path")
    args = parser.parse_args()
    
    fetch_usd_jpy(args.date, args.output)
