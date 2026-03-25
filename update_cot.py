"""
update_cot.py
-------------
Run this every Friday after 3:30 PM EST to update COT positioning data.
Downloads the latest CFTC Disaggregated COT report and updates the dashboard.

HOW TO RUN:
  python update_cot.py

SCHEDULE AUTOMATICALLY (Windows Task Scheduler):
  - Run every Friday at 4:00 PM EST
"""

import urllib.request
import zipfile
import csv
import os
import json
import pathlib
import re
from datetime import datetime

HERE = pathlib.Path(__file__).parent

# Markets we track — CFTC market names as they appear in the data
MARKET_NAMES = {
    'COTTON NO. 2 - ICE FUTURES U.S.':           'Cotton',
    'CORN - CHICAGO BOARD OF TRADE':              'Corn',
    'SOYBEANS - CHICAGO BOARD OF TRADE':          'Soybeans',
    'WHEAT-SRW - CHICAGO BOARD OF TRADE':         'SRW Wheat',
    'WHEAT-HRW - CHICAGO BOARD OF TRADE':         'HRW Wheat',
    'SUGAR NO. 11 - ICE FUTURES U.S.':            'Sugar',
    'CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE': 'WTI Crude',
    'GOLD - COMMODITY EXCHANGE INC.':             'Gold',
    'SILVER - COMMODITY EXCHANGE INC.':           'Silver',
    'LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE':  'Live Cattle',
    'LEAN HOGS - CHICAGO MERCANTILE EXCHANGE':    'Lean Hogs',
    'SOYBEAN OIL - CHICAGO BOARD OF TRADE':       'Soy Oil',
    'SOYBEAN MEAL - CHICAGO BOARD OF TRADE':      'Soy Meal',
    'COCOA - ICE FUTURES U.S.':                   'Cocoa',
    'COFFEE C - ICE FUTURES U.S.':                'Coffee',
}

def download_latest_cot():
    print("=" * 50)
    print("CFTC COT UPDATE")
    print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    year = datetime.now().year
    url  = f"https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip"
    out_zip = HERE / f'cot_{year}.zip'

    print(f"\nDownloading {year} COT data from CFTC...")
    try:
        urllib.request.urlretrieve(url, out_zip)
        print(f"✓ Downloaded → {out_zip.name}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual steps:")
        print(f"1. Go to: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm")
        print(f"2. Download: Disaggregated Futures Only — {year}")
        print(f"3. Save the zip file to this folder")
        return None

    # Extract CSV
    print("Extracting...")
    with zipfile.ZipFile(out_zip) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            print("✗ No CSV found in zip")
            return None
        z.extract(csv_files[0], HERE)
        csv_path = HERE / csv_files[0]
        print(f"✓ Extracted: {csv_files[0]}")

    return csv_path

def extract_latest_positions(csv_path):
    """Read the CSV and get the latest week's positions for our markets."""
    print(f"\nReading {csv_path.name}...")

    results = {}
    latest_dates = {}

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mkt = row.get('Market_and_Exchange_Names','').strip().upper()
            comm = MARKET_NAMES.get(mkt)
            if not comm:
                continue

            date_str = row.get('Report_Date_as_MM_DD_YYYY','').strip()
            oi       = int(row.get('Open_Interest_All', 0) or 0)

            # Track latest date per commodity
            if comm not in latest_dates or date_str > latest_dates[comm]:
                latest_dates[comm] = date_str

                mm_long  = int(row.get('M_Money_Positions_Long_ALL',  0) or 0)
                mm_short = int(row.get('M_Money_Positions_Short_ALL', 0) or 0)
                pm_long  = int(row.get('Prod_Merc_Positions_Long_ALL', 0) or 0)
                pm_short = int(row.get('Prod_Merc_Positions_Short_ALL',0) or 0)
                sw_long  = int(row.get('Swap_Positions_Long_All',  0) or 0)
                sw_short = int(row.get('Swap__Positions_Short_All',0) or 0)
                or_long  = int(row.get('Other_Rept_Positions_Long_ALL', 0) or 0)
                or_short = int(row.get('Other_Rept_Positions_Short_ALL',0) or 0)

                mm_net = mm_long - mm_short
                mm_net_pct  = round(mm_net / oi * 100, 1) if oi else 0
                mm_long_pct = round(mm_long / oi * 100, 1) if oi else 0
                mm_short_pct= round(mm_short/ oi * 100, 1) if oi else 0

                results[comm] = {
                    'date': date_str,
                    'oi':   oi,
                    'mm_net': mm_net,
                    'mm_long': mm_long,
                    'mm_short': mm_short,
                    'mm_net_pct': mm_net_pct,
                    'mm_long_pct': mm_long_pct,
                    'mm_short_pct': mm_short_pct,
                    'prod_net': pm_long - pm_short,
                    'swap_net': sw_long - sw_short,
                    'other_net': or_long - or_short,
                }

    return results

def update_app_with_new_cot(positions):
    """Patch the BCOM dict in app.py with fresh contract counts."""
    if not positions:
        print("✗ No positions data to update")
        return

    print("\n✓ Latest COT positions:")
    for comm, p in sorted(positions.items()):
        print(f"  {comm:12s}  date={p['date']}  OI={p['oi']:,}  MM_net={p['mm_net']:+,}  ({p['mm_net_pct']:+.1f}% OI)")

    # Update the BCOM dict's mm_net, mm_long, mm_short, prod_net, swap_net, other_net
    app_file = HERE / 'app.py'
    with open(app_file) as f:
        src = f.read()

    for comm, p in positions.items():
        # Pattern: 'Cotton': {'gs':99.5, ..., 'mm_net':-72937, 'mm_long':41093, ...}
        # We update just the contract count fields
        pattern = re.compile(
            rf"('{re.escape(comm)}':\s*{{[^}}]*?'mm_net':\s*)-?\d+([^}}]*?'mm_long':\s*)\d+([^}}]*?'mm_short':\s*)\d+([^}}]*?'prod_net':\s*)-?\d+([^}}]*?'swap_net':\s*)-?\d+([^}}]*?'other_net':\s*)-?\d+"
        )
        replacement = (
            rf"\g<1>{p['mm_net']}\g<2>{p['mm_long']}\g<3>{p['mm_short']}"
            rf"\g<4>{p['prod_net']}\g<5>{p['swap_net']}\g<6>{p['other_net']}"
        )
        new_src = pattern.sub(replacement, src)
        if new_src != src:
            src = new_src
            print(f"  ✓ Updated {comm} in app.py")
        else:
            print(f"  ⚠ Could not patch {comm} (pattern not matched — manual update may be needed)")

    with open(app_file, 'w') as f:
        f.write(src)

    print(f"\n✓ app.py updated with COT data through {list(positions.values())[0]['date']}")
    print("  Restart the dashboard to see changes.")

if __name__ == '__main__':
    csv_path = download_latest_cot()
    if csv_path:
        positions = extract_latest_positions(csv_path)
        update_app_with_new_cot(positions)
    print("\nDone.")
